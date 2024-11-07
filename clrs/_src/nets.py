# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""JAX implementation of CLRS basic network."""

import functools

from typing import Dict, List, Optional, Tuple

import chex

from clrs._src import decoders
from clrs._src import encoders
from clrs._src import probing
from clrs._src import processors
from clrs._src import samplers
from clrs._src import specs

import haiku as hk
import jax
import jax.numpy as jnp
import jmp
import time
import types
import warnings

from haiku._src import basic
from haiku._src import initializers
from haiku._src import module
import numpy as np
import copy

# If you are forking replace this with `import haiku as hk`.
# hk = types.ModuleType("haiku")
hk.Module = module.Module
hk.Linear = basic.Linear
hk.transparent = module.transparent
hk.initializers = initializers
del basic, module, initializers

# added by xcy: prepare for mixed precision training

half = jnp.float16
full = jnp.float32

my_policy = jmp.Policy(compute_dtype=half,
                       param_dtype=full,
                       output_dtype=half)

_Array = chex.Array
_DataPoint = probing.DataPoint
_Features = samplers.Features
_FeaturesChunked = samplers.FeaturesChunked
_Location = specs.Location
_Spec = specs.Spec
_Stage = specs.Stage
_Trajectory = samplers.Trajectory
_Type = specs.Type


@chex.dataclass
class _MessagePassingScanState:
    hint_preds: chex.Array
    output_preds: chex.Array
    hiddens: chex.Array
    lstm_state: Optional[hk.LSTMState]
    weight: Optional[chex.Array]


@chex.dataclass
class _MessagePassingOutputChunked:
    hint_preds: chex.Array
    output_preds: chex.Array


@chex.dataclass
class MessagePassingStateChunked:
    inputs: chex.Array
    hints: chex.Array
    is_first: chex.Array
    hint_preds: chex.Array
    hiddens: chex.Array
    lstm_state: Optional[hk.LSTMState]


class Bank_selfAtt(hk.Module):
    def __init__(self, name=None):
        super().__init__(name=name)

    def __call__(self, bank_list):
        # print(bank_list.shape)
        nb_batch = bank_list.shape[0]
        bank_list = jnp.expand_dims(bank_list, axis=-1)  # 30, 128, 1
        inner_Linear = hk.Linear(4)
        bank_list = inner_Linear(bank_list)  # (30, 128, 4)
        # print(bank_list.shape)

        bank_list = jnp.reshape(bank_list, (nb_batch, 128 * 4))  # (30, 512)
        H = 128 * 4
        scale = 1.0 / H

        Wq = hk.Linear(H)
        Wk = hk.Linear(H)
        Wv = hk.Linear(H)

        Q = Wq(bank_list)  # 30, 128 * 4
        K = Wk(bank_list)  # 30, 128 * 4
        V = Wv(bank_list)  # 30, 128 * 4

        K = jnp.transpose(K, (1, 0))
        QK = jnp.matmul(Q, K)  # 30, 30 第一个是查询次数，第二个是被查询的个数，也就是和为1的那个last_dim
        QK = QK * scale
        QK = jax.nn.softmax(QK, axis=-1)
        QKV = jnp.matmul(QK, V)  # 30, 128 * 4

        # 第一组DataPoint之间的注意力做完后，还要加MLP或者别的东西吗？先只让原来的加上，或者用结果直接往后做？先直接加
        QKV = bank_list + QKV
        Ln_dp = hk.LayerNorm(axis=-1, param_axis=-1, create_scale=True, create_offset=True)
        QKV = Ln_dp(QKV)  # 这个LN未来可测一下作用大不大
        # 接下来做 特征之间的注意力。
        DP = jnp.reshape(QKV, (nb_batch, 128, 4))  # (30, 128, 4)

        scale1 = 1.0 / 8
        Wq1 = hk.Linear(8)  # 4->8  QK都是8做点积
        Wk1 = hk.Linear(8)  # 4->8
        Wv1 = hk.Linear(4)  # 4->4

        Q1 = Wq1(DP)  # 30, 128, 8
        K1 = Wk1(DP)  # 30, 128, 8
        V1 = Wv1(DP)  # 30, 128, 4
        K1 = jnp.transpose(K1, (0, 2, 1))  # 30, 8 128
        QK1 = jnp.matmul(Q1, K1)  # 30, 128, 128  # 最后一个128是lastdim，符合我们的设想，拿一组特征当bank，一共128个这样的特征。
        QK1 = QK1 * scale1
        QK1 = jax.nn.softmax(QK1, axis=-1)
        assert QK1.shape == (nb_batch, 128, 128)
        assert V1.shape == (nb_batch, 128, 4)
        QKV1 = jnp.matmul(QK1, V1)  # 30, 128, 4
        # 残差网络 or ...Ln是否要实现？过了最后这个Linear和reshape再实现，因为bank里的数字要搞小一点。
        QKV1 = QKV1 + DP

        out_Linear = hk.Linear(1)
        QKV1 = out_Linear(QKV1)  # 30, 128, 1
        QKV1 = jnp.reshape(QKV1, (nb_batch, 128))
        Ln_bank = hk.LayerNorm(axis=-1, param_axis=-1, create_scale=True, create_offset=True)
        QKV1 = Ln_bank(QKV1)
        return QKV1


class MultiHeadAttention2(hk.Module):
    """Multi-headed attention (MHA) module.

    This module is intended for attending over sequences of vectors.

    Rough sketch:
    - Compute keys (K), queries (Q), and values (V) as projections of inputs.
    - Attention weights are computed as W = softmax(QK^T / sqrt(key_size)).
    - Output is another projection of WV^T.

    For more detail, see the original Transformer paper:
      "Attention is all you need" https://arxiv.org/abs/1706.03762.

    Glossary of shapes:
    - T: Sequence length.
    - D: Vector (embedding) size.
    - H: Number of attention heads.
    """

    def __init__(
            self,
            num_heads: int,
            key_size: int,
            # TODO(b/240019186): Remove `w_init_scale`.
            w_init_scale: Optional[float] = None,
            *,
            w_init: Optional[hk.initializers.Initializer] = None,
            value_size: Optional[int] = None,
            model_size: Optional[int] = None,
            name: Optional[str] = None,
    ):
        """Initialises the module.

        Args:
          num_heads: Number of independent attention heads (H).
          key_size: The size of keys (K) and queries used for attention.
          w_init_scale: DEPRECATED. Please use w_init instead.
          w_init: Initialiser for weights in the linear map. Once `w_init_scale` is
            fully deprecated `w_init` will become mandatory. Until then it has a
            default value of `None` for backwards compatability.
          value_size: Optional size of the value projection (V). If None, defaults
            to the key size (K).
          model_size: Optional size of the output embedding (D'). If None, defaults
            to the key size multiplied by the number of heads (K * H).
          name: Optional name for this module.
        """
        super().__init__(name=name)
        self.num_heads = num_heads
        self.key_size = key_size
        self.value_size = value_size or key_size
        self.model_size = model_size or key_size * num_heads

        # Backwards-compatibility for w_init_scale.

    #        if w_init_scale is not None:
    #            warnings.warn(
    #                "w_init_scale is deprecated; please pass an explicit weight "
    #                "initialiser instead.", DeprecationWarning)
    #        if w_init and w_init_scale:
    #            raise ValueError("Please provide only `w_init`, not `w_init_scale`.")
    #        if w_init is None and w_init_scale is None:
    #            raise ValueError("Please provide a weight initializer: `w_init`. "
    #                             "`w_init` will become mandatory once `w_init_scale` is "
    #                             "fully deprecated.")
    #        if w_init is None:
    #            w_init = hk.initializers.VarianceScaling(w_init_scale)
    #        self.w_init = w_init

    def __call__(
            self,
            query: jnp.ndarray,
            key: jnp.ndarray,
            value: jnp.ndarray,
            mask: Optional[jnp.ndarray] = None,
    ):
        """Computes (optionally masked) MHA with queries, keys & values.

        This module broadcasts over zero or more 'batch-like' leading dimensions.

        Args:
          query: Embeddings sequence used to compute queries; shape [..., T', D_q].
          key: Embeddings sequence used to compute keys; shape [..., T, D_k].
          value: Embeddings sequence used to compute values; shape [..., T, D_v].
          mask: Optional mask applied to attention weights; shape [..., H=1, T', T].

        Returns:
          A new sequence of embeddings, consisting of a projection of the
            attention-weighted value projections; shape [..., T', D'].
        """

        # In shape hints below, we suppress the leading dims [...] for brevity.
        # Hence e.g. [A, B] should be read in every case as [..., A, B].
        *leading_dims, sequence_length, _ = query.shape
        projection = self._linear_projection

        # Compute key/query/values (overload K/Q/V to denote the respective sizes).
        query_heads = projection(query, self.key_size, "query")  # [T', H, Q=K]
        key_heads = projection(key, self.key_size, "key")  # [T, H, K]
        value_heads = projection(value, self.value_size, "value")  # [T, H, V]

        # Compute attention weights.
        attn_logits = jnp.einsum("...thd,...Thd->...htT", query_heads, key_heads)
        attn_logits = attn_logits / np.sqrt(self.key_size).astype(key.dtype)  # 0120 hofer  np.sqrt().astype(key.dtype)
        if mask is not None:
            if mask.ndim != attn_logits.ndim:
                raise ValueError(
                    f"Mask dimensionality {mask.ndim} must match logits dimensionality "
                    f"{attn_logits.ndim}."
                )
            attn_logits = jnp.where(mask, attn_logits, -1e30)

        # x = x.at[idx].set(y)
        # =====hofer2
        # sorted_indices = jnp.argsort(-attn_logits, axis=-1)
        # top_indices = sorted_indices[..., -10:]  # 倒数10个赋值0  到时候改这就好了。
        # if not leading_dims:  # 空列表 说明没有batch，那就是3维的索引
        # attn_logits = attn_logits.at[
        # np.arange(attn_logits.shape[0])[:, None, None], np.arange(attn_logits.shape[1])[:,
        # None], top_indices].set(-1e30)  # 最难写的一步，索引。。
        # else:  # 否则就是4维的索引，含有batch
        # attn_logits = attn_logits.at[
        # np.arange(attn_logits.shape[0])[:, None, None, None], np.arange(attn_logits.shape[1])[:, None,
        # None], np.arange(
        # attn_logits.shape[2])[:, None], top_indices].set(-1e30)
        # ====hofer2
        attn_weights = jax.nn.softmax(attn_logits)  # [H, T', T]  0119hofer
        #        attn_weights = attn_logits# 0119hofer
        #        jax.debug.print('hofer================={x}',x =attn_weights[...,:3])
        # =====hofer
        #        sorted_indices = jnp.argsort(-attn_weights, axis=-1)
        #        top_indices = sorted_indices[..., -5:]  # 倒数5个赋值0  到时候改这就好了。
        #        if not leading_dims:  # 空列表 说明没有batch，那就是3维的索引
        #            attn_weights = attn_weights.at[np.arange(attn_weights.shape[0])[:, None, None],
        #            np.arange(attn_weights.shape[1])[:,None], top_indices].set(0.0)
        #        else:  # 否则就是4维的索引，含有batch
        #            attn_weights = attn_weights.at[np.arange(attn_weights.shape[0])[:, None, None, None],
        #            np.arange(attn_weights.shape[1])[:, None,None],
        #            np.arange(attn_weights.shape[2])[:, None], top_indices].set(0.0)
        # ====hofer
        # Weight the values by the attention and flatten the head vectors.
        attn = jnp.einsum("...htT,...Thd->...thd", attn_weights, value_heads)  # 小t是查询次数，输出为(t,head,v映射的维度)
        attn = jnp.reshape(attn, (*leading_dims, sequence_length, -1))  # [T', H*V]

        # Apply another projection to get the final embeddings.
        final_projection = hk.Linear(self.model_size)
        return final_projection(attn), attn_weights  # [T', D']  #  jnp.array([0.0]

    @hk.transparent
    def _linear_projection(
            self,
            x: jnp.ndarray,
            head_size: int,
            name: Optional[str] = None,
    ) -> jnp.ndarray:
        y = hk.Linear(self.num_heads * head_size, name=name)(x)
        *leading_dims, _ = x.shape
        return y.reshape((*leading_dims, self.num_heads, head_size))


class AttNet(hk.Module):
    def __init__(self, model_size=128, name=None):
        super().__init__(name=name)
        self.Mulatt = MultiHeadAttention2(1, key_size=128, value_size=128,  # 这俩值记得改回128 0521 为了能让quickselect跑
                                          model_size=model_size)

    def __call__(self, x, y, z):
        out, weight = self.Mulatt(x, y, z)  # 要改回去的话，这里要写成 hk.MultiHeadAttention
        # x = hk.LayerNorm(axis=-1, param_axis=-1, create_scale=True, create_offset=True)(x)  # hofer 1226
        # processor模式下 modelsize是128 decoder模式下，对于et和ht不同，一个是256一个是384
        return out, weight
class GateNet(hk.Module):
    def __init__(self,name=None):
        super().__init__(name=name)
        self.o1 = hk.Linear(128)
        self.gate1 = hk.Linear(128)
        self.gate2 = hk.Linear(128)
        self.gate3 = hk.Linear(128, b_init=hk.initializers.Constant(-3))

    def __call__(self, nxt_hidden, msg_node):
        z = self.o1(nxt_hidden)
        ret = z + msg_node
        gate = jax.nn.sigmoid(self.gate3(jax.nn.relu(self.gate1(nxt_hidden) + self.gate2(msg_node))))
        ret = ret * gate + nxt_hidden * (1 - gate)
        return ret

class Net(hk.Module):
    """Building blocks (networks) used to encode and decode messages."""

    def __init__(
            self,
            spec: List[_Spec],
            hidden_dim: int,
            encode_hints: bool,
            decode_hints: bool,
            processor_factory: processors.ProcessorFactory,
            use_lstm: bool,
            encoder_init: str,
            dropout_prob: float,
            hint_teacher_forcing: float,
            hint_repred_mode='soft',
            nb_dims=None,
            nb_msg_passing_steps=1,
            name: str = 'net',
    ):
        """Constructs a `Net`."""
        super().__init__(name=name)

        self._dropout_prob = dropout_prob
        self._hint_teacher_forcing = hint_teacher_forcing
        self._hint_repred_mode = hint_repred_mode
        self.spec = spec
        self.hidden_dim = hidden_dim
        self.encode_hints = encode_hints
        self.decode_hints = decode_hints
        self.processor_factory = processor_factory
        self.nb_dims = nb_dims
        self.use_lstm = use_lstm
        self.encoder_init = encoder_init
        self.nb_msg_passing_steps = nb_msg_passing_steps

    def _msg_passing_step(self,
                          mp_state: _MessagePassingScanState,
                          i: int,
                          hints: List[_DataPoint],
                          repred: bool,
                          lengths: chex.Array,
                          batch_size: int,
                          nb_nodes: int,
                          inputs: _Trajectory,
                          first_step: bool,
                          spec: _Spec,
                          encs: Dict[str, List[hk.Module]],
                          decs: Dict[str, Tuple[hk.Module]],
                          return_hints: bool,
                          return_all_outputs: bool,
                          bank_list: list,
                          Att: Dict[str, hk.Module],
                          gatenet: hk.Module,
                          use_bank: bool
                          ):
        if self.decode_hints and not first_step:
            assert self._hint_repred_mode in ['soft', 'hard', 'hard_on_eval']
            hard_postprocess = (self._hint_repred_mode == 'hard' or
                                (self._hint_repred_mode == 'hard_on_eval' and repred))
            decoded_hint = decoders.postprocess(spec,
                                                mp_state.hint_preds,
                                                sinkhorn_temperature=0.1,
                                                sinkhorn_steps=25,
                                                hard=hard_postprocess)
        if repred and self.decode_hints and not first_step:
            cur_hint = []
            for hint in decoded_hint:
                cur_hint.append(decoded_hint[hint])
        else:
            cur_hint = []
            needs_noise = (self.decode_hints and not first_step and
                           self._hint_teacher_forcing < 1.0)
            if needs_noise:
                # For noisy teacher forcing, choose which examples in the batch to force
                force_mask = jax.random.bernoulli(
                    hk.next_rng_key(), self._hint_teacher_forcing,
                    (batch_size,))
            else:
                force_mask = None
            for hint in hints:
                hint_data = jnp.asarray(hint.data)[i]
                _, loc, typ = spec[hint.name]
                if needs_noise:
                    if (typ == _Type.POINTER and
                            decoded_hint[hint.name].type_ == _Type.SOFT_POINTER):
                        # When using soft pointers, the decoded hints cannot be summarised
                        # as indices (as would happen in hard postprocessing), so we need
                        # to raise the ground-truth hint (potentially used for teacher
                        # forcing) to its one-hot version.
                        hint_data = hk.one_hot(hint_data, nb_nodes)
                        typ = _Type.SOFT_POINTER
                    hint_data = jnp.where(_expand_to(force_mask, hint_data),
                                          hint_data,
                                          decoded_hint[hint.name].data)
                cur_hint.append(
                    probing.DataPoint(
                        name=hint.name, location=loc, type_=typ, data=hint_data))

        hiddens, output_preds_cand, hint_preds, lstm_state, weight = self._one_step_pred(
            inputs, cur_hint, mp_state.hiddens,
            batch_size, nb_nodes, mp_state.lstm_state,
            spec, encs, decs, repred, bank_list, Att, gatenet, use_bank)

        if first_step:
            output_preds = output_preds_cand
        else:
            output_preds = {}
            for outp in mp_state.output_preds:
                is_not_done = _is_not_done_broadcast(lengths, i,
                                                     output_preds_cand[outp])
                output_preds[outp] = is_not_done * output_preds_cand[outp] + (
                        1.0 - is_not_done) * mp_state.output_preds[outp]

        new_mp_state = _MessagePassingScanState(  # pytype: disable=wrong-arg-types  # numpy-scalars
            hint_preds=hint_preds,  # dict
            output_preds=output_preds,  # dict
            hiddens=hiddens,
            lstm_state=lstm_state,
            weight=None)
        # Save memory by not stacking unnecessary fields
        accum_mp_state = _MessagePassingScanState(  # pytype: disable=wrong-arg-types  # numpy-scalars
            hint_preds=hint_preds if return_hints else None,
            output_preds=output_preds if return_all_outputs else None,
            hiddens=None, lstm_state=None, weight=weight)  # 以元组形式传出，但是元组里面的每个叶子都是jnp数组。

        # Complying to jax.scan, the first returned value is the state we carry over
        # the second value is the output that will be stacked over steps.
        return new_mp_state, accum_mp_state

    def __call__(self, features_list: List[_Features], repred: bool,
                 algorithm_index: int,
                 return_hints: bool,
                 return_all_outputs: bool,
                 bank_feed,
                 use_bank):
        """Process one batch of data.

        Args:
          features_list: A list of _Features objects, each with the inputs, hints
            and lengths for a batch o data corresponding to one algorithm.
            The list should have either length 1, at train/evaluation time,
            or length equal to the number of algorithms this Net is meant to
            process, at initialization.
          repred: False during training, when we have access to ground-truth hints.
            True in validation/test mode, when we have to use our own
            hint predictions.
          algorithm_index: Which algorithm is being processed. It can be -1 at
            initialisation (either because we are initialising the parameters of
            the module or because we are intialising the message-passing state),
            meaning that all algorithms should be processed, in which case
            `features_list` should have length equal to the number of specs of
            the Net. Otherwise, `algorithm_index` should be
            between 0 and `length(self.spec) - 1`, meaning only one of the
            algorithms will be processed, and `features_list` should have length 1.
          return_hints: Whether to accumulate and return the predicted hints,
            when they are decoded.
          return_all_outputs: Whether to return the full sequence of outputs, or
            just the last step's output.

        Returns:
          A 2-tuple with (output predictions, hint predictions)
          for the selected algorithm.
        """
        #        print(f'algorithm_index{algorithm_index}')  # hofer 0109
        #        print('=========')

        self.bankencoders = self._construct_encoders()
        node_bank = []
        # edge_bank = []
        # graph_bank = []
        # Knode_bank = []
        bank_list = []
        if algorithm_index == -1:
            algorithm_indices = range(len(features_list))
        else:
            algorithm_indices = [algorithm_index]
        assert len(algorithm_indices) == len(features_list)

        self.encoders, self.decoders = self._construct_single_encoders_decoders(algorithm_index,
                                                                                self.spec[algorithm_index])
        # 0201 hofer 测试原版代码
        # 0203 step=2  hint中dp均为，2*batch*f...
        # use_bank = False
        if use_bank:
            for algo_index, feedback in enumerate(bank_feed):  # 这个for头就不动了
                # 把一个算法的信息聚合成一个128的信息
                inputs = feedback.features.inputs  # bank_batch*f...
                hints = feedback.features.hints  # 原为bank_batch*f...  现为step*bank_batch*f
                # lengths = feedback.features.lengths  # 这句话属是没用 暂时注释掉
                batch_size, nb_nodes = _data_dimensions(feedback.features)
                # nb_mp_steps = 16  # 因为我们这个step现在固定就是16步了

                input_node_fts = jnp.zeros((batch_size, nb_nodes, self.hidden_dim))  # b*n*h  一个样本有n个h，一个节点的特征是h
                input_edge_fts = jnp.zeros(
                    (batch_size, nb_nodes, nb_nodes, self.hidden_dim))  # b*n*n*h  一个边的特征是h，n个点有n*n个边。
                input_graph_fts = jnp.zeros((batch_size, self.hidden_dim))  # bh


                hint0_node_fts = jnp.zeros((batch_size, nb_nodes, self.hidden_dim))  # b*n*h  一个样本有n个h，一个节点的特征是h
                # hint0_edge_fts = jnp.zeros(
                #     (batch_size, nb_nodes, nb_nodes, self.hidden_dim))  # b*n*n*h  一个边的特征是h，n个点有n*n个边。
                # hint0_graph_fts = jnp.zeros((batch_size, self.hidden_dim))  # bh

                encs = self.bankencoders[algo_index]
                # if algo_index == algorithm_index:
                #     encs = self.encoders[0]

                # 对input进行encoder
                for newdp in inputs:
                    # data类型可能有bug
                    newdp = encoders.preprocess(newdp, nb_nodes)
                    assert newdp.type_ != _Type.SOFT_POINTER
                    encoder = encs[newdp.name]
                    input_node_fts = encoders.accum_node_fts(encoder, newdp, input_node_fts)  # bnh
                    input_edge_fts = encoders.accum_edge_fts(encoder, newdp, input_edge_fts)  # bnnh
                    input_graph_fts = encoders.accum_graph_fts(encoder, newdp, input_graph_fts)  # bh
                # 对hint的第一步进行encoder，通过更改accum的实现做到的，结束后和input内容相加。
                for dp in hints:
                    dp = encoders.preprocess(dp, nb_nodes)  # 带着s维度去做这个hk.onehot会不会有问题？ 没问题应该这个功能是把每个数变成了一个向量
                    assert dp.type_ != _Type.SOFT_POINTER
                    encoder = encs[dp.name]  # 下面都调用的是_step0 ，只会处理第0步的信息。a
                    hint0_node_fts = encoders.accum_node_fts_step0(encoder, dp, hint0_node_fts)  # bnh
                    # hint0_edge_fts = encoders.accum_edge_fts_step0(encoder, dp, hint0_edge_fts)  # bnnh  dp.name dp.location node  dp.data.shape  step*batch*node
                    # hint0_graph_fts = encoders.accum_graph_fts_step0(encoder, dp, hint0_graph_fts)  # bh

                hint0_node_fts += input_node_fts  # bnh
                # hint0_edge_fts += input_edge_fts  # bnnh  这个边的和图的暂且先不用了。
                # hint0_graph_fts += input_graph_fts  # bh
                # hint0_node_fts = jnp.sum(hint0_node_fts, axis=1) / nb_nodes
                # 随机切片
                rand_node = np.random.randint(0, nb_nodes)
                hint0_node_fts = hint0_node_fts[:,rand_node] # bh
                
                assert hint0_node_fts.shape == (batch_size, 128)

                # 对全部的hints进行encoder
                for dp in hints:
                    lengths = feedback.features.lengths
                    tempdata = jnp.sum(dp.data, axis=0)
                    dp.data = tempdata / _iso_expand_to(lengths, tempdata)
                    dp = encoders.preprocess(dp, nb_nodes)  # 带着s维度去做这个hk.onehot会不会有问题？ 没问题应该这个功能是把每个数变成了一个向量
                    assert dp.type_ != _Type.SOFT_POINTER
                    encoder = encs[dp.name]  # 从字典中索引出这个name对应的 “encoder们”   encoder是个list。
                    input_node_fts = encoders.accum_node_fts(encoder, dp, input_node_fts)  # bnh
                    input_edge_fts = encoders.accum_edge_fts(encoder, dp, input_edge_fts)  # bnnh
                    input_graph_fts = encoders.accum_graph_fts(encoder, dp, input_graph_fts)  # bh

                # input_node_fts = input_node_fts[:, rand_node]  # bnh->bh
                input_node_fts = jnp.sum(input_node_fts, axis=1)  # bnh->bh
                input_edge_fts = jnp.sum(input_edge_fts, axis=(1, 2))  # bnnh->bh
                # input_graph_fts = jnp.sum(input_graph_fts, axis=(0))  # bh->bh
                # hofer 0117
                #            one_hot_vector = jnp.zeros(bank_length)
                #            one_hot_vector = one_hot_vector.at[algo_index].set(1.0)
                #            pos_embedding = pos_Linear(one_hot_vector)
                #            node_bank.append(input_node_fts / (batch_size * nb_nodes) + pos_embedding)
                #            edge_bank.append(input_edge_fts / batch_size + pos_embedding)
                #            graph_bank.append(input_graph_fts / batch_size + pos_embedding)
                # hofer 0117
                # before 0117
                temp = input_node_fts / (nb_nodes)
                # temp = input_node_fts
                #            temp = jax.nn.tanh(nodehint_Linear(temp))
                #            temp = jax.nn.relu(temp)
                temp = jax.nn.tanh(temp)
                for i in range(temp.shape[0]):
                    t = temp[i]
                    node_bank.append(t)


        bank_list.append(node_bank)  # 0


        # hofer 0120
        # 创建3个ATT网络  change by hofer 12.11
        # 不该是3个，而是每个算法对应一套。  # hofer 1226
        qkv_list = []
        for i in [1]:
            node_Attention1 = AttNet(model_size=128, name='node_Q_Attbank')
            Att = {'node1': node_Attention1}
            qkv_list.append(Att)

        gate_list = []
        for i in range(1):
            gatenet = GateNet(name = f'gate_algo_{i}_')
            gate_list.append(gatenet)
        #     for enc in self.encoders:
        #         for networks in enc.values():
        #             for network in networks:
        #                 hk.mixed_precision.set_policy(network,my_policy)
        #     for dec in self.decoders:
        #         for networks in dec.values():
        #             for network in networks:
        #                 hk.mixed_precision.set_policy(network,my_policy)
        # hk.mixed_precision.set_policy(dec,my_policy)
        self.processor = self.processor_factory(self.hidden_dim)
        # hk.mixed_precision.set_policy(self.processor,my_policy)

        # Optionally construct LSTM.
        if self.use_lstm:
            self.lstm = hk.LSTM(
                hidden_size=self.hidden_dim,
                name='processor_lstm')
            lstm_init = self.lstm.initial_state
        else:
            self.lstm = None
            lstm_init = lambda x: 0

        for algorithm_index, features in zip(algorithm_indices, features_list):
            inputs = features.inputs
            hints = features.hints
            lengths = features.lengths

            batch_size, nb_nodes = _data_dimensions(features)
            #            print(f'batch_size:{batch_size}, nb_nodes:{nb_nodes}')

            nb_mp_steps = max(1, hints[0].data.shape[0] - 1)
            hiddens = jnp.zeros((batch_size, nb_nodes, self.hidden_dim))

            if self.use_lstm:
                lstm_state = lstm_init(batch_size * nb_nodes)
                lstm_state = jax.tree_util.tree_map(
                    lambda x, b=batch_size, n=nb_nodes: jnp.reshape(x, [b, n, -1]),
                    lstm_state)
            else:
                lstm_state = None

            mp_state = _MessagePassingScanState(  # pytype: disable=wrong-arg-types  # numpy-scalars
                hint_preds=None, output_preds=None,
                hiddens=hiddens, lstm_state=lstm_state, weight=None)

            # Do the first step outside of the scan because it has a different
            # computation graph.
            common_args = dict(
                hints=hints,
                repred=repred,
                inputs=inputs,
                batch_size=batch_size,
                nb_nodes=nb_nodes,
                lengths=lengths,
                spec=self.spec[algorithm_index],
                encs=self.encoders[0],
                decs=self.decoders[0],
                return_hints=return_hints,
                return_all_outputs=return_all_outputs,
                bank_list=bank_list,
                Att=qkv_list[0],  # 1226
                gatenet=gate_list[0],
                use_bank=use_bank
            )
            mp_state, lean_mp_state = self._msg_passing_step(
                mp_state,
                i=0,
                first_step=True,
                **common_args)

            # Then scan through the rest.
            scan_fn = functools.partial(
                self._msg_passing_step,
                first_step=False,
                **common_args)

            output_mp_state, accum_mp_state = hk.scan(
                scan_fn,
                mp_state,
                jnp.arange(nb_mp_steps - 1) + 1,
                length=nb_mp_steps - 1)

        # We only return the last algorithm's output. That's because
        # the output only matters when a single algorithm is processed; the case
        # `algorithm_index==-1` (meaning all algorithms should be processed)
        # is used only to init parameters.
        accum_mp_state = jax.tree_util.tree_map(
            lambda init, tail: jnp.concatenate([init[None], tail], axis=0),
            lean_mp_state, accum_mp_state)

        def invert(d):
            """Dict of lists -> list of dicts."""
            if d:
                return [dict(zip(d, i)) for i in zip(*d.values())]

        if return_all_outputs:
            output_preds = {k: jnp.stack(v)
                            for k, v in accum_mp_state.output_preds.items()}
        else:
            output_preds = output_mp_state.output_preds
        hint_preds = invert(accum_mp_state.hint_preds)
        # weight维度:node节点：step步数*8*node个数*30  对于edge和graph：step*8*batch*30
        return output_preds, hint_preds, accum_mp_state.weight

    def _construct_encoders(self):
        """Constructs encoders , separate for each algorithm."""
        encoders_ = []
        enc_algo_idx = None
        for (algo_idx, spec) in enumerate(
                self.spec):  # 30个算法的list，每个元素对应一个算法的Spec字典Dict[strname:(str,str,str)]  hofer 1221
            enc = {}
            #            dec = {}
            for name, (stage, loc, t) in spec.items():
                if stage == _Stage.INPUT or (
                        stage == _Stage.HINT and self.encode_hints):
                    # Build input encoders.
                    if name == specs.ALGO_IDX_INPUT_NAME:  # 不知道这条分支的意义，ALGO_IDX_INPUT_NAME = 'algo_idx'。没有谁的名字等于这个吧。。
                        if enc_algo_idx is None:
                            enc_algo_idx = [hk.Linear(self.hidden_dim,
                                                      name=f'{name}_enc_linear_bank')]
                        enc[name] = enc_algo_idx
                    else:  # 给enc字典里加键值对，key是代表hint或者input的name，value是对应的encoder网络列表，当然大部分人的这个列表长度就是1。
                        enc[name] = encoders.construct_encoders(  # 返回值是个列表，代表这个'i' 'pos'对应的encoder们，有的可能对应俩线性层。
                            # 返回的列表又成了enc字典的键值对。
                            stage, loc, t, hidden_dim=self.hidden_dim,
                            init=self.encoder_init,
                            name=f'algo_{algo_idx}_{name}_bank')  # algo_0_j   algo_15_pred  诸如此类

            encoders_.append(enc)

        return encoders_  # 返回了俩list，list里面有30个字典。字典里是 name : list or tuple这样的键值对
        # 这里的list和tuple装的是这个name对应的encoder或者decoder需要的网络们

    def _construct_encoders_decoders(self):
        """Constructs encoders and decoders, separate for each algorithm."""
        encoders_ = []
        decoders_ = []
        enc_algo_idx = None

        for (algo_idx, spec) in enumerate(self.spec):  # algo_idx, spec这个改成参数，然后for循环去了
            enc = {}
            dec = {}
            for name, (stage, loc, t) in spec.items():
                if stage == _Stage.INPUT or (
                        stage == _Stage.HINT and self.encode_hints):
                    # Build input encoders.
                    if name == specs.ALGO_IDX_INPUT_NAME:
                        if enc_algo_idx is None:
                            enc_algo_idx = [hk.Linear(self.hidden_dim,
                                                      name=f'{name}_enc_linear')]
                        enc[name] = enc_algo_idx
                    else:
                        enc[name] = encoders.construct_encoders(
                            stage, loc, t, hidden_dim=self.hidden_dim,
                            init=self.encoder_init,
                            name=f'algo_{algo_idx}_{name}')

                if stage == _Stage.OUTPUT or (
                        stage == _Stage.HINT and self.decode_hints):
                    # Build output decoders.

                    dec[name] = decoders.construct_decoders(
                        loc, t, hidden_dim=self.hidden_dim,
                        nb_dims=self.nb_dims[algo_idx][name],
                        name=f'algo_{algo_idx}_{name}')

            encoders_.append(enc)
            decoders_.append(dec)

        return encoders_, decoders_

    def _construct_single_encoders_decoders(self, algo_idx, spec):
        """Constructs encoders and decoders, separate for each algorithm."""
        encoders_ = []
        decoders_ = []
        enc_algo_idx = None

        # for (algo_idx, spec) in enumerate(self.spec):  # algo_idx, spec这个改成参数，然后for循环去了
        enc = {}
        dec = {}
        for name, (stage, loc, t) in spec.items():
            if stage == _Stage.INPUT or (
                    stage == _Stage.HINT and self.encode_hints):
                # Build input encoders.
                if name == specs.ALGO_IDX_INPUT_NAME:
                    if enc_algo_idx is None:
                        enc_algo_idx = [hk.Linear(self.hidden_dim,
                                                  name=f'{name}_enc_linear')]
                    enc[name] = enc_algo_idx
                else:
                    enc[name] = encoders.construct_encoders(
                        stage, loc, t, hidden_dim=self.hidden_dim,
                        init=self.encoder_init,
                        name=f'algo_{algo_idx}_{name}')

            if stage == _Stage.OUTPUT or (
                    stage == _Stage.HINT and self.decode_hints):
                # Build output decoders.

                dec[name] = decoders.construct_decoders(
                    loc, t, hidden_dim=self.hidden_dim,
                    nb_dims=self.nb_dims[algo_idx][name],
                    name=f'algo_{algo_idx}_{name}')

        encoders_.append(enc)
        decoders_.append(dec)

        return encoders_, decoders_

    def _one_step_pred(
            self,
            inputs: _Trajectory,
            hints: _Trajectory,
            hidden: _Array,
            batch_size: int,
            nb_nodes: int,
            lstm_state: Optional[hk.LSTMState],
            spec: _Spec,
            encs: Dict[str, List[hk.Module]],
            decs: Dict[str, Tuple[hk.Module]],
            repred: bool,
            bank_list: list,
            Att: dict,
            gatenet: hk.Module,
            use_bank: bool
    ):
        """Generates one-step predictions."""

        # Initialise empty node/edge/graph features and adjacency matrix.
        node_fts = jnp.zeros((batch_size, nb_nodes, self.hidden_dim))
        edge_fts = jnp.zeros((batch_size, nb_nodes, nb_nodes, self.hidden_dim))
        graph_fts = jnp.zeros((batch_size, self.hidden_dim))
        adj_mat = jnp.repeat(
            jnp.expand_dims(jnp.eye(nb_nodes), 0), batch_size, axis=0)

        # ENCODE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Encode node/edge/graph features from inputs and (optionally) hints.
        trajectories = [inputs]
        if self.encode_hints:
            trajectories.append(hints)

        for trajectory in trajectories:
            for dp in trajectory:
                try:
                    dp = encoders.preprocess(dp, nb_nodes)
                    assert dp.type_ != _Type.SOFT_POINTER
                    adj_mat = encoders.accum_adj_mat(dp, adj_mat)
                    encoder = encs[dp.name]
                    edge_fts = encoders.accum_edge_fts(encoder, dp, edge_fts)  # bnnh
                    node_fts = encoders.accum_node_fts(encoder, dp, node_fts)  # bnh
                    graph_fts = encoders.accum_graph_fts(encoder, dp, graph_fts)  # bh
                except Exception as e:
                    raise Exception(f'Failed to process {dp}') from e

        # decoderbank的输入  input+当前的hint，当前的hint就相当于是某个新算法开始的状态。
        query_input = node_fts.copy()  # bnh
        # query_input = np.sum(query_input, axis=1)

        # 自注意力网络引进，lookup出额外的特征。
        # nodebank_list = jnp.stack(bank_list[0])
        # edgebank_list = jnp.stack(bank_list[1])
        # graphbank_list = jnp.stack(bank_list[2])
        #
        # node_input = jnp.sum(node_fts, axis=(1))  # bh
        # edge_input = jnp.sum(edge_fts, axis=(1, 2))  # bh
        # graph_input = graph_fts  # bh
        #
        # node_res = Att['node'](node_input, nodebank_list)  # bh
        # edge_res = Att['edge'](edge_input, edgebank_list)
        # graph_res = Att['graph'](graph_input, graphbank_list)
        #
        # # 简单的广播方式相加
        #
        # node_fts += jnp.expand_dims(node_res, 1)  # bnh + bh扩维
        # edge_fts += jnp.expand_dims(edge_res, (1, 2))  # bnnh  +bh扩维
        # graph_fts += graph_res  # bh

        # PROCESS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        nxt_hidden = hidden
        for _ in range(self.nb_msg_passing_steps):
            nxt_hidden, nxt_edge, weight = self.processor(  # weight是个元组，包含了一步的多个权重信息
                node_fts,
                edge_fts,
                graph_fts,
                adj_mat,
                nxt_hidden,
                Att,
                bank_list,
                batch_size=batch_size,
                nb_nodes=nb_nodes,
                use_bank=use_bank
            )

        if not repred:  # dropout only on training
            nxt_hidden = hk.dropout(hk.next_rng_key(), self._dropout_prob, nxt_hidden)

        if self.use_lstm:
            # lstm doesn't accept multiple batch dimensions (in our case, batch and
            # nodes), so we vmap over the (first) batch dimension.
            nxt_hidden, nxt_lstm_state = jax.vmap(self.lstm)(nxt_hidden, lstm_state)
        else:
            nxt_lstm_state = None

        

        nodebank_list = jnp.stack(bank_list[0])
        # edgebank_list = jnp.stack(bank_list[1])  # 取出边和图的bank，shape = 30*h 当作ATT的y使用。
        # graphbank_list = jnp.stack(bank_list[2])
        ln1 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        # ln2 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        # ln3 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        ln4 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        nodebank_list = ln1(nodebank_list)
        # edgebank_list = ln2(edgebank_list)
        # graphbank_list = ln3(graphbank_list)

        
        # nodeBankAttNet = Bank_selfAtt(name='nodebank_de')
        # edgeBankAttNet = Bank_selfAtt(name='edgebank_de')
        # graphBankAttNet = Bank_selfAtt(name='graphbank_de')
        
        # nodebank_list = nodeBankAttNet(nodebank_list)
        
        # edgebank_list = edgeBankAttNet(edgebank_list)
        # graphbank_list = graphBankAttNet(graphbank_list)
        # gate_node = hk.Linear(h_t.shape[-1], name='gate_node')

        if nxt_edge is not None:
            e_t = jnp.concatenate([edge_fts, nxt_edge], axis=-1)
        else:
            e_t = edge_fts

        # 第一个实现版本 到bank里查询并相加1
        # msg_node, weight1 = Att['node1'](h_t, nodebank_list)  # bn 3h
        # msg_edge, _ = Att['edge'](e_t, edgebank_list)  # bnnh  这个b进去之后就成了length了，里面就没batch这一说了。
        # msg_graph, _ = Att['graph'](graph_fts, graphbank_list)  # bh

        # 第二个实现版本 用input+当前步的hint去input+hint[0]的里面查   即用query_input  到里Knodebank_list，V值用nodebank_list
        # Q:bnh  K 240 h V 240 h  ->msg_node :bn3h
        #hofer 20240423

        
        # hofer 20240423 把nxt_hidden bnh加入到bank中，方法是对n进行聚合，因为bank里是b个h，已经对每个样本的node进行了聚合；试试能不能降低方差
        # 以下为20240423前的bank注意力代码
#        msg_node, weight1 = Att['node1'](nxt_hidden, nodebank_list, nodebank_list)  #bnh,?,?

# 以下为了降低方差
        nodebank_list = np.repeat(nodebank_list[np.newaxis, ...] ,nxt_hidden.shape[0], axis=0)
        qself = jnp.sum(nxt_hidden,axis=1,keepdims = True)/nxt_hidden.shape[1];
        nodebank_list = jnp.concatenate((qself,nodebank_list),axis = 1)
        msg_node, weight1 = Att['node1'](nxt_hidden, nodebank_list, nodebank_list)
# 从这往上是2024 0423 hofer 改之后的代码，为了降低方差。

        # 这里的weight拿出来，放在onestep的返回值里。
        weight = weight1
        ret = gatenet(nxt_hidden, msg_node)
        # nxt_hidden = ret    # nxt_hidden不要动。
        
        h_t = jnp.concatenate([node_fts, hidden, ret], axis=-1)  # 打印一下这个shape，我猜是 (b, n, 128*3)
        # assert h_t.shape == (batch_size, nb_nodes, 128 * 3)

        # DECODE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Decode features and (optionally) hints.
        hint_preds, output_preds = decoders.decode_fts(
            decoders=decs,
            spec=spec,
            h_t=h_t,
            adj_mat=adj_mat,
            edge_fts=e_t,
            graph_fts=graph_fts,
            inf_bias=self.processor.inf_bias,
            inf_bias_edge=self.processor.inf_bias_edge,
            repred=repred,
        )

        return nxt_hidden, output_preds, hint_preds, nxt_lstm_state, weight


class NetChunked(Net):
    """A Net that will process time-chunked data instead of full samples."""

    def _msg_passing_step(self,
                          mp_state: MessagePassingStateChunked,
                          xs,
                          repred: bool,
                          init_mp_state: bool,
                          batch_size: int,
                          nb_nodes: int,
                          spec: _Spec,
                          encs: Dict[str, List[hk.Module]],
                          decs: Dict[str, Tuple[hk.Module]],
                          bank_list: list,
                          Att: Dict[str, hk.Module],
                          gatenet: hk.Module,
                          use_bank: bool
                          ):
        """Perform one message passing step.

        This function is unrolled along the time axis to process a data chunk.

        Args:
          mp_state: message-passing state. Includes the inputs, hints,
            beginning-of-sample markers, hint predictions, hidden and lstm state
            to be used for prediction in the current step.
          xs: A 3-tuple of with the next timestep's inputs, hints, and
            beginning-of-sample markers. These will replace the contents of
            the `mp_state` at the output, in readiness for the next unroll step of
            the chunk (or the first step of the next chunk). Besides, the next
            timestep's hints are necessary to compute diffs when `decode_diffs`
            is True.
          repred: False during training, when we have access to ground-truth hints.
            True in validation/test mode, when we have to use our own
            hint predictions.
          init_mp_state: Indicates if we are calling the method just to initialise
            the message-passing state, before the beginning of training or
            validation.
          batch_size: Size of batch dimension.
          nb_nodes: Number of nodes in graph.
          spec: The spec of the algorithm being processed.
          encs: encoders for the algorithm being processed.
          decs: decoders for the algorithm being processed.
        Returns:
          A 2-tuple with the next mp_state and an output consisting of
          hint predictions and output predictions.
        """

        def _as_prediction_data(hint):
            if hint.type_ == _Type.POINTER:
                return hk.one_hot(hint.data, nb_nodes)
            return hint.data

        nxt_inputs, nxt_hints, nxt_is_first = xs
        inputs = mp_state.inputs
        is_first = mp_state.is_first
        hints = mp_state.hints
        if init_mp_state:
            prev_hint_preds = {h.name: _as_prediction_data(h) for h in hints}
            hints_for_pred = hints
        else:
            prev_hint_preds = mp_state.hint_preds
            if self.decode_hints:
                if repred:
                    force_mask = jnp.zeros(batch_size, dtype=bool)
                elif self._hint_teacher_forcing == 1.0:
                    force_mask = jnp.ones(batch_size, dtype=bool)
                else:
                    force_mask = jax.random.bernoulli(
                        hk.next_rng_key(), self._hint_teacher_forcing,
                        (batch_size,))
                assert self._hint_repred_mode in ['soft', 'hard', 'hard_on_eval']
                hard_postprocess = (
                        self._hint_repred_mode == 'hard' or
                        (self._hint_repred_mode == 'hard_on_eval' and repred))
                decoded_hints = decoders.postprocess(spec,
                                                     prev_hint_preds,
                                                     sinkhorn_temperature=0.1,
                                                     sinkhorn_steps=25,
                                                     hard=hard_postprocess)
                hints_for_pred = []
                for h in hints:
                    typ = h.type_
                    hint_data = h.data
                    if (typ == _Type.POINTER and
                            decoded_hints[h.name].type_ == _Type.SOFT_POINTER):
                        hint_data = hk.one_hot(hint_data, nb_nodes)
                        typ = _Type.SOFT_POINTER
                    hints_for_pred.append(probing.DataPoint(
                        name=h.name, location=h.location, type_=typ,
                        data=jnp.where(_expand_to(is_first | force_mask, hint_data),
                                       hint_data, decoded_hints[h.name].data)))
            else:
                hints_for_pred = hints

        hiddens = jnp.where(is_first[..., None, None], 0.0, mp_state.hiddens)
        if self.use_lstm:
            lstm_state = jax.tree_util.tree_map(
                lambda x: jnp.where(is_first[..., None, None], 0.0, x),
                mp_state.lstm_state)
        else:
            lstm_state = None
        hiddens, output_preds, hint_preds, lstm_state, _ = self._one_step_pred(  # 最后一个_是weight，但是训练时不用，所以不接受
            inputs, hints_for_pred, hiddens,
            batch_size, nb_nodes, lstm_state,
            spec, encs, decs, repred, bank_list, Att, gatenet,use_bank)

        new_mp_state = MessagePassingStateChunked(  # pytype: disable=wrong-arg-types  # numpy-scalars
            hiddens=hiddens, lstm_state=lstm_state, hint_preds=hint_preds,
            inputs=nxt_inputs, hints=nxt_hints, is_first=nxt_is_first)
        mp_output = _MessagePassingOutputChunked(  # pytype: disable=wrong-arg-types  # numpy-scalars
            hint_preds=hint_preds,
            output_preds=output_preds)
        return new_mp_state, mp_output

    def __call__(self, features_list: List[_FeaturesChunked],
                 mp_state_list: List[MessagePassingStateChunked],
                 repred: bool, init_mp_state: bool,
                 algorithm_index: int,
                 bank_feed,
                 use_bank):
        """Process one chunk of data.

        Args:
          features_list: A list of _FeaturesChunked objects, each with the
            inputs, hints and beginning- and end-of-sample markers for
            a chunk (i.e., fixed time length) of data corresponding to one
            algorithm. All features are expected
            to have dimensions chunk_length x batch_size x ...
            The list should have either length 1, at train/evaluation time,
            or length equal to the number of algorithms this Net is meant to
            process, at initialization.
          mp_state_list: list of message-passing states. Each message-passing state
            includes the inputs, hints, beginning-of-sample markers,
            hint prediction, hidden and lstm state from the end of the previous
            chunk, for one algorithm. The length of the list should be the same
            as the length of `features_list`.
          repred: False during training, when we have access to ground-truth hints.
            True in validation/test mode, when we have to use our own hint
            predictions.
          init_mp_state: Indicates if we are calling the network just to initialise
            the message-passing state, before the beginning of training or
            validation. If True, `algorithm_index` (see below) must be -1 in order
            to initialize the message-passing state of all algorithms.
          algorithm_index: Which algorithm is being processed. It can be -1 at
            initialisation (either because we are initialising the parameters of
            the module or because we are intialising the message-passing state),
            meaning that all algorithms should be processed, in which case
            `features_list` and `mp_state_list` should have length equal to the
            number of specs of the Net. Otherwise, `algorithm_index` should be
            between 0 and `length(self.spec) - 1`, meaning only one of the
            algorithms will be processed, and `features_list` and `mp_state_list`
            should have length 1.

        Returns:
          A 2-tuple consisting of:
          - A 2-tuple with (output predictions, hint predictions)
            for the selected algorithm. Each of these has
            chunk_length x batch_size x ... data, where the first time
            slice contains outputs for the mp_state
            that was passed as input, and the last time slice contains outputs
            for the next-to-last slice of the input features. The outputs that
            correspond to the final time slice of the input features will be
            calculated when the next chunk is processed, using the data in the
            mp_state returned here (see below). If `init_mp_state` is True,
            we return None instead of the 2-tuple.
          - The mp_state (message-passing state) for the next chunk of data
            of the selected algorithm. If `init_mp_state` is True, we return
            initial mp states for all the algorithms.
        """

        self.bankencoders = self._construct_encoders()

        node_bank = []
        bank_list = []  # 0 1 2 点 边 图bank
        if algorithm_index == -1:
            algorithm_indices = range(len(features_list))
        else:
            algorithm_indices = [algorithm_index]
        assert len(algorithm_indices) == len(features_list)

        self.encoders, self.decoders = self._construct_single_encoders_decoders(algorithm_index,
                                                                                self.spec[algorithm_index])
        # 0201 hofer 测试原版代码
        # use_bank = False
        if use_bank:
            for algo_index, feedback in enumerate(bank_feed):  # 这个for头就不动了
                # 把一个算法的信息聚合成一个128的信息
                inputs = feedback.features.inputs  # bank_batch*f...
                hints = feedback.features.hints  # 原为bank_batch*f...  现为step*bank_batch*f
                # lengths = feedback.features.lengths  # 这句话属是没用 暂时注释掉
                batch_size, nb_nodes = _data_dimensions(feedback.features)
                # nb_mp_steps = 16  # 因为我们这个step现在固定就是16步了

                input_node_fts = jnp.zeros((batch_size, nb_nodes, self.hidden_dim))  # b*n*h  一个样本有n个h，一个节点的特征是h
                input_edge_fts = jnp.zeros(
                    (batch_size, nb_nodes, nb_nodes, self.hidden_dim))  # b*n*n*h  一个边的特征是h，n个点有n*n个边。
                input_graph_fts = jnp.zeros((batch_size, self.hidden_dim))  # bh

                hint0_node_fts = jnp.zeros((batch_size, nb_nodes, self.hidden_dim))  # b*n*h  一个样本有n个h，一个节点的特征是h
                # hint0_edge_fts = jnp.zeros(
                #     (batch_size, nb_nodes, nb_nodes, self.hidden_dim))  # b*n*n*h  一个边的特征是h，n个点有n*n个边。
                # hint0_graph_fts = jnp.zeros((batch_size, self.hidden_dim))  # bh

                encs = self.bankencoders[algo_index]
                # if algo_index == algorithm_index:
                #     encs = self.encoders[0]

                # 对input进行encoder
                for newdp in inputs:
                    # data类型可能有bug
                    newdp = encoders.preprocess(newdp, nb_nodes)
                    assert newdp.type_ != _Type.SOFT_POINTER
                    encoder = encs[newdp.name]
                    input_node_fts = encoders.accum_node_fts(encoder, newdp, input_node_fts)  # bnh
                    input_edge_fts = encoders.accum_edge_fts(encoder, newdp, input_edge_fts)  # bnnh
                    input_graph_fts = encoders.accum_graph_fts(encoder, newdp, input_graph_fts)  # bh
                # 对hint的第一步进行encoder，通过更改accum的实现做到的，结束后和input内容相加。
                for dp in hints:
                    dp = encoders.preprocess(dp, nb_nodes)  # 带着s维度去做这个hk.onehot会不会有问题？ 没问题应该这个功能是把每个数变成了一个向量
                    assert dp.type_ != _Type.SOFT_POINTER
                    encoder = encs[dp.name]  # 下面都调用的是_step0 ，只会处理第0步的信息。a
                    hint0_node_fts = encoders.accum_node_fts_step0(encoder, dp, hint0_node_fts)  # bnh
                    # hint0_edge_fts = encoders.accum_edge_fts_step0(encoder, dp, hint0_edge_fts)  # bnnh  dp.name dp.location node  dp.data.shape  step*batch*node
                    # hint0_graph_fts = encoders.accum_graph_fts_step0(encoder, dp, hint0_graph_fts)  # bh

                hint0_node_fts += input_node_fts  # bnh
                # hint0_edge_fts += input_edge_fts  # bnnh  这个边的和图的暂且先不用了。
                # hint0_graph_fts += input_graph_fts  # bh
                # hint0_node_fts = jnp.sum(hint0_node_fts, axis=1) / nb_nodes
                # 随机切片
                rand_node = np.random.randint(0, nb_nodes)
                hint0_node_fts = hint0_node_fts[:, rand_node]  # bh
                assert hint0_node_fts.shape == (batch_size, 128)


                # 对全部的hints进行encoder
                for dp in hints:
                    lengths = feedback.features.lengths
                    tempdata = jnp.sum(dp.data, axis=0)
                    dp.data = tempdata / _iso_expand_to(lengths, tempdata)
                    dp = encoders.preprocess(dp, nb_nodes)  # 带着s维度去做这个hk.onehot会不会有问题？ 没问题应该这个功能是把每个数变成了一个向量
                    assert dp.type_ != _Type.SOFT_POINTER
                    encoder = encs[dp.name]  # 从字典中索引出这个name对应的 “encoder们”   encoder是个list。
                    input_node_fts = encoders.accum_node_fts(encoder, dp, input_node_fts)  # bnh
                    input_edge_fts = encoders.accum_edge_fts(encoder, dp, input_edge_fts)  # bnnh
                    input_graph_fts = encoders.accum_graph_fts(encoder, dp, input_graph_fts)  # bh

                # input_node_fts = input_node_fts[:, rand_node]  # bnh->bh
                input_node_fts = jnp.sum(input_node_fts, axis=1)  # bnh->bh
                input_edge_fts = jnp.sum(input_edge_fts, axis=(1, 2))  # bnnh->bh
                # input_graph_fts = jnp.sum(input_graph_fts, axis=(0))  # bh->bh
                # hofer 0117
                #            one_hot_vector = jnp.zeros(bank_length)
                #            one_hot_vector = one_hot_vector.at[algo_index].set(1.0)
                #            pos_embedding = pos_Linear(one_hot_vector)
                #            node_bank.append(input_node_fts / (batch_size * nb_nodes) + pos_embedding)
                #            edge_bank.append(input_edge_fts / batch_size + pos_embedding)
                #            graph_bank.append(input_graph_fts / batch_size + pos_embedding)
                # hofer 0117
                # before 0117
                temp = input_node_fts / (nb_nodes)
                # temp = input_node_fts
                #            temp = jax.nn.tanh(nodehint_Linear(temp))
                #            temp = jax.nn.relu(temp)
                temp = jax.nn.tanh(temp)
                for i in range(temp.shape[0]):
                    t = temp[i]
                    node_bank.append(t)

                # before 0117

        #  有关bank的一切结束。
        bank_list.append(node_bank)  # 0

        # 创建3个ATT网络  change by hofer 12.11
        # 不该是3个，而是每个算法对应一套。  # hofer 1226
        qkv_list = []
        for i in [1]:
            node_Attention1 = AttNet(model_size=128, name='node_Q_Attbank')
            Att = {'node1': node_Attention1}
            qkv_list.append(Att)

        gate_list = []
        for i in range(1):
            gatenet = GateNet(name=f'gate_algo_{i}_')
            gate_list.append(gatenet)

        self.processor = self.processor_factory(self.hidden_dim)
        # Optionally construct LSTM.
        if self.use_lstm:
            self.lstm = hk.LSTM(
                hidden_size=self.hidden_dim,
                name='processor_lstm')
            lstm_init = self.lstm.initial_state
        else:
            self.lstm = None
            lstm_init = lambda x: 0

        if init_mp_state:
            output_mp_states = []
            for algorithm_index, features, mp_state in zip(
                    algorithm_indices, features_list, mp_state_list):
                inputs = features.inputs
                hints = features.hints
                batch_size, nb_nodes = _data_dimensions_chunked(features)

                if self.use_lstm:
                    lstm_state = lstm_init(batch_size * nb_nodes)
                    lstm_state = jax.tree_util.tree_map(
                        lambda x, b=batch_size, n=nb_nodes: jnp.reshape(x, [b, n, -1]),
                        lstm_state)
                    mp_state.lstm_state = lstm_state
                mp_state.inputs = jax.tree_util.tree_map(lambda x: x[0], inputs)
                mp_state.hints = jax.tree_util.tree_map(lambda x: x[0], hints)
                mp_state.is_first = jnp.zeros(batch_size, dtype=int)
                mp_state.hiddens = jnp.zeros((batch_size, nb_nodes, self.hidden_dim))
                next_is_first = jnp.ones(batch_size, dtype=int)

                mp_state, _ = self._msg_passing_step(
                    mp_state,
                    (mp_state.inputs, mp_state.hints, next_is_first),
                    repred=repred,
                    init_mp_state=True,
                    batch_size=batch_size,
                    nb_nodes=nb_nodes,
                    spec=self.spec[algorithm_index],
                    encs=self.encoders[0],
                    decs=self.decoders[0],
                    bank_list=bank_list,
                    Att=qkv_list[0],
                    gatenet=gate_list[0],
                    use_bank=use_bank
                )
                output_mp_states.append(mp_state)
            return None, output_mp_states

        for algorithm_index, features, mp_state in zip(
                algorithm_indices, features_list, mp_state_list):
            inputs = features.inputs
            hints = features.hints
            is_first = features.is_first
            batch_size, nb_nodes = _data_dimensions_chunked(features)

            scan_fn = functools.partial(
                self._msg_passing_step,
                repred=repred,
                init_mp_state=False,
                batch_size=batch_size,
                nb_nodes=nb_nodes,
                spec=self.spec[algorithm_index],
                encs=self.encoders[0],
                decs=self.decoders[0],
                bank_list=bank_list,
                Att=qkv_list[0],
                gatenet=gate_list[0],
                use_bank=use_bank
            )

            mp_state, scan_output = hk.scan(
                scan_fn,
                mp_state,
                (inputs, hints, is_first),
            )

        # We only return the last algorithm's output and state. That's because
        # the output only matters when a single algorithm is processed; the case
        # `algorithm_index==-1` (meaning all algorithms should be processed)
        # is used only to init parameters.
        return (scan_output.output_preds, scan_output.hint_preds), mp_state


def _data_dimensions(features: _Features) -> Tuple[int, int]:
    """Returns (batch_size, nb_nodes)."""
    for inp in features.inputs:
        if inp.location in [_Location.NODE, _Location.EDGE]:
            return inp.data.shape[:2]
    assert False


def _data_dimensions_chunked(features: _FeaturesChunked) -> Tuple[int, int]:
    """Returns (batch_size, nb_nodes)."""
    for inp in features.inputs:
        if inp.location in [_Location.NODE, _Location.EDGE]:
            return inp.data.shape[1:3]
    assert False


def _expand_to(x: _Array, y: _Array) -> _Array:
    while len(y.shape) > len(x.shape):
        x = jnp.expand_dims(x, -1)
    return x


def _iso_expand_to(x, y):  # 为什么要iso独立出来？因为length不要在原文的基础上和data看齐，边的data和点的data shape不同
    z = copy.deepcopy(x)
    while len(y.shape) > len(z.shape):
        z = jnp.expand_dims(z, -1)
    return z


def _is_not_done_broadcast(lengths, i, tensor):
    is_not_done = (lengths > i + 1) * 1.0
    while len(is_not_done.shape) < len(tensor.shape):  # pytype: disable=attribute-error  # numpy-scalars
        is_not_done = jnp.expand_dims(is_not_done, -1)
    return is_not_done
