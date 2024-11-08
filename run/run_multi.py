"""Run training of multi algorithmic tasks from CLRS with Open-Book."""

import functools
import os
import shutil
from typing import Any, Dict, List
import time
import random
from absl import app
from absl import flags
from absl import logging
import clrs
import jax
import numpy as np
import requests
import tensorflow as tf
import optax
import jax.numpy as jnp
import copy
import math
import sys


np.set_printoptions(threshold=math.inf)
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".9"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
# =====================================================================================================================================
# name = 'bfs'
name = sys.argv[2]
rseed = sys.argv[1]
rseed = int(rseed)
# =====================================================================================================================================

algo_lists = ['articulation_points',
              'activity_selector',
              'bellman_ford',
              'bfs',
              'binary_search',
              'bridges',
              'bubble_sort',
              'dag_shortest_paths',
              'dfs',
              'dijkstra',
              'find_maximum_subarray_kadane',
              'floyd_warshall',
              'graham_scan',
              'heapsort',
              'insertion_sort',
              'jarvis_march',
              'kmp_matcher',
              'lcs_length',
              'matrix_chain_order',
              'minimum',
              'mst_kruskal',
              'mst_prim',
              'naive_string_matcher',
              'optimal_bst',
              'quickselect',
              'quicksort',
              'segments_intersect',
              'strongly_connected_components',
              'task_scheduling',
              'topological_sort']
algo_lists2 = ['articulation_points']
algo_dicts = {'articulation_points': 0,
              'activity_selector': 1,
              'bellman_ford': 2,
              'bfs': 3,
              'binary_search': 4,
              'bridges': 5,
              'bubble_sort': 6,
              'dag_shortest_paths': 7,
              'dfs': 8,
              'dijkstra': 9,
              'find_maximum_subarray_kadane': 10,
              'floyd_warshall': 11,
              'graham_scan': 12,
              'heapsort': 13,
              'insertion_sort': 14,
              'jarvis_march': 15,
              'kmp_matcher': 16,
              'lcs_length': 17,
              'matrix_chain_order': 18,
              'minimum': 19,
              'mst_kruskal': 20,
              'mst_prim': 21,
              'naive_string_matcher': 22,
              'optimal_bst': 23,
              'quickselect': 24,
              'quicksort': 25,
              'segments_intersect': 26,
              'strongly_connected_components': 27,
              'task_scheduling': 28,
              'topological_sort': 29}
single_algo_score = {
    'articulation_points': 0.8832,
    'activity_selector': 0.9518,
    'bellman_ford': 0.9739,
    'bfs': 0.9973,
    'binary_search': 0.7758,
    'bridges': 0.9399,
    'bubble_sort': 0.6768,
    'dag_shortest_paths': 0.9819,
    'dfs': 0.4779,
    'dijkstra': 0.9605,
    'find_maximum_subarray_kadane': 0.7636,
    'floyd_warshall': 0.4852,
    'graham_scan': 0.9362,
    'heapsort': 0.3104,
    'insertion_sort': 0.7814,
    'jarvis_march': 0.9101,
    'kmp_matcher': 0.1951,
    'lcs_length': 0.8051,
    'matrix_chain_order': 0.9168,
    'minimum': 0.9778,
    'mst_kruskal': 0.8980,
    'mst_prim': 0.8639,
    'naive_string_matcher': 0.7867,
    'optimal_bst': 0.7377,
    'quickselect': 0.47,
    'quicksort': 0.6464,
    'segments_intersect': 0.9764,
    'strongly_connected_components': 0.4343,
    'task_scheduling': 0.8725,
    'topological_sort': 0.8727
}
# seed setup
random.seed(42)
# use growth GPU memory
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
# handle JAX memory error
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"

# print("XLA_PYTHON_CLIENT_ALLOCATOR=platform")
# print(os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"])
# print(os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"])

flags.DEFINE_list('algorithms', algo_lists, 'Which algorithms to run.')
flags.DEFINE_list('train_lengths', ['4', '7', '11', '13', '16'],
                  'Which training sizes to use. A size of -1 means '
                  'use the benchmark dataset.')
flags.DEFINE_integer('length_needle', -8,
                     'Length of needle for training and validation '
                     '(not testing) in string matching algorithms. '
                     'A negative value randomizes the length for each sample '
                     'between 1 and the opposite of the value. '
                     'A value of 0 means use always 1/4 of the length of '
                     'the haystack (the default sampler behavior).')
flags.DEFINE_integer('seed', rseed, 'Random seed to set')

flags.DEFINE_boolean('random_pos', True,
                     'Randomize the pos input common to all algos.')
flags.DEFINE_boolean('enforce_permutations', True,
                     'Whether to enforce permutation-type node pointers.')
flags.DEFINE_boolean('enforce_pred_as_input', True,
                     'Whether to change pred_h hints into pred inputs.')
flags.DEFINE_integer('batch_size', 32, 'Batch size used for training.')
flags.DEFINE_boolean('chunked_training', True,
                     'Whether to use chunking for training.')
flags.DEFINE_integer('chunk_length', 16,
                     'Time chunk length used for training (if '
                     '`chunked_training` is True.')
flags.DEFINE_integer('train_steps', 10000 + 1, 'Number of training iterations.')
flags.DEFINE_integer('eval_every', 50, 'Evaluation frequency (in steps).')
flags.DEFINE_integer('test_every', 500, 'Evaluation frequency (in steps).')

flags.DEFINE_integer('hidden_size', 128,
                     'Number of hidden units of the model.')
flags.DEFINE_integer('nb_heads', 1, 'Number of heads for GAT processors')
flags.DEFINE_integer('nb_msg_passing_steps', 1,
                     'Number of message passing steps to run per hint.')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate to use.')
flags.DEFINE_float('grad_clip_max_norm', 1.0,
                   'Gradient clipping by norm. 0.0 disables grad clipping')
flags.DEFINE_float('dropout_prob', 0.0, 'Dropout rate to use.')
flags.DEFINE_float('hint_teacher_forcing', 0.0,
                   'Probability that ground-truth teacher hints are encoded '
                   'during training instead of predicted hints. Only '
                   'pertinent in encoded_decoded modes.')
flags.DEFINE_enum('hint_mode', 'encoded_decoded',
                  ['encoded_decoded', 'decoded_only', 'none'],
                  'How should hints be used? Note, each mode defines a '
                  'separate task, with various difficulties. `encoded_decoded` '
                  'requires the model to explicitly materialise hint sequences '
                  'and therefore is hardest, but also most aligned to the '
                  'underlying algorithmic rule. Hence, `encoded_decoded` '
                  'should be treated as the default mode for our benchmark. '
                  'In `decoded_only`, hints are only used for defining '
                  'reconstruction losses. Often, this will perform well, but '
                  'note that we currently do not make any efforts to '
                  'counterbalance the various hint losses. Hence, for certain '
                  'tasks, the best performance will now be achievable with no '
                  'hint usage at all (`none`).')
flags.DEFINE_enum('hint_repred_mode', 'soft', ['soft', 'hard', 'hard_on_eval'],
                  'How to process predicted hints when fed back as inputs.'
                  'In soft mode, we use softmaxes for categoricals, pointers '
                  'and mask_one, and sigmoids for masks. '
                  'In hard mode, we use argmax instead of softmax, and hard '
                  'thresholding of masks. '
                  'In hard_on_eval mode, soft mode is '
                  'used for training and hard mode is used for evaluation.')
flags.DEFINE_boolean('use_ln', True,
                     'Whether to use layer normalisation in the processor.')
flags.DEFINE_boolean('use_lstm', False,
                     'Whether to insert an LSTM after message passing.')
flags.DEFINE_integer('nb_triplet_fts', 8,
                     'How many triplet features to compute?')

flags.DEFINE_enum('encoder_init', 'xavier_on_scalars',
                  ['default', 'xavier_on_scalars'],
                  'Initialiser to use for the encoders.')  # check
flags.DEFINE_enum('processor_type', 'triplet_gmpnn',  # check
                  ['deepsets', 'mpnn', 'pgn', 'pgn_mask',
                   'triplet_mpnn', 'triplet_pgn', 'triplet_pgn_mask',
                   'gat', 'gatv2', 'gat_full', 'gatv2_full',
                   'gpgn', 'gpgn_mask', 'gmpnn',
                   'triplet_gpgn', 'triplet_gpgn_mask', 'triplet_gmpnn'],
                  'Processor type to use as the network P.')

flags.DEFINE_string('checkpoint_path', './tmp/model_params',
                    'Path in which checkpoints are saved.')
flags.DEFINE_string('dataset_path', './tmp/CLRS30',
                    'Path in which dataset is stored.')
flags.DEFINE_boolean('freeze_processor', False,
                     'Whether to freeze the processor of the model.')

FLAGS = flags.FLAGS

PRED_AS_INPUT_ALGOS = [
    'binary_search',
    'minimum',
    'find_maximum_subarray',
    'find_maximum_subarray_kadane',
    'matrix_chain_order',
    'lcs_length',
    'optimal_bst',
    'activity_selector',
    'task_scheduling',
    'naive_string_matcher',
    'kmp_matcher',
    'jarvis_march']

print(jax.local_devices())
print('1')
print(tf.config.list_physical_devices("GPU"))
a = tf.constant([1, 2, 3])
print(a.device)
print(f'Jax.devices：=============={jax.devices}')


# jax.default_device = jax.devices("gpu")[0]


def unpack(v):
    try:
        return v.item()  # DeviceArray
    except (AttributeError, ValueError):
        return v


def _iterate_sampler(sampler, batch_size):
    while True:
        yield sampler.next(batch_size)


def _maybe_download_dataset(dataset_path):
    """Download CLRS30 dataset if needed."""
    dataset_folder = os.path.join(dataset_path, clrs.get_clrs_folder())
    if os.path.isdir(dataset_folder):
        logging.info('Dataset found at %s. Skipping download.', dataset_folder)
        return dataset_folder
    logging.info('Dataset not found in %s. Downloading...', dataset_folder)

    clrs_url = clrs.get_dataset_gcp_url()
    request = requests.get(clrs_url, allow_redirects=True)
    clrs_file = os.path.join(dataset_path, os.path.basename(clrs_url))
    os.makedirs(dataset_folder)
    open(clrs_file, 'wb').write(request.content)
    shutil.unpack_archive(clrs_file, extract_dir=dataset_folder)
    os.remove(clrs_file)
    return dataset_folder


def make_sampler(length: int,
                 rng: Any,
                 algorithm: str,
                 split: str,
                 batch_size: int,
                 multiplier: int,
                 randomize_pos: bool,
                 enforce_pred_as_input: bool,
                 enforce_permutations: bool,
                 chunked: bool,
                 chunk_length: int,
                 sampler_kwargs: Dict[str, Any],
                 bank: bool = False):
    """Create a sampler with given options.

    Args:
      length: Size of samples (i.e., number of nodes in the graph).
        A length of -1 will mean that the benchmark
        dataset (for the given split) is used. Positive sizes will instantiate
        samplers of the corresponding size.
      rng: Numpy random state.
      algorithm: The name of the algorithm to sample from.
      split: 'train', 'val' or 'test'.
      batch_size: Samples per batch.
      multiplier: Integer multiplier for the number of samples in the dataset,
        only used for positive sizes. Negative multiplier means infinite samples.
      randomize_pos: Whether to randomize the `pos` input.
      enforce_pred_as_input: Whether to convert fixed pred_h hints to inputs.
      enforce_permutations: Whether to enforce permutation pointers.
      chunked: Whether to chunk the dataset.
      chunk_length: Unroll length of chunks, if `chunked` is True.
      sampler_kwargs: Extra args passed to the sampler.
    Returns:
      A sampler (iterator), the number of samples in the iterator (negative
      if infinite samples), and the spec.
    """
    if length < 0:  # load from file
        dataset_folder = _maybe_download_dataset(FLAGS.dataset_path)
        sampler, num_samples, spec = clrs.create_dataset(folder=dataset_folder,
                                                         algorithm=algorithm,
                                                         batch_size=batch_size,
                                                         split=split)
        sampler = sampler.as_numpy_iterator()
    else:
        num_samples = clrs.CLRS30[split]['num_samples'] * multiplier
        sampler, spec = clrs.build_sampler(
            algorithm,
            bank=bank,
            seed=rng.randint(2 ** 32),
            num_samples=num_samples,
            length=length,
            **sampler_kwargs,
        )
        sampler = _iterate_sampler(sampler, batch_size)

    if randomize_pos:
        sampler = clrs.process_random_pos(sampler, rng)
    if enforce_pred_as_input and algorithm in PRED_AS_INPUT_ALGOS:
        spec, sampler = clrs.process_pred_as_input(spec, sampler)
    spec, sampler = clrs.process_permutations(spec, sampler, enforce_permutations)
    if chunked:
        sampler = clrs.chunkify(sampler, chunk_length)
    return sampler, num_samples, spec


def make_multi_sampler(sizes, rng, **kwargs):
    """Create a sampler with cycling sample sizes."""
    ss = []
    tot_samples = 0
    for length in sizes:
        sampler, num_samples, spec = make_sampler(length, rng, **kwargs)
        ss.append(sampler)
        tot_samples += num_samples

    def cycle_samplers():
        while True:
            for s in ss:
                yield next(s)

    return cycle_samplers(), tot_samples, spec


def _concat(dps, axis):
    return jax.tree_util.tree_map(lambda *x: np.concatenate(x, axis), *dps)


def collect_and_eval(sampler, predict_fn, sample_count, rng_key, extras):
    """Collect batches of output and hint preds and evaluate them."""
    processed_samples = 0
    preds = []
    outputs = []
    weight_all = []
    while processed_samples < sample_count:
        feedback = next(sampler)
        batch_size = feedback.outputs[0].data.shape[0]
        outputs.append(feedback.outputs)
        new_rng_key, rng_key = jax.random.split(rng_key)
        cur_preds, _, weight = predict_fn(new_rng_key, feedback.features)
        preds.append(cur_preds)
        weight_all.append(weight)
        processed_samples += batch_size
    outputs = _concat(outputs, axis=0)
    preds = _concat(preds, axis=0)
    out = clrs.evaluate(outputs, preds)
    if extras:
        out.update(extras)
    return {k: unpack(v) for k, v in out.items()}, weight_all


def create_samplers(rng, train_lengths: List[int]):
    """Create all the samplers."""
    train_samplers = []
    val_samplers = []
    val_sample_counts = []
    test_samplers = []
    test_sample_counts = []
    spec_list = []

    for algo_idx, algorithm in enumerate(FLAGS.algorithms):
        # Make full dataset pipeline run on CPU (including prefetching).
        with tf.device('/cpu:0'):

            if algorithm in ['naive_string_matcher', 'kmp_matcher']:
                # Fixed haystack + needle; variability will be in needle
                # Still, for chunked training, we maintain as many samplers
                # as train lengths, since, for each length there is a separate state,
                # and we must keep the 1:1 relationship between states and samplers.
                max_length = max(train_lengths)
                if max_length > 0:  # if < 0, we are using the benchmark data
                    max_length = (max_length * 5) // 4
                train_lengths = [max_length]
                if FLAGS.chunked_training:
                    train_lengths = train_lengths * len(train_lengths)

            logging.info('Creating samplers for algo %s', algorithm)

            p = tuple([0.1 + 0.1 * i for i in range(9)])
            if p and algorithm in ['articulation_points', 'bridges',
                                   'mst_kruskal', 'bipartite_matching']:
                # Choose a lower connection probability for the above algorithms,
                # otherwise trajectories are very long
                p = tuple(np.array(p) / 2)
            length_needle = FLAGS.length_needle
            sampler_kwargs = dict(p=p, length_needle=length_needle)
            if length_needle == 0:
                sampler_kwargs.pop('length_needle')

            common_sampler_args = dict(
                algorithm=FLAGS.algorithms[algo_idx],
                rng=rng,
                enforce_pred_as_input=FLAGS.enforce_pred_as_input,
                enforce_permutations=FLAGS.enforce_permutations,
                chunk_length=FLAGS.chunk_length,
            )

            train_args = dict(sizes=train_lengths,
                              split='train',
                              batch_size=FLAGS.batch_size,
                              multiplier=-1,
                              randomize_pos=FLAGS.random_pos,
                              chunked=FLAGS.chunked_training,
                              sampler_kwargs=sampler_kwargs,
                              **common_sampler_args)
            train_sampler, _, spec = make_multi_sampler(**train_args)

            mult = clrs.CLRS_30_ALGS_SETTINGS[algorithm]['num_samples_multiplier']
            val_args = dict(sizes=[np.amax(train_lengths)],
                            split='val',
                            batch_size=FLAGS.batch_size,
                            multiplier=2 * mult,
                            randomize_pos=FLAGS.random_pos,
                            chunked=False,
                            sampler_kwargs=sampler_kwargs,
                            **common_sampler_args)
            val_sampler, val_samples, spec = make_multi_sampler(**val_args)

            test_args = dict(sizes=[-1],
                             split='test',
                             batch_size=32,
                             multiplier=2 * mult,
                             randomize_pos=False,
                             chunked=False,
                             sampler_kwargs={},
                             **common_sampler_args)
            test_sampler, test_samples, spec = make_multi_sampler(**test_args)

        spec_list.append(spec)
        train_samplers.append(train_sampler)
        val_samplers.append(val_sampler)
        val_sample_counts.append(val_samples)
        test_samplers.append(test_sampler)
        test_sample_counts.append(test_samples)

    return (train_samplers,
            val_samplers, val_sample_counts,
            test_samplers, test_sample_counts,
            spec_list)


def create_bank_samplers(rng, train_lengths: List[int], bank_batch):
    """Create all the samplers."""
    train_samplers = []
    val_samplers = []
    val_sample_counts = []
    test_samplers = []
    test_sample_counts = []
    spec_list = []

    for algo_idx, algorithm in enumerate(FLAGS.algorithms):
        # Make full dataset pipeline run on CPU (including prefetching).
        with tf.device('/cpu:0'):

            if algorithm in ['naive_string_matcher', 'kmp_matcher']:
                # Fixed haystack + needle; variability will be in needle
                # Still, for chunked training, we maintain as many samplers
                # as train lengths, since, for each length there is a separate state,
                # and we must keep the 1:1 relationship between states and samplers.
                max_length = max(train_lengths)
                if max_length > 0:  # if < 0, we are using the benchmark data
                    max_length = (max_length * 5) // 4
                train_lengths = [max_length]
                if FLAGS.chunked_training:
                    train_lengths = train_lengths * len(train_lengths)

            logging.info('Creating samplers for algo %s', algorithm)

            p = tuple([0.1 + 0.1 * i for i in range(9)])
            if p and algorithm in ['articulation_points', 'bridges',
                                   'mst_kruskal', 'bipartite_matching']:
                # Choose a lower connection probability for the above algorithms,
                # otherwise trajectories are very long
                p = tuple(np.array(p) / 2)
            length_needle = FLAGS.length_needle
            sampler_kwargs = dict(p=p, length_needle=length_needle)
            if length_needle == 0:
                sampler_kwargs.pop('length_needle')

            common_sampler_args = dict(
                algorithm=FLAGS.algorithms[algo_idx],
                rng=rng,
                enforce_pred_as_input=FLAGS.enforce_pred_as_input,
                enforce_permutations=FLAGS.enforce_permutations,
                chunk_length=FLAGS.chunk_length,
            )

            train_args = dict(sizes=train_lengths,
                              split='train',
                              batch_size=bank_batch,
                              multiplier=-1,
                              randomize_pos=FLAGS.random_pos,
                              chunked=False,
                              sampler_kwargs=sampler_kwargs,
                              bank=True,
                              **common_sampler_args)
            train_sampler, _, spec = make_multi_sampler(**train_args)

            mult = clrs.CLRS_30_ALGS_SETTINGS[algorithm]['num_samples_multiplier']
            val_args = dict(sizes=[np.amax(train_lengths)],
                            split='val',
                            batch_size=FLAGS.batch_size,
                            multiplier=2 * mult,
                            randomize_pos=FLAGS.random_pos,
                            chunked=False,
                            sampler_kwargs=sampler_kwargs,
                            **common_sampler_args)
            val_sampler, val_samples, spec = make_multi_sampler(**val_args)

        spec_list.append(spec)
        train_samplers.append(train_sampler)
        val_samplers.append(val_sampler)
        val_sample_counts.append(1)
        test_samplers.append(1)
        test_sample_counts.append(1)

    return (train_samplers,
            val_samplers, val_sample_counts,
            test_samplers, test_sample_counts,
            spec_list)


def create_single_samplers(rng, train_lengths: List[int], algo_idx, algorithm):
    """Create all the samplers."""
    train_samplers = []
    val_samplers = []
    val_sample_counts = []
    test_samplers = []
    test_sample_counts = []
    spec_list = []

    # Make full dataset pipeline run on CPU (including prefetching).
    with tf.device('/cpu:0'):

        if algorithm in ['naive_string_matcher', 'kmp_matcher']:
            # Fixed haystack + needle; variability will be in needle
            # Still, for chunked training, we maintain as many samplers
            # as train lengths, since, for each length there is a separate state,
            # and we must keep the 1:1 relationship between states and samplers.
            max_length = max(train_lengths)
            if max_length > 0:  # if < 0, we are using the benchmark data
                max_length = (max_length * 5) // 4
            train_lengths = [max_length]
            if FLAGS.chunked_training:
                train_lengths = train_lengths * len(train_lengths)

        logging.info('Creating samplers for algo %s', algorithm)

        p = tuple([0.1 + 0.1 * i for i in range(9)])
        if p and algorithm in ['articulation_points', 'bridges',
                               'mst_kruskal', 'bipartite_matching']:
            # Choose a lower connection probability for the above algorithms,
            # otherwise trajectories are very long
            p = tuple(np.array(p) / 2)
        length_needle = FLAGS.length_needle
        sampler_kwargs = dict(p=p, length_needle=length_needle)
        if length_needle == 0:
            sampler_kwargs.pop('length_needle')

        common_sampler_args = dict(
            algorithm=FLAGS.algorithms[algo_idx],
            rng=rng,
            enforce_pred_as_input=FLAGS.enforce_pred_as_input,
            enforce_permutations=FLAGS.enforce_permutations,
            chunk_length=FLAGS.chunk_length,
        )

        train_args = dict(sizes=train_lengths,
                          split='train',
                          batch_size=FLAGS.batch_size,
                          multiplier=-1,
                          randomize_pos=FLAGS.random_pos,
                          chunked=FLAGS.chunked_training,
                          sampler_kwargs=sampler_kwargs,
                          **common_sampler_args)
        train_sampler, _, spec = make_multi_sampler(**train_args)

        mult = clrs.CLRS_30_ALGS_SETTINGS[algorithm]['num_samples_multiplier']
        val_args = dict(sizes=[np.amax(train_lengths)],
                        split='val',
                        batch_size=FLAGS.batch_size,
                        multiplier=2 * mult,
                        randomize_pos=FLAGS.random_pos,
                        chunked=False,
                        sampler_kwargs=sampler_kwargs,
                        **common_sampler_args)
        val_sampler, val_samples, spec = make_multi_sampler(**val_args)

        test_args = dict(sizes=[-1],
                         split='test',
                         batch_size=32,
                         multiplier=2 * mult,
                         randomize_pos=False,
                         chunked=False,
                         sampler_kwargs={},
                         **common_sampler_args)
        test_sampler, test_samples, spec = make_multi_sampler(**test_args)

    spec_list.append(spec)
    train_samplers.append(train_sampler)
    val_samplers.append(val_sampler)
    val_sample_counts.append(val_samples)
    test_samplers.append(test_sampler)
    test_sample_counts.append(test_samples)

    return (train_samplers,
            val_samplers, val_sample_counts,
            test_samplers, test_sample_counts,
            spec_list)


def main(unused_argv):
    print(FLAGS)
    if FLAGS.hint_mode == 'encoded_decoded':
        encode_hints = True
        decode_hints = True
    elif FLAGS.hint_mode == 'decoded_only':
        encode_hints = False
        decode_hints = True
    elif FLAGS.hint_mode == 'none':
        encode_hints = False
        decode_hints = False
    else:
        raise ValueError('Hint mode not in {encoded_decoded, decoded_only, none}.')

    train_lengths = [int(x) for x in FLAGS.train_lengths]

    rng = np.random.RandomState(FLAGS.seed)
    rng_key = jax.random.PRNGKey(rng.randint(2 ** 32))

    algo_idx = algo_lists.index(name)

    # Create samplers
    bank_batch = 8
    (train_samplers,
     val_samplers, val_sample_counts,
     test_samplers, test_sample_counts,
     _) = create_single_samplers(rng, train_lengths, algo_idx, name)
    bank_samplers, dummy_val_samplers, _, _, _, spec_list = create_bank_samplers(rng, train_lengths, bank_batch)

    processor_factory = clrs.get_processor_factory(
        FLAGS.processor_type,
        use_ln=FLAGS.use_ln,
        nb_triplet_fts=FLAGS.nb_triplet_fts,
        nb_heads=FLAGS.nb_heads
    )

    model_params = dict(
        processor_factory=processor_factory,
        hidden_dim=FLAGS.hidden_size,
        encode_hints=encode_hints,
        decode_hints=decode_hints,
        encoder_init=FLAGS.encoder_init,
        use_lstm=FLAGS.use_lstm,
        learning_rate=FLAGS.learning_rate,
        grad_clip_max_norm=FLAGS.grad_clip_max_norm,
        checkpoint_path=FLAGS.checkpoint_path,
        freeze_processor=FLAGS.freeze_processor,
        dropout_prob=FLAGS.dropout_prob,
        hint_teacher_forcing=FLAGS.hint_teacher_forcing,
        hint_repred_mode=FLAGS.hint_repred_mode,
        nb_msg_passing_steps=FLAGS.nb_msg_passing_steps,
    )

    eval_model = clrs.models.BaselineModel(
        spec=spec_list,
        dummy_trajectory=[next(t) for t in dummy_val_samplers],
        **model_params
    )
    if FLAGS.chunked_training:
        train_model = clrs.models.BaselineModelChunked(
            spec=spec_list,
            dummy_trajectory=[next(t) for t in bank_samplers],
            **model_params
        )
    else:
        train_model = eval_model

    # Training loop.
    # Metrics Setup
    val_score_list2 = [-1.0, -2.0, -3.0]
    step_list = [0, 0, 0]
    pkl_base_name = 'valbest.pkl'
    maxval = -1.0
    maxval_index = -1

    current_train_items = 0
    step = 0
    next_eval = 0
    # Make sure scores improve on first step, but not overcome best score
    # until all algos have had at least one evaluation.
    length_idx = 0

    start_time = time.time()
    while step < FLAGS.train_steps:

        feedback_list = [next(t) for t in train_samplers]
        bank_list = [next(t) for t in bank_samplers]
        # Initialize model.
        if step == 0:
            all_features = [f.features for f in feedback_list]
            if FLAGS.chunked_training:
                # We need to initialize the model with samples of all lengths for
                # all algorithms. Also, we need to make sure that the order of these
                # sample sizes is the same as the order of the actual training sizes.
                all_length_features = [all_features] + [
                    [next(t).features for t in train_samplers]
                    for _ in range(len(train_lengths))]
                train_model.init(all_length_features[:-1], FLAGS.seed + 1, bank_list, algo_idx)
            else:
                train_model.init(all_features, FLAGS.seed + 1, feedback_list)

        # Training step.

        feedback = feedback_list[0]
        rng_key, new_rng_key = jax.random.split(rng_key)
        if FLAGS.chunked_training:
            # In chunked training, we must indicate which training length we are
            # using, so the model uses the correct state.
            length_and_algo_idx = (length_idx, algo_idx)
        else:
            # In non-chunked training, all training lengths can be treated equally,
            # since there is no state to maintain between batches.
            length_and_algo_idx = algo_idx
        cur_loss = train_model.feedback(rng_key, feedback, length_and_algo_idx,
                                        bank_list, True)
        rng_key = new_rng_key

        if FLAGS.chunked_training:
            examples_in_chunk = np.sum(feedback.features.is_last).item()
        else:
            examples_in_chunk = len(feedback.features.lengths)
        current_train_items += examples_in_chunk

        print('Index %d Algo %s step %i current loss %f, current_train_items %i' % (
            algo_idx, name, step,
            cur_loss, current_train_items))

        # Periodically evaluate model
        if step >= next_eval:
            eval_model.params = train_model.params

            common_extras = {'examples_seen': current_train_items,
                             'step': step,
                             'algorithm': FLAGS.algorithms[algo_idx]}

            # Validation info.
            new_rng_key, rng_key = jax.random.split(rng_key)
            val_stats, weight_all = collect_and_eval(
                val_samplers[0],
                functools.partial(eval_model.predict, algorithm_index=algo_idx, bank_list=bank_list),
                val_sample_counts[0],
                new_rng_key,
                extras=common_extras)

            print('Index %d (val) algo %s step %d: %s' % (
                algo_idx, FLAGS.algorithms[algo_idx], step, val_stats))
            val_scores = val_stats['score']

            next_eval += FLAGS.eval_every

            # If best total score, update best checkpoint.
            # Also save a best checkpoint on the first step.

            min_score = min(val_score_list2)
            min_index = val_score_list2.index(min_score)

            if (val_scores > min_score) or step == 0:
                print('Current model succeed to update avgmodel.')
                print(
                    'Current all score:{},higher than pre_model:{}'.format(
                        val_scores,
                        min_score
                    ))
                val_score_list2[min_index] = val_scores
                step_list[min_index] = step
                if (val_scores > maxval):
                    maxval = val_scores
                    maxval_index = min_index
                print(f'Index====={min_index}')
                train_model.save_model(str(min_index) + name + pkl_base_name)
            else:
                print('Current model failed to update avgmodel.')
                print(
                    'Current all score:{},lower than pre_model:{}'.format(
                        val_scores,
                        min_score
                    ))

            if step == FLAGS.train_steps - 1:
                print('last val has been stored')
                train_model.save_model('last_single' + name + '.pkl')

        step += 1
        length_idx = (length_idx + 1) % len(train_lengths)

    cost_time = time.time() - start_time
    cost_time = cost_time // 60
    max_valScore = -1.0
    max_lastScore = -1.0
    origin_score = 0
    print('Whole cost time ={}'.format(cost_time))
    for i in range(1):
        bank_list = [next(t) for t in bank_samplers]
    logging.info('Restoring best model from checkpoint...')
    max_index = -1
    for i in range(len(val_score_list2)):
        print(f'best val{i} model,from step{step_list[i]}:')
        eval_model.restore_model(str(i) + name + pkl_base_name, only_load_processor=False)
        score_list = -2.0
        common_extras = {'examples_seen': current_train_items,
                         'step': step,
                         'algorithm': FLAGS.algorithms[algo_idx]}

        new_rng_key, rng_key = jax.random.split(rng_key)
        test_stats, _ = collect_and_eval(
            test_samplers[0],
            functools.partial(eval_model.predict, algorithm_index=algo_idx, bank_list=bank_list),
            test_sample_counts[0],
            new_rng_key,
            extras=common_extras)
        score_list = test_stats['score']
        if (score_list > max_valScore):
            max_valScore = score_list
            max_index = i
        print('Index %d (test) algo %s : %s' % (algo_idx, FLAGS.algorithms[algo_idx], test_stats))
        print(f'best val{i} score ={score_list}')
        if (i == maxval_index):
            origin_score = score_list

    print('=====================================================')
    for i in range(1):
        bank_list = [next(t) for t in bank_samplers]
        print(f'last val{i} score:')
        score_list = -2.0
        eval_model.restore_model('last_single' + name + '.pkl', only_load_processor=False)

        common_extras = {'examples_seen': current_train_items,
                         'step': step,
                         'algorithm': FLAGS.algorithms[algo_idx]}

        new_rng_key, rng_key = jax.random.split(rng_key)
        test_stats, weight_all = collect_and_eval(
            test_samplers[0],
            functools.partial(eval_model.predict, algorithm_index=algo_idx, bank_list=bank_list),
            test_sample_counts[0],
            new_rng_key,
            extras=common_extras)

        score_list = test_stats['score']
        if (score_list > max_lastScore):
            max_lastScore = score_list
        print('Index %d (test) algo %s : %s' % (algo_idx, FLAGS.algorithms[algo_idx], test_stats))
        print('last val score ={}'.format(score_list))

    bank_list = [next(t) for t in bank_samplers]
    eval_model.restore_model(str(max_index) + name + pkl_base_name, only_load_processor=False)
    common_extras = {'examples_seen': current_train_items,
                     'step': step,
                     'algorithm': FLAGS.algorithms[algo_idx]}

    new_rng_key, rng_key = jax.random.split(rng_key)
    test_stats, weight_all = collect_and_eval(
        test_samplers[0],
        functools.partial(eval_model.predict, algorithm_index=algo_idx, bank_list=bank_list),
        test_sample_counts[0],
        new_rng_key,
        extras=common_extras)

    weight_all_length = len(weight_all)

    all_weight = 0
    sum_type = 8
    final_type = 240 / sum_type

    for weight in weight_all:
        tempdata = jnp.sum(weight, axis=(0, 1, 2, 3)) / (
                    weight.shape[0] * weight.shape[1] * weight.shape[2] * weight.shape[3])  #
        assert tempdata.shape == (241,)
        tempdata = tempdata[:240]
        # 240->30 or 8
        sum_weight = 0
        for i in range(sum_type):
            sum_weight += tempdata[i::sum_type]
        assert sum_weight.shape == (final_type,)
        all_weight += sum_weight

    all_weight = all_weight / weight_all_length
    assert all_weight.shape == (final_type,)

    weight_idx = jnp.argsort(all_weight)
    weight_value = jnp.sort(all_weight)
    # algo_lists
    for i in range(all_weight.shape[0]):
        print(f'Algorithm{algo_lists[i]}score = {all_weight[i]}')

    for i, j in zip(weight_idx, weight_value):
        print(f'Algorithm{algo_lists[i]}score = {j}')
        print('')

    print('============================================')

    print('Done!')
    print(all_weight)
    print(f'Best saved model score： {max_valScore}')
    print(f'Last model score:  {max_lastScore}')


if __name__ == '__main__':
    app.run(main)