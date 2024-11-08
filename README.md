# Open-Book Neural Algorithmic Reasoning
This is the official code for the article "Open-Book Neural Algorithmic Reasoning". In this paper, 
a new framework is introduced which can utilize information from the training set. 
Using this framework, we conducted two main experiments: 
Single-Task Augmenting and Multi-Task Augmenting.  
In the files we provided, you’ll find two main folders. 
The folder named `OpenBook` contains five files that are intended to replace existing files based on the CLRS library, 
while the `run` folder contains files for the training and testing experiments.

## Requirements
```
absl-py>=0.13.0
attrs>=21.4.0
chex>=0.0.8
dm-haiku>=0.0.4
jax>=0.2.18
jaxlib>=0.1.69
numpy>=1.21.1
opt-einsum>=3.3.0
optax>=0.0.9
six>=1.16.0
tensorflow>=2.9.0
tfds-nightly==4.5.2.dev202204190046
toolz>=0.11.1
```

## Installation
We follow the framework of the CLRS library and modify some files in it.
So, you need to first install the CLRS packages. 

```commandline
pip install git + https://github.com/google-deepmind/clrs.git
```
After installation, you can replace the files with the same names in the `clrs/_src` folder by placing the `baselines.py`, `encoders.py`, `nets.py`, `processors.py`, and `samplers.py` files from the `OpenBook` folder into it.


## Running Experiments

All the hyperparameters are located in the `flags.DEFINE` sections at the beginning of the files, where you can modify settings for random seeds, processors, batch sizes, and more.
and the current default values of the hyperparameters are the experimental settings for this paper.  
When running the run files, you need to provide two parameters: the random seed and the name of the algorithm to be trained, 
the latter of which can be found in the `algo_lists`.
```python
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
```
To run the single-task augmenting experiment:
```
python run_single.py 40 bfs
```

To run the multi-task augmenting experiment:
```
python run_multi.py 40 bfs
```


## Evaluation

Once the training starts in all the run files, the evaluation will be performed automatically after the training ends.
Therefore, there is no need for additional evaluate code. However, if you manually stop the training, you can still
recover the automatically saved parameters and perform evaluation using the test files.

```eval
# single test
python run_single_test.py 40 bfs
# multi test
python run_multi_test.py 40 bfs
```