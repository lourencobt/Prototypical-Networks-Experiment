#%%
import os
import sys
import gin
import numpy as np
import tensorflow as tf
import torch

# ! TO REMOVE WHEN IMPLEMENTATION IS FINISHED 
# # * This environment variables are meant to be defined by the user
# * of the meta_dataset_reader
os.environ['META_DATASET_ROOT'] = "/home/guests/lbt/meta-dataset"
os.environ['RECORDS_ROOT'] = "/home/guests/lbt/data/records"

# Read Environment Variables to define META_DATASET and RECORDS path
META_DATASET_ROOT = os.environ['META_DATASET_ROOT']
META_RECORDS_ROOT = os.environ['RECORDS_ROOT']

# Path of the meta_dataset_reader Gin configuration file 
GIN_CONFIG_ROOT = os.path.abspath('gin/meta_dataset_config.gin')

sys.path.append(META_DATASET_ROOT)
from meta_dataset.data import config
from meta_dataset.data import dataset_spec as dataset_spec_lib
from meta_dataset.data import learning_spec
from meta_dataset.data import pipeline

# ! TO REMOVE?
from plots import plot_batch, plot_episode

# ! INPUT 
DATASETS = ['aircraft', 'cu_birds', 'dtd', 'fungi', 'ilsvrc_2012',
                'omniglot', 'quickdraw', 'vgg_flower']
DEVICE = 0

#%%
class MetaDatasetReader():
  def __init__(self, data_path, mode, shuffle):
    self.data_path = data_path
    gin.parse_config_file(GIN_CONFIG_ROOT)
    self.train_datasets, self.validation_datasets, self.test_datasets = self.__get_datasets()

  def __get_dataset_spec(self, datasets):
    """ Gets the list of Dataset Specifications of datasets

    Args:
      datasets: A list of datasets to load the dataset specification files

    Returns:
      A list of Dataset Specifications
    
    """
    
    dataset_specs = []
    for dataset_name in datasets:
      dataset_records_path = os.path.join(self.data_path, dataset_name)
      dataset_spec = dataset_spec_lib.load_dataset_spec(dataset_records_path)
      dataset_specs.append(dataset_spec)
    
    return dataset_specs

  @gin.configurable("datasets")
  def __get_datasets(self, train_datasets, validation_datasets, test_datasets):
    """Gets the list of dataset names.

    Args:
      train_datasets: A string of comma-separated dataset names for training.
      validation_datasets: A string of comma-separated dataset names for validation.
      test_datasets: A string of comma-separated dataset names for evaluation.

    Returns:
      Three lists of dataset names
    """
    
    assert (train_datasets and validation_datasets and test_datasets) is not None
    assert isinstance((train_datasets and validation_datasets and test_datasets), str)

    train_datasets = [d.strip() for d in train_datasets.split(',')]
    validation_datasets = [d.strip() for d in validation_datasets.split(',')]
    test_datasets = [d.strip() for d in test_datasets.split(',')]
    
    return train_datasets, validation_datasets, test_datasets
     
#%%
dataset = MetaDatasetReader(META_RECORDS_ROOT, "train", False)









#%%
gin.parse_config_file(GIN_CONFIG_ROOT)

#%%
SPLIT = learning_spec.Split.TRAIN

def _to_torch(sample):
  for key, val in sample.items():
      if isinstance(val, str):
          continue
      val = torch.from_numpy(val.numpy())
      if 'image' in key:
          val = val.permute(0, 3, 1, 2)
      else:
          val = val.long()
      # sample[key] = val.to(DEVICE)
      sample[key] = val
  return sample

def _to_torch2(sample):
  for key, val in sample.items():
      if isinstance(val, str):
          continue
      val = torch.from_numpy()
      if 'image' in key:
          val = val.permute(0, 3, 1, 2)
      else:
          val = val.long()
      # sample[key] = val.to(DEVICE)
      sample[key] = val
  return sample

# For eager execution 
def iterate_dataset(dataset, n):
    for idx, episode in enumerate(dataset):
      if idx == n:
        break
      yield idx, episode

# For eager execution 
def iterate_dataset2(dataset, n):
    for idx, task in enumerate(dataset):
      if idx == n:
        break
      (episode, source_id) = task
      task_dict = {
            'context_images': episode[0],
            'context_labels': episode[1],
            'target_images': episode[3],
            'target_labels': episode[4]
            }
      yield idx, _to_torch(task_dict)

def _load_dataset_spec(datasets):
    assert isinstance(datasets, list)
    
    # Load the dataset_spec.json file of each dataset 
    dataset_specs = []
    for dataset_name in datasets:
        dataset_records_path = os.path.join(RECORDS_ROOT, dataset_name)
        dataset_spec = dataset_spec_lib.load_dataset_spec(dataset_records_path)
        dataset_specs.append(dataset_spec)
    
    return dataset_specs

#%%
def create_multisource_episode_pipeline(datasets, dataset_specs, shuffle):
  use_bilevel_ontology_list = [False]*len(datasets)
  use_dag_ontology_list = [False]*len(datasets)

  # Enable ontology aware sampling for Omniglot and ImageNet. 
  # There isn't the need to put Omniglot or Imagenet in a specific place in the list.
  if 'omniglot' in datasets:
    omniglot_index = datasets.index('omniglot')
  if 'ilsvrc_2012' in datasets:
    imagenet_index = datasets.index('ilsvrc_2012')

  use_bilevel_ontology_list[omniglot_index] = True
  use_dag_ontology_list[imagenet_index] = True

  # Create an Episode Description that is an instance of EpisodeDescriptionConfig that stores the attributes 
  # present in the meta_dataset_config gin file
  variable_ways_shots = config.EpisodeDescriptionConfig()
  
  data_config = config.DataConfig()

  if shuffle:
    dataset_episodic = pipeline.make_multisource_episode_pipeline(
      dataset_spec_list=dataset_specs, 
      use_dag_ontology_list=use_dag_ontology_list,
      use_bilevel_ontology_list=use_bilevel_ontology_list,
      split=SPLIT, 
      episode_descr_config=variable_ways_shots,
      image_size=data_config.image_height, 
      num_prefetch=data_config.num_prefetch,
      shuffle_buffer_size=data_config.shuffle_buffer_size,
      read_buffer_size_bytes=data_config.read_buffer_size_bytes
    )
  else:
    dataset_episodic = pipeline.make_multisource_episode_pipeline(
      dataset_spec_list=dataset_specs, 
      use_dag_ontology_list=use_dag_ontology_list,
      use_bilevel_ontology_list=use_bilevel_ontology_list,
      split=SPLIT, 
      episode_descr_config=variable_ways_shots,
      image_size=data_config.image_height,
      num_prefetch=data_config.num_prefetch,
      read_buffer_size_bytes=data_config.read_buffer_size_bytes
    )

  return dataset_episodic


# %%
if __name__ == '__main__':
  import tqdm
  import time

  dataset_specs = _load_dataset_spec(DATASETS)
  time1 = time.perf_counter()
  dataset = create_multisource_episode_pipeline(DATASETS, dataset_specs, False)
  time2 = time.perf_counter()
  print("Create1 = "+ str(time2-time1) + "seconds")

  time1 = time.perf_counter()
  dataset1 = dataset.as_numpy_iterator()
  dataset1 = iter(dataset1)
  for i in range(100):
    (episode, source_id) = dataset1.get_next()
    task_dict = {
                'context_images': episode[0],
                'context_labels': episode[1],
                'target_images': episode[3],
                'target_labels': episode[4]
                }
    _to_torch(task_dict)
  time2 = time.perf_counter()
  print("Loop1 = "+ str(time2-time1) + "seconds")

  time1 = time.perf_counter()
  dataset = create_multisource_episode_pipeline(DATASETS, dataset_specs, False)
  time2 = time.perf_counter()
  print("Create2 = "+ str(time2-time1) + "seconds")

  time1 = time.perf_counter()
  for i, (episode, source_id) in iterate_dataset(dataset,100):
    task_dict = {
                'context_images': episode[0],
                'context_labels': episode[1],
                'target_images': episode[3],
                'target_labels': episode[4]
                }
    _to_torch2(task_dict)
  time2 = time.perf_counter()
  print("Loop2 = "+ str(time2-time1) + "seconds")

  time1 = time.perf_counter()
  dataset = create_multisource_episode_pipeline(DATASETS, dataset_specs, False)
  time2 = time.perf_counter()
  print("Create3 = "+ str(time2-time1) + "seconds")

  time1 = time.perf_counter()
  for i, task_dict in iterate_dataset2(dataset,100):
    _to_torch(task_dict)
  time2 = time.perf_counter()
  print("Loop3 = "+ str(time2-time1) + "seconds")
  
  # N_EPISODES=2
  # for idx, (episode, source_id) in iterate_dataset(dataset, N_EPISODES):
  #   print('Episode id: %d from source %s' % (idx, dataset_specs[source_id].name))
  #   episode = [a.numpy() for a in episode]
  #   plot_episode(support_images=episode[0], support_class_ids=episode[2],
  #               query_images=episode[3], query_class_ids=episode[5])
# %%
