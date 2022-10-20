# Meta Dataset Reader - based on: 
#   https://github.com/peymanbateni/simple-cnaps/blob/master/simple-cnaps-src/meta_dataset_reader.py
#   https://github.com/VICO-UoE/URL/blob/master/data/meta_dataset_reader.py
#   https://github.com/google-research/meta-dataset/blob/main/Intro_to_Metadataset.ipynb
#
# This Reader is prepared to read Episodes and Batches (Currently, not available) from .tfrecords datasets, where each .tfrecord file has all the images from a specific class. To know more, explore https://github.com/google-research/meta-dataset/.

import os
import sys
import gin
import numpy as np
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
  tf.config.experimental.set_memory_growth(device, True)

import torch
from paths import META_DATASET_ROOT, GIN_CONFIG_ROOT_MEDICAL
from utils import device

sys.path.append(META_DATASET_ROOT)
from meta_dataset.data import config
from meta_dataset.data import dataset_spec as dataset_spec_lib
from meta_dataset.data import learning_spec
from meta_dataset.data import pipeline_medical as pipeline
 
SAMPLING_SEED = 1234

class MetaDatasetReader():
  """
  Class to be inherited by different types of MetaDatasetReaders, for example, episode or batch readers
  """
  def __init__(self, data_path, mode, shuffle):
    """
    Initializes a MetaDatasetReader

    Args:
      data_path: A string, the path to the Records Folder 
      mode: A string, the mode of the data processing pipeline. Options: "train", "val" or "test" modes. 
      shuffle: A bool, shuffle or not the dataset samples
    """
    self.data_path = data_path
    self.mode = mode
    self.shuffle = shuffle

    gin.parse_config_file(GIN_CONFIG_ROOT_MEDICAL)
    self.train_datasets, self.val_datasets, self.test_datasets = self._get_datasets()
    self.datasets_dict = {}
    self.data_config = config.DataConfig()

  def _get_dataset_specs(self, datasets):
    """ Gets the list of Dataset Specifications of datasets

    Args:
      datasets: A list of datasets to load the dataset specification files

    Returns:
      A list of Dataset Specifications or a Dataset Specification, if only one dataset is provided
    
    """
    if isinstance(datasets, list):
      dataset_specs = []
      for dataset_name in datasets:
          dataset_records_path = os.path.join(self.data_path, dataset_name)
          dataset_spec = dataset_spec_lib.load_dataset_spec(dataset_records_path)
          dataset_specs.append(dataset_spec)
      return dataset_specs
    else:
      dataset_name = datasets
      dataset_records_path = os.path.join(self.data_path, dataset_name)
      dataset_spec = dataset_spec_lib.load_dataset_spec(dataset_records_path)
      return dataset_spec

  @gin.configurable("datasets")
  def _get_datasets(self, train_datasets, val_datasets, test_datasets):
    """Gets the list of dataset names.

    Args:
      train_datasets: A string of comma-separated dataset names for training.
      val_datasets: A string of comma-separated dataset names for validation.
      test_datasets: A string of comma-separated dataset names for evaluation.

    Returns:
      Three lists of dataset names
    """

    assert (train_datasets and val_datasets and test_datasets) is not None
    assert isinstance((train_datasets and val_datasets and test_datasets), str)

    train_datasets = [d.strip() for d in train_datasets.split(',')]
    val_datasets = [d.strip() for d in val_datasets.split(',')]
    test_datasets = [d.strip() for d in test_datasets.split(',')]
    
    return train_datasets, val_datasets, test_datasets

class MetaDatasetEpisodeReader(MetaDatasetReader):
  """
  Class that wraps the Meta-Dataset episode readers.
  """
  def __init__(self, data_path, mode, shuffle, num_support=None, num_query=None):
    """
    Initializes a MetaDatasetEpisodeReader

    Args:
      data_path: A string, the path to the Records Folder 
      mode: A string, the mode of the data processing pipeline. Options: "train", "val" or "test" modes. 
      shuffle: A bool, decides to shuffle or not the datasets from which to extract the data and also the samples from each dataset
    """
    super().__init__(data_path, mode, shuffle)
    
    if self.mode == 'train':
      self.split = learning_spec.Split.TRAIN
      dataset_spec = self._get_dataset_specs(self.train_datasets)

      if len(self.train_datasets) == 1:
        # Create an Episode Description that is an instance of EpisodeDescriptionConfig that stores the attributes
        # present in the meta_dataset_config gin file + the given attributes
        num_ways = dataset_spec.classes_per_split[self.split]
        self.episode_description = config.EpisodeDescriptionConfig(num_ways=num_ways, num_support=num_support, num_query=num_query, min_ways=num_ways) 
        print(f"TrainDataset: {dataset_name} -> Num_ways = {self.episode_description.num_ways}, Num_support = {self.episode_description.num_support}, Num_query = {self.episode_description.num_query}")
        self.episodic_dataset = self.create_singlesource_episode_dataset(self.train_datasets)
      else:
        self.episode_description = []
        for idx,spec in enumerate(dataset_spec):
          num_ways = spec.classes_per_split[self.split]
          self.episode_description.append(config.EpisodeDescriptionConfig(num_ways=num_ways, num_support=num_support, num_query=num_query, min_ways=num_ways) ) 
          print(f"TrainDataset: {spec.name} -> Num_ways = {self.episode_description[idx].num_ways}, Num_support = {self.episode_description[idx].num_support}, Num_query = {self.episode_description[idx].num_query}")
        self.episodic_dataset = self.create_multisource_episode_dataset(self.train_datasets, dataset_spec)

    elif self.mode == 'val':
      self.split = learning_spec.Split.VALID
      for dataset_name in self.val_datasets:
        dataset_spec = self._get_dataset_specs(dataset_name)
        
        num_ways = dataset_spec.classes_per_split[self.split]
        self.episode_description = config.EpisodeDescriptionConfig(num_ways=num_ways, num_support=num_support, num_query=num_query, min_ways=num_ways) 
        print(f"ValidationDataset: {dataset_name} -> Num_ways = {self.episode_description.num_ways}, Num_support = {self.episode_description.num_support}, Num_query = {self.episode_description.num_query}")
        dataset = self.create_singlesource_episode_dataset(dataset_name, dataset_spec)
        self.datasets_dict[dataset_name] = dataset

    elif self.mode == 'test':
      self.split = learning_spec.Split.TEST

      for dataset_name in self.test_datasets:
        dataset_spec = self._get_dataset_specs(dataset_name)
                
        num_ways = dataset_spec.classes_per_split[self.split]
        self.episode_description = config.EpisodeDescriptionConfig(num_ways=num_ways, num_support=num_support, num_query=num_query, min_ways=num_ways) 
        print(f"TestDataset: {dataset_name} -> Num_ways = {self.episode_description.num_ways}, Num_support = {self.episode_description.num_support}, Num_query = {self.episode_description.num_query}")
        dataset = self.create_singlesource_episode_dataset(dataset_name, dataset_spec)
        self.datasets_dict[dataset_name] = dataset

    else:
      raise Exception("Invalid Mode. The available modes are: 'train', 'val' or 'test'.")

  def get_train_iterator(self, n):
    if self.mode == "train":
      for idx, (episode, source_id) in enumerate(self.episodic_dataset):
        if idx == n:
          break
        task_dict = {
          'support_images': episode[0].numpy(),
          'support_labels': episode[1].numpy(),
          'support_gt': episode[2].numpy(),
          'query_images': episode[3].numpy(),
          'query_labels': episode[4].numpy(),
          'query_gt': episode[5].numpy()
          }  
        source_id.numpy()

        yield idx, (task_dict, source_id)
      else:
        raise Exception("Invalid Iterator. Use get_val_or_test_iterator")

  def get_val_or_test_iterator(self, dataset, n):
    if self.mode == "val" or self.mode == "test":
      for idx, (episode, source_id) in enumerate(self.datasets_dict[dataset]):
        if idx == n:
          break
        task_dict = {
          'support_images': episode[0].numpy(),
          'support_labels': episode[1].numpy(),
          'support_gt': episode[2].numpy(),
          'query_images': episode[3].numpy(),
          'query_labels': episode[4].numpy(),
          'query_gt': episode[5].numpy()
          }  
        source_id.numpy()

        yield idx, (task_dict, source_id)
      else:
        raise Exception("Invalid Iterator. Use get_val_or_test_iterator")

  def create_multisource_episode_dataset(self, datasets, dataset_specs):
    """ Create a multi source episode dataset, aka, pipeline
    
    Each episode only contains data from one single source. For each episode, its
    source is sampled uniformly across all sources.

    Args:
      datasets: A list, datasets to include in the pipeline
    
    Returns:
      A Dataset instance that outputs tuples of fully-assembled and decoded
      episodes zipped with the ID of their data source of origin.
    """
    use_bilevel_ontology_list = [False]*len(datasets)
    use_dag_ontology_list = [False]*len(datasets)

    # Enable ontology aware sampling for Omniglot and ImageNet. 
    if 'omniglot' in datasets and self.episode_description.num_ways is None and self.episode_description.min_examples_in_class == 0:
      omniglot_index = datasets.index('omniglot')
      use_bilevel_ontology_list[omniglot_index] = True
    if 'ilsvrc_2012' in datasets and self.episode_description.num_ways is None:
      imagenet_index = datasets.index('ilsvrc_2012')
      use_dag_ontology_list[imagenet_index] = True

    if self.shuffle:
      episodic_dataset = pipeline.make_multisource_episode_pipeline(
        dataset_spec_list=dataset_specs, 
        use_dag_ontology_list=use_dag_ontology_list,
        use_bilevel_ontology_list=use_bilevel_ontology_list,
        split=self.split, 
        episode_descr_configs=self.episode_description,
        image_size=self.data_config.image_height, 
        num_prefetch=self.data_config.num_prefetch,
        shuffle_buffer_size=self.data_config.shuffle_buffer_size,
        read_buffer_size_bytes=self.data_config.read_buffer_size_bytes
      )
    else:
      episodic_dataset = pipeline.make_multisource_episode_pipeline(
        dataset_spec_list=dataset_specs, 
        use_dag_ontology_list=use_dag_ontology_list,
        use_bilevel_ontology_list=use_bilevel_ontology_list,
        split=self.split, 
        episode_descr_configs=self.episode_description,
        image_size=self.data_config.image_height,
        num_prefetch=self.data_config.num_prefetch,
        read_buffer_size_bytes=self.data_config.read_buffer_size_bytes, 
        source_sampling_seed=SAMPLING_SEED,
        episode_sampling_seed=SAMPLING_SEED
      )

    return episodic_dataset

  def create_singlesource_episode_dataset(self, dataset, dataset_spec):
    """ Create a single source episode dataset, aka, pipeline
    
    Each episode only contains data from one single source. For each episode, its
    source is sampled uniformly across all sources.

    Args:
      dataset: A string, name of the dataset to include in the pipeline
    
    Returns:
      A Dataset instance that outputs tuples of fully-assembled and decoded
      episodes zipped with the ID of their data source of origin.
    """
    use_bilevel_ontology = False
    use_dag_ontology = False

    # Enable ontology aware sampling for Omniglot and ImageNet. 
    if 'omniglot' == dataset and self.episode_description.num_ways is None and self.episode_description.min_examples_in_class == 0:
      use_bilevel_ontology = True
    if 'ilsvrc_2012' == dataset and self.episode_description.num_ways is None:
      use_dag_ontology = True

    if self.shuffle:
      episodic_dataset = pipeline.make_one_source_episode_pipeline(
        dataset_spec = dataset_spec,
        use_dag_ontology = use_dag_ontology,
        use_bilevel_ontology = use_bilevel_ontology,
        split = self.split,
        episode_descr_config = self.episode_description,
        image_size = self.data_config.image_height,
        num_prefetch = self.data_config.num_prefetch, 
        shuffle_buffer_size = self.data_config.shuffle_buffer_size,
        read_buffer_size_bytes = self.data_config.read_buffer_size_bytes
      )
    else:
      episodic_dataset = pipeline.make_one_source_episode_pipeline(
        dataset_spec = dataset_spec,
        use_dag_ontology = use_dag_ontology,
        use_bilevel_ontology = use_bilevel_ontology,
        split = self.split,
        episode_descr_config = self.episode_description,
        image_size = self.data_config.image_height,
        num_prefetch = self.data_config.num_prefetch, 
        read_buffer_size_bytes = self.data_config.read_buffer_size_bytes, 
        episode_sampling_seed=SAMPLING_SEED
      )

    return episodic_dataset