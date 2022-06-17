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
tf.config.experimental.set_memory_growth(physical_devices[0], True)

import torch
from paths import META_DATASET_ROOT, GIN_CONFIG_ROOT
from utils import device

sys.path.append(META_DATASET_ROOT)
from meta_dataset.data import config
from meta_dataset.data import dataset_spec as dataset_spec_lib
from meta_dataset.data import learning_spec
from meta_dataset.data import pipeline

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

    # ! TO REMOVE
    # gin.enter_interactive_mode()

    gin.parse_config_file(GIN_CONFIG_ROOT)
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

  def _to_torch(self, sample):
    for key, val in sample.items():
        if isinstance(val, str):
            continue
        val = torch.from_numpy(val.numpy())
        if 'image' in key:
            val = val.permute(0, 3, 1, 2)
        else:
            val = val.long()
        sample[key] = val.to(device)
    return sample

class MetaDatasetEpisodeReader(MetaDatasetReader):
  """
  Class that wraps the Meta-Dataset episode readers.
  """
  def __init__(self, data_path, mode, shuffle):
    """
    Initializes a MetaDatasetEpisodeReader

    Args:
      data_path: A string, the path to the Records Folder 
      mode: A string, the mode of the data processing pipeline. Options: "train", "val" or "test" modes. 
      shuffle: A bool, shuffle or not the dataset samples
    """
    super().__init__(data_path, mode, shuffle)
    # Create an Episode Description that is an instance of EpisodeDescriptionConfig that stores the attributes
    # present in the meta_dataset_config gin file
    self.episode_description = config.EpisodeDescriptionConfig()

    # ! Use single dataset for fixed 'num_ways', since using multiple datasets is not supported/tested. 
    if self.mode == 'train':
      self.split = learning_spec.Split.TRAIN
      if len(self.train_datasets) == 1:
        self.episodic_dataset = self.create_singlesource_episode_dataset(self.train_datasets)
      else:
        self.episodic_dataset = self.create_multisource_episode_dataset(self.train_datasets)

    elif self.mode == 'val':
      self.split = learning_spec.Split.VALID

      for dataset_name in self.val_datasets:
        dataset = self.create_singlesource_episode_dataset(dataset_name)
        self.datasets_dict[dataset_name] = dataset

    elif self.mode == 'test':
      self.split = learning_spec.Split.TEST

      for dataset_name in self.test_datasets:
        dataset = self.create_singlesource_episode_dataset(dataset_name)
        self.datasets_dict[dataset_name] = dataset

    else:
      raise Exception("Invalid Mode. The available modes are: 'train', 'val' or 'test'.")

  def get_iterator(self, n):
    for idx, (episode, source_id) in enumerate(self.episodic_dataset):
      if idx == n:
        break
      task_dict = {
        'support_images': episode[0],
        'support_labels': episode[1],
        'support_gt': episode[2],
        'query_images': episode[3],
        'query_labels': episode[4],
        'query_gt': episode[5]
        }  
      yield idx, (self._to_torch(task_dict), source_id)

  def create_multisource_episode_dataset(self, datasets):
    """ Create a multi source episode dataset, aka, pipeline
    
    Each episode only contains data from one single source. For each episode, its
    source is sampled uniformly across all sources.

    Args:
      datasets: A list, datasets to include in the pipeline
    
    Returns:
      A Dataset instance that outputs tuples of fully-assembled and decoded
      episodes zipped with the ID of their data source of origin.
    """
    self.dataset_specs = self._get_dataset_specs(self.train_datasets)

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
        dataset_spec_list=self.dataset_specs, 
        use_dag_ontology_list=use_dag_ontology_list,
        use_bilevel_ontology_list=use_bilevel_ontology_list,
        split=self.split, 
        episode_descr_config=self.episode_description,
        image_size=self.data_config.image_height, 
        num_prefetch=self.data_config.num_prefetch,
        shuffle_buffer_size=self.data_config.shuffle_buffer_size,
        read_buffer_size_bytes=self.data_config.read_buffer_size_bytes
      )
    else:
      episodic_dataset = pipeline.make_multisource_episode_pipeline(
        dataset_spec_list=self.dataset_specs, 
        use_dag_ontology_list=use_dag_ontology_list,
        use_bilevel_ontology_list=use_bilevel_ontology_list,
        split=self.split, 
        episode_descr_config=self.episode_description,
        image_size=self.data_config.image_height,
        num_prefetch=self.data_config.num_prefetch,
        read_buffer_size_bytes=self.data_config.read_buffer_size_bytes
      )

    return episodic_dataset

  def create_singlesource_episode_dataset(self, dataset):
    """ Create a single source episode dataset, aka, pipeline
    
    Each episode only contains data from one single source. For each episode, its
    source is sampled uniformly across all sources.

    Args:
      dataset: A string, name of the dataset to include in the pipeline
    
    Returns:
      A Dataset instance that outputs tuples of fully-assembled and decoded
      episodes zipped with the ID of their data source of origin.
    """

    self.dataset_spec = self._get_dataset_specs(dataset)

    use_bilevel_ontology = False
    use_dag_ontology = False

    # Enable ontology aware sampling for Omniglot and ImageNet. 
    if 'omniglot' == dataset and self.episode_description.num_ways is None and self.episode_description.min_examples_in_class == 0:
      use_bilevel_ontology = True
    if 'ilsvrc_2012' == dataset and self.episode_description.num_ways is None:
      use_dag_ontology = True

    if self.shuffle:
      episodic_dataset = pipeline.make_one_source_episode_pipeline(
        dataset_spec = self.dataset_spec,
        use_dag_ontology = use_dag_ontology,
        use_bilevel_ontology = use_bilevel_ontology,
        split = self.split,
        episode_descr_config = self.episode_description,
        image_size = self.data_config.image_height,
        num_prefetch = self.data_config.num_prefetch, 
        suffle_buffer_size = self.data_config.shuffle_buffer_size,
        read_buffer_size_bytes = self.data_config.read_buffer_size_bytes
      )
    else:
        episodic_dataset = pipeline.make_one_source_episode_pipeline(
        dataset_spec = self.dataset_spec,
        use_dag_ontology = use_dag_ontology,
        use_bilevel_ontology = use_bilevel_ontology,
        split = self.split,
        episode_descr_config = self.episode_description,
        image_size = self.data_config.image_height,
        num_prefetch = self.data_config.num_prefetch, 
        read_buffer_size_bytes = self.data_config.read_buffer_size_bytes
      )

    return episodic_dataset