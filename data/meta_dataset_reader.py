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
    assert isinstance(mode, str)
    self.mode = mode
    assert isinstance(shuffle, bool)
    self.shuffle = shuffle
    # ! TO REMOVE
    gin.enter_interactive_mode()
    gin.parse_config_file(GIN_CONFIG_ROOT)
    self.train_datasets, self.val_datasets, self.test_datasets = self._get_datasets()
    
    # ! Confirm that its good here. Confirm that it is applicable both to Episode and Batch Reader
    self.data_config = config.DataConfig()

  def _get_dataset_specs(self, datasets):
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
  def __init__(self, data_path, mode, shuffle):
    super().__init__(data_path, mode, shuffle)
    # Create an Episode Description that is an instance of EpisodeDescriptionConfig that stores the attributes
    # present in the meta_dataset_config gin file
    self.episode_description = config.EpisodeDescriptionConfig()

    # ! Do not forget that to use fixed 'num_ways', disable ontology based sampling for omniglot and imagenet.
    # ! Use single dataset for fixed 'num_ways', since using multiple datasets is not supported/tested. 

    if self.mode == 'train':
      self.dataset_specs = self._get_dataset_specs(self.train_datasets)
      self.split = learning_spec.Split.TRAIN
      if len(self.train_datasets) == 1:
        # create single source dataset (INPUT self.shuffle)
        pass
      else:
        # create multi source dataset (INPUT self.shuffle)
        self.episodic_dataset = self.create_multisource_episode_pipeline(self.train_datasets, True, True)
        print(self.episodic_dataset)
        pass

    elif self.mode == 'val':
      self.dataset_specs = self._get_dataset_specs(self.val_datasets)
      self.split = learning_spec.Split.VALID
      if len(self.val_datasets) == 1:
        # create single source dataset (INPUT self.shuffle)
        pass
      else:
        # create multi source dataset (INPUT self.shuffle)
        pass

    elif self.mode == 'test':
      self.dataset_specs = self._get_dataset_specs(self.test_datasets)
      self.split = learning_spec.Split.TEST
      # ! Probably here this doesn't make sense. Probably you evaluate one dataset at a time
      if len(self.test_datasets) == 1:
        # create single source dataset (INPUT self.shuffle)
        pass
      else:
        # create multi source dataset (INPUT self.shuffle)
        pass
      pass

    else:
      raise Exception("Invalid Mode. The available modes are: 'train', 'val' or 'test'.")

  def create_multisource_episode_pipeline(self, datasets,  use_bilevel_ontology, use_dag_ontology):
    use_bilevel_ontology_list = [False]*len(datasets)
    use_dag_ontology_list = [False]*len(datasets)

    # Enable ontology aware sampling for Omniglot and ImageNet. 
    if 'omniglot' in datasets and use_bilevel_ontology == True:
      omniglot_index = datasets.index('omniglot')
      use_bilevel_ontology_list[omniglot_index] = True
    if 'ilsvrc_2012' in datasets and use_dag_ontology == True:
      imagenet_index = datasets.index('ilsvrc_2012')
      use_dag_ontology_list[imagenet_index] = True

    if self.shuffle:
      dataset_episodic = pipeline.make_multisource_episode_pipeline(
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
      dataset_episodic = pipeline.make_multisource_episode_pipeline(
        dataset_spec_list=self.dataset_specs, 
        use_dag_ontology_list=use_dag_ontology_list,
        use_bilevel_ontology_list=use_bilevel_ontology_list,
        split=self.split, 
        episode_descr_config=self.episode_description,
        image_size=self.data_config.image_height,
        num_prefetch=self.data_config.num_prefetch,
        read_buffer_size_bytes=self.data_config.read_buffer_size_bytes
      )

    return dataset_episodic


#%%
dataset = MetaDatasetEpisodeReader(META_RECORDS_ROOT, "train", False)