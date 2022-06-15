# #@title Imports and Utility Functions
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import sys
import os
from collections import Counter
import gin
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

os.environ['META_DATASET_ROOT'] = "/home/guests/lbt/meta-dataset"
META_DATASET_ROOT = os.environ['META_DATASET_ROOT']
sys.path.append(META_DATASET_ROOT)
from meta_dataset.data import config
from meta_dataset.data import dataset_spec as dataset_spec_lib
from meta_dataset.data import learning_spec
from meta_dataset.data import pipeline

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def plot_episode(support_images, support_class_ids, query_images,
                 query_class_ids, size_multiplier=1, max_imgs_per_col=10,
                 max_imgs_per_row=10):
  for name, images, class_ids in zip(('Support', 'Query'),
                                     (support_images, query_images),
                                     (support_class_ids, query_class_ids)):
    n_samples_per_class = Counter(class_ids)
    n_samples_per_class = {k: min(v, max_imgs_per_col)
                           for k, v in n_samples_per_class.items()}
    id_plot_index_map = {k: i for i, k
                         in enumerate(n_samples_per_class.keys())}
    num_classes = min(max_imgs_per_row, len(n_samples_per_class.keys()))
    max_n_sample = max(n_samples_per_class.values())
    figwidth = max_n_sample
    figheight = num_classes
    if name == 'Support':
      print('#Classes: %d' % len(n_samples_per_class.keys()))
    figsize = (figheight * size_multiplier, figwidth * size_multiplier)
    fig, axarr = plt.subplots(
        figwidth, figheight, figsize=figsize)
    fig.suptitle('%s Set' % name, size='20')
    fig.tight_layout(pad=3, w_pad=0.1, h_pad=0.1)
    reverse_id_map = {v: k for k, v in id_plot_index_map.items()}
    for i, ax in enumerate(axarr.flat):
      ax.patch.set_alpha(0)
      # Print the class ids, this is needed since, we want to set the x axis
      # even there is no picture.
      ax.set(xlabel=reverse_id_map[i % figheight], xticks=[], yticks=[])
      ax.label_outer()
    for image, class_id in zip(images, class_ids):
      # First decrement by one to find last spot for the class id.
      n_samples_per_class[class_id] -= 1
      # If class column is filled or not represented: pass.
      if (n_samples_per_class[class_id] < 0 or
          id_plot_index_map[class_id] >= max_imgs_per_row):
        continue
      # If width or height is 1, then axarr is a vector.
      if axarr.ndim == 1:
        ax = axarr[n_samples_per_class[class_id]
                   if figheight == 1 else id_plot_index_map[class_id]]
      else:
        ax = axarr[n_samples_per_class[class_id], id_plot_index_map[class_id]]
      ax.imshow(image / 2 + 0.5)
    plt.show()

def plot_batch(images, labels, size_multiplier=1):
  num_examples = len(labels)
  figwidth = np.ceil(np.sqrt(num_examples)).astype('int32')
  figheight = num_examples // figwidth
  figsize = (figwidth * size_multiplier, (figheight + 1.5) * size_multiplier)
  _, axarr = plt.subplots(figwidth, figheight, dpi=300, figsize=figsize)

  for i, ax in enumerate(axarr.transpose().ravel()):
    # Images are between -1 and 1.
    ax.imshow(images[i] / 2 + 0.5)
    ax.set(xlabel=labels[i], xticks=[], yticks=[])
  plt.show()

  # 1
BASE_PATH = '/home/guests/lbt/data/records'
os.chdir('/home/guests/lbt/meta-dataset')
GIN_FILE_PATH = 'meta_dataset/learn/gin/setups/data_config.gin'

gin.parse_config_file(GIN_FILE_PATH)

os.chdir('/home/guests/lbt/few-shot/data/temp')
# 3
# Comment out to disable eager execution.
# tf.compat.v1.enable_eager_execution()
# 4
# ! Function that makes easier to iterate in a dataset. 
# ! If not executing eagerly, there is less abstraction and there is the need to make the iterator.
# ! If executing eagerly, just doing the for is sufficient
def iterate_dataset(dataset, n):
  if not tf.executing_eagerly():
    print("Not Executing Eagerly")
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    with tf.Session() as sess:
      for idx in range(n):
        yield idx, sess.run(next_element)
  else:
    print("Executing Eagerly")
    for idx, episode in enumerate(dataset):
      if idx == n:
        break
      yield idx, episode
# 5
SPLIT = learning_spec.Split.TRAIN

ALL_DATASETS = ['aircraft', 'cu_birds', 'dtd', 'fungi', 'ilsvrc_2012',
                'omniglot', 'quickdraw', 'vgg_flower']

# ! This for just loads all the dataset_spec files, that each dataset has. Then, they are all appended to a list.
# ! Each of the dataset_spec is an instance of a DatasetSpecification class, that is present in the file "meta-dataset/data/dataset_spec.py"
all_dataset_specs = []
for dataset_name in ALL_DATASETS:
  dataset_records_path = os.path.join(BASE_PATH, dataset_name)
  dataset_spec = dataset_spec_lib.load_dataset_spec(dataset_records_path)
  all_dataset_specs.append(dataset_spec)

  use_bilevel_ontology_list = [False]*len(ALL_DATASETS)
use_dag_ontology_list = [False]*len(ALL_DATASETS)
# Enable ontology aware sampling for Omniglot and ImageNet. 
use_bilevel_ontology_list[5] = True
use_dag_ontology_list[4] = True

# ! This creates an instance of EpisodeDescriptionConfig that includes the attributes present in the data_config.gin file
variable_ways_shots = config.EpisodeDescriptionConfig(
    num_query=None, num_support=None, num_ways=None)

# ! Creates an episodic dataset. Meaning that the dataset is composed by episodes. This dataset uses tf.data.Dataset API
dataset_episodic = pipeline.make_multisource_episode_pipeline(
    dataset_spec_list=all_dataset_specs,
    use_dag_ontology_list=use_dag_ontology_list,
    use_bilevel_ontology_list=use_bilevel_ontology_list,
    episode_descr_config=variable_ways_shots,
    split=SPLIT,
    image_size=84,
    # shuffle_buffer_size=1000
    )

# # 1
# idx, (episode, source_id) = next(iterate_dataset(dataset_episodic, 1))
# print('Got an episode from dataset:', all_dataset_specs[source_id].name)

# # 2
# for t, name in zip(episode,
#                    ['support_images', 'support_labels', 'support_class_ids',
#                     'query_images', 'query_labels', 'query_class_ids']):
#   print(name, t.shape)

# # 3
# episode = [a.numpy() for a in episode]

# # 4
# support_class_ids, query_class_ids = episode[2], episode[5]
# print(Counter(support_class_ids))
# print(Counter(query_class_ids))


for i, episode in iterate_dataset(dataset_episodic, 100):
    print(i)