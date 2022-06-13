from collections import Counter
import matplotlib.pyplot as plt
import numpy as np


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