from data.meta_dataset_reader import MetaDatasetEpisodeReader
from paths import META_RECORDS_ROOT
from tqdm import tqdm

from data.plots import plot_episode

train_loader = MetaDatasetEpisodeReader(META_RECORDS_ROOT, "train", False)


for idx, (task, source_id) in train_loader.get_iterator(2):
  print('Episode id: %d from source %s' % (idx, train_loader.dataset_specs[source_id].name ))
  
  plot_episode(support_images=task['support_images'], support_class_ids=task['support_gt'],
               query_images=task['query_images'], query_class_ids=task['query_gt'])

