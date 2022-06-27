#%%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from data.meta_dataset_reader import MetaDatasetEpisodeReader
from paths import META_RECORDS_ROOT
from tqdm import tqdm
import pickle

from data.plots import plot_episode

train_loader = MetaDatasetEpisodeReader(META_RECORDS_ROOT, "train", False)

dataset = []
for idx, (task, source_id) in tqdm(train_loader.get_iterator(10)): 
  for key, val in task.items():
    task[key] = val.cpu().numpy()
  
  dataset.append((idx, (task, source_id)))

#%%
with open('dataset224.txt', 'wb') as f:
  pickle.dump(dataset, f)

  # print('Episode id: %d from source %s' % (idx, train_loader.dataset_specs[source_id].name ))
  
  # plot_episode(support_images=task['support_images'], support_class_ids=task['support_gt'],
  #              query_images=task['query_images'], query_class_ids=task['query_gt'])

