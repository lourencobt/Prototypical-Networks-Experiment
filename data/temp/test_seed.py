#%%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from data.meta_dataset_reader import MetaDatasetEpisodeReader
from paths import META_RECORDS_ROOT

from data.plots import plot_episode

train_loader = MetaDatasetEpisodeReader(META_RECORDS_ROOT, "train", False)

for idx, (episode, source_id) in enumerate(train_loader.episodic_dataset):
    if idx == 1:
        break
    print(episode[0])

    print('Episode id: %d from source %s' % (idx, train_loader.dataset_specs[source_id].name ))
    episode = [a.numpy() for a in episode]
    plot_episode(support_images=episode[0], support_class_ids=episode[2],
                query_images=episode[3], query_class_ids=episode[5])
# %%
