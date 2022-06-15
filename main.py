from data.meta_dataset_reader import MetaDatasetEpisodeReader
from paths import META_RECORDS_ROOT
from tqdm import tqdm
train_loader = MetaDatasetEpisodeReader(META_RECORDS_ROOT, "train", False)

# for i, episode in tqdm(train_loader.get_iterator(10000)):
#     ip = episode

iterator = iter(train_loader.episodic_dataset)
for i in tqdm(range(10000)):
    idx, _=next(iterator)