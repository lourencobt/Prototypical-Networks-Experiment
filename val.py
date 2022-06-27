#%%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ""

from data.meta_dataset_reader import MetaDatasetEpisodeReader
from paths import META_RECORDS_ROOT
from data.plots import plot_episode

#%%
dataloader = MetaDatasetEpisodeReader(META_RECORDS_ROOT, "val", False)

#%%
# dataset = dataloader.datasets_dict["aircraft"]

# for idx, (episode, source_id) in enumerate(dataset):
#     if idx == 5: 
#         break
#     task = {
#         'support_images': episode[0].numpy(),
#         'support_labels': episode[1].numpy(),
#         'support_gt': episode[2].numpy(),
#         'query_images': episode[3].numpy(),
#         'query_labels': episode[4].numpy(),
#         'query_gt': episode[5].numpy()
#         }  
#     print('Episode id: %d from source %s' % (idx, source_id ))
#     plot_episode(support_images=task['support_images'], support_class_ids=task['support_labels'], query_images=task['query_images'], query_class_ids=task['query_labels'],
#         max_imgs_per_col=100,max_imgs_per_row=100)

for idx, task in dataloader.get_val_or_test_iterator("fungi", 2):
    print(f'source: ")
    # print('Episode id: %d from source %s' % (idx, dataloader.dataset_spec.name ))
    # plot_episode(support_images=task['support_images'], support_class_ids=task['support_labels'], query_images=task['query_images'], query_class_ids=task['query_labels'],
    #     max_imgs_per_col=100,max_imgs_per_row=100)

#   def _to_torch(self, sample, source_id):
#     for key, val in sample.items():
#         if isinstance(val, str):
#             continue
#         val = torch.from_numpy(val.numpy())
#         if 'image' in key:
#             val = val.permute(0, 3, 1, 2)
#         else:
#             val = val.long()
#         sample[key] = val
#     source_id = source_id.numpy()
#     return sample, source_id