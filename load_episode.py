#%%
import os
# ! TO REMOVE
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from data.meta_dataset_reader import MetaDatasetEpisodeReader
from paths import META_RECORDS_ROOT
from tqdm import tqdm
from utils import device
import pickle
import torch
import torchvision.models as tv_models
from torchvision import transforms as tr
from models.proto_net import PrototypicalNetwork

from data.plots import plot_episode

with open('dataset224.txt', 'rb') as f:
  dataset = pickle.load(f)

encoder = tv_models.resnet34(pretrained=True, progress=True)

for param in encoder.parameters():
    param.requires_grad = False

encoder.eval()
encoder = encoder.to(device)

model = PrototypicalNetwork(encoder, 'l2', torch.nn.CrossEntropyLoss())

#%%
loss_acc = 0
accuracy_acc = 0
for idx, (task, source_id) in dataset:
  for key, val in task.items():
    val = torch.from_numpy(val)
    if 'image' in key:
      T = tr.Compose([
        tr.Lambda(lambda x: (x+1)/2), 
        tr.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
        ])
      val = T(val)
    val = val.to(device)
    task[key] = val
    
  support_images=task['support_images']
  support_labels=task['support_labels']
  query_images=task['query_images']
  query_labels=task['query_labels']
  
  # print(support_images.shape)
  # print(support_labels.shape)
  # print(query_images.shape)
  # print(query_labels.shape)
  
  model(support_images, support_labels, query_images, query_labels)
  
  loss = model.loss()
  loss_acc += float(loss)
  accuracy = model.accuracy()
  accuracy_acc += float(accuracy)
  
  print( f'Acc: {accuracy}\t, Loss: {loss}')


print(f'\nMedian_Acc: {accuracy_acc/10}\t, Median_Loss: {loss_acc/10}')



# %%
