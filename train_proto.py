#%%
from __future__ import print_function, division

import os
from datetime import datetime
from utils import device

# ! TO REMOVE
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from data.meta_dataset_reader import MetaDatasetEpisodeReader
from paths import META_RECORDS_ROOT

import torch
import torch.nn as nn
import numpy as np
from torchvision import  models, transforms
import matplotlib.pyplot as plt
from models.proto_net import PrototypicalNetwork

NUM_TRAIN_TASKS = 2000
NUM_VALIDATION_TASKS = 5
VALIDATION_FREQUENCY = 200

#Load Data
train_loader = MetaDatasetEpisodeReader(META_RECORDS_ROOT, "train", False)
val_loader = MetaDatasetEpisodeReader(META_RECORDS_ROOT, "val", False)

T = transforms.Compose([
    transforms.Lambda(lambda x: (x+1)/2), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
    ])

def get_iterator(dataloader, n, transforms=None):
    for idx, (task, source_id) in dataloader.get_train_iterator(n):
        for key, val in task.items():
            if isinstance(val, str):
                continue
            val = torch.from_numpy(val)
            if 'image' in key:
                val = val.permute(0, 3, 1, 2)
                if transforms != None:
                    val = transforms(val)
            else:
                val = val.long()
            task[key] = val
        yield idx, (task, source_id)

def get_val_or_test_iterator(dataloader, dataset, n, transforms=None):  
    for idx, (task, source_id) in dataloader.get_val_or_test_iterator(dataset, n):
        for key, val in task.items():
            if isinstance(val, str):
                continue
            val = torch.from_numpy(val)
            if 'image' in key:
                val = val.permute(0, 3, 1, 2)
                if transforms != None:
                    val = transforms(val)
            else:
                val = val.long()
            task[key] = val
        yield idx, (task, source_id)

def train():
    #Model 
    encoder = models.resnet18(pretrained=True)
    # for encoder_parm in encoder.parameters():
    #     encoder_parm.requires_grad = False

    # encoder.fc.requires_grad_()

    model = PrototypicalNetwork(encoder, 'l2').to(device)

    # Optimizing parameters
    learning_rate = 0.005
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    
    losses = []
    train_accuracies = []
    total_iterations = NUM_TRAIN_TASKS

    for iteration, (task, source_id) in get_iterator(train_loader, NUM_TRAIN_TASKS, T):
        model.train()
        support_images=task['support_images'].to(device)
        support_labels=task['support_labels'].to(device)
        query_images=task['query_images'].to(device)
        query_labels=task['query_labels'].to(device)

        # Compute prediction error
        target_logits = model(support_images, support_labels, query_images)
        task_loss = loss_fn(target_logits, query_labels)
        task_accuracy = model.accuracy(target_logits, query_labels)

        train_accuracies.append(task_accuracy)
        losses.append(task_loss)

        # ! Optimize per batch?
        # Backpropagation
        task_loss.backward()
        optimizer.zero_grad()
        optimizer.step()    

        if (iteration + 1) % 2 == 0:
            # print training stats
            print('[{}] Task [{}/{}], Train Loss: {:.7f}, Train Accuracy: {:.7f}'
                            .format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),iteration + 1, total_iterations, torch.Tensor(losses).mean().item(),
                                    torch.Tensor(train_accuracies).mean().item()))
            train_accuracies = []
            losses = []
        
        if ((iteration + 1) % VALIDATION_FREQUENCY == 0) and (iteration + 1) != total_iterations:
            validate(model)

    # #save the final model
    # torch.save(model.state_dict(), self.checkpoint_path_final)

def validate(model):
    model.eval()
    with torch.no_grad():
        accuracy_dict = {}

        for dataset in val_loader.val_datasets:

            accuracies = []
            for iteration, (task, source_id) in get_val_or_test_iterator(val_loader, dataset, NUM_VALIDATION_TASKS, T):
                support_images=task['support_images'].to(device)
                support_labels=task['support_labels'].to(device)
                query_images=task['query_images'].to(device)
                query_labels=task['query_labels'].to(device)

                target_logits = model(support_images, support_labels, query_images)
                task_accuracy = model.accuracy(target_logits, query_labels)

                accuracies.append(task_accuracy.cpu())
                accuracies = accuracies
        
            accuracy = np.array(accuracies).mean() * 100.0
            accuracy_confidence = (196.0 * np.array(accuracies).std()) / np.sqrt(len(accuracies))

            print("{0:}: {1:.1f}+/-{2:.1f}".format(dataset, accuracy, accuracy_confidence))

        

if __name__ == '__main__':
    train()

#  with tf.compat.v1.Session(config=config) as session:
#             if self.args.mode == 'train' or self.args.mode == 'train_test':
#                 train_accuracies = []
#                 losses = []
#                 total_iterations = NUM_TRAIN_TASKS
#                 for iteration in range(total_iterations):
#                     torch.set_grad_enabled(True)
#                     task_dict = self.metadataset.get_train_task()
#                     task_loss, task_accuracy = self.train_task(task_dict)
#                     train_accuracies.append(task_accuracy)
#                     losses.append(task_loss)

#                     # optimize
#                     if ((iteration + 1) % self.args.tasks_per_batch == 0) or (iteration == (total_iterations - 1)):
#                         self.optimizer.step()
#                         self.optimizer.zero_grad()

#                     if (iteration + 1) % 1000 == 0:
#                         # print training stats
#                         print_and_log(self.logfile,'Task [{}/{}], Train Loss: {:.7f}, Train Accuracy: {:.7f}'
#                                       .format(iteration + 1, total_iterations, torch.Tensor(losses).mean().item(),
#                                               torch.Tensor(train_accuracies).mean().item()))
#                         train_accuracies = []
#                         losses = []

#                     if ((iteration + 1) % VALIDATION_FREQUENCY == 0) and (iteration + 1) != total_iterations:
#                         # validate
#                         accuracy_dict = self.validate(session)
#                         self.validation_accuracies.print(self.logfile, accuracy_dict)
#                         # save the model if validation is the best so far
#                         if self.validation_accuracies.is_better(accuracy_dict):
#                             self.validation_accuracies.replace(accuracy_dict)
#                             torch.save(self.model.state_dict(), self.checkpoint_path_validation)
#                             print_and_log(self.logfile, 'Best validation model was updated.')
#                             print_and_log(self.logfile, '')

#                 # save the final model
#                 torch.save(self.model.state_dict(), self.checkpoint_path_final)

#             if self.args.mode == 'train_test':
#                 self.test(self.checkpoint_path_final, session)
#                 self.test(self.checkpoint_path_validation, session)

#             if self.args.mode == 'test':
#                 self.test(self.args.test_model_path, session)

#             self.logfile.close()


#     def test(self, path, session):
#         self.model = self.init_model()
#         self.model.load_state_dict(torch.load(path))
#         print_and_log(self.logfile, "")  # add a blank line
#         print_and_log(self.logfile, 'Testing model {0:}: '.format(path))

#         with torch.no_grad():
#             for item in self.test_set:
#                 accuracies = []
#                 for _ in range(NUM_TEST_TASKS):
#                     task_dict = self.metadataset.get_test_task(item)
#                     context_images, target_images, context_labels, target_labels = self.prepare_task(task_dict)
#                     target_logits = self.model(context_images, context_labels, target_images)
#                     accuracy = self.accuracy_fn(target_logits, target_labels)
#                     accuracies.append(accuracy.item())
#                     del target_logits

#                 accuracy = np.array(accuracies).mean() * 100.0
#                 accuracy_confidence = (196.0 * np.array(accuracies).std()) / np.sqrt(len(accuracies))

#                 print_and_log(self.logfile, '{0:}: {1:3.1f}+/-{2:2.1f}'.format(item, accuracy, accuracy_confidence))
