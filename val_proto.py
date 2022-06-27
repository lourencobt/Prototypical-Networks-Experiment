#%%
import os
from datetime import datetime
from utils import device

# ! TO REMOVE
os.environ['CUDA_VISIBLE_DEVICES'] = ""

from data.meta_dataset_reader import MetaDatasetEpisodeReader
from paths import META_RECORDS_ROOT

import torch
from torchvision import models, transforms
import matplotlib.pyplot as plt
from models.proto_net import PrototypicalNetwork
import numpy as np

NUM_TRAIN_TASKS = 2000
NUM_VALIDATION_TASKS = 5
NUM_TEST_TASKS = 200
VALIDATION_FREQUENCY = 10000

#Load Data
val_loader = MetaDatasetEpisodeReader(META_RECORDS_ROOT, "val", False)
test_loader = MetaDatasetEpisodeReader(META_RECORDS_ROOT, "test", False)

T = transforms.Compose([
    transforms.Lambda(lambda x: (x+1)/2), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
    ])

def get_iterator(dataloader, dataset, n, transforms=None):  
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
#Model 
encoder = models.resnet18(pretrained=True)

model = PrototypicalNetwork(encoder, 'l2').to(device)

def validate(model):
    model.eval()
    with torch.no_grad():
        accuracy_dict = {}

        for dataset in val_loader.val_datasets:

            accuracies = []
            for iteration, (task, source_id) in get_iterator(val_loader, dataset, NUM_VALIDATION_TASKS, T):
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

def test(model):
        # self.model = self.init_model()
        # self.model.load_state_dict(torch.load(path))
        # print_and_log(self.logfile, "")  # add a blank line
        # print_and_log(self.logfile, 'Testing model {0:}: '.format(path))

        with torch.no_grad():
            for dataset in test_loader.test_datasets:
                accuracies = []
                for iteration, (task, source_id) in get_iterator(test_loader, dataset, NUM_TEST_TASKS, T):
                    support_images=task['support_images'].to(device)
                    support_labels=task['support_labels'].to(device)
                    query_images=task['query_images'].to(device)
                    query_labels=task['query_labels'].to(device)

                    
                    target_logits = model(support_images, support_labels, query_images)
                    task_accuracy = model.accuracy(target_logits, query_labels)

                    accuracies.append(task_accuracy.cpu())
                    accuracies = accuracies
                    del target_logits

                accuracy = np.array(accuracies).mean() * 100.0
                accuracy_confidence = (196.0 * np.array(accuracies).std()) / np.sqrt(len(accuracies))

                print('{0:}: {1:3.1f}+/-{2:2.1f}'.format(dataset, accuracy, accuracy_confidence))

if __name__ == '__main__':
    validate(model)
    test(model)



#     # losses = []
#     # train_accuracies = []
#     # total_iterations = NUM_TRAIN_TASKS

#     # for iteration, (task, source_id) in get_iterator(NUM_TRAIN_TASKS):
#     #     support_images=task['support_images'].to(device)
#     #     support_labels=task['support_labels'].to(device)
#     #     query_images=task['query_images'].to(device)
#     #     query_labels=task['query_labels'].to(device)

#     #     # Compute prediction error
#     #     target_logits = model(support_images, support_labels, query_images)
#     #     task_loss = loss_fn(target_logits, query_labels)
#     #     task_accuracy = model.accuracy(target_logits, query_labels)

#     #     train_accuracies.append(task_accuracy)
#     #     losses.append(task_loss)

#     #     # ! Optimize per batch?
#     #     # Backpropagation
#     #     task_loss.backward()
#     #     optimizer.zero_grad()
#     #     optimizer.step()    

#     #     if (iteration + 1) % 5 == 0:
#     #         # print training stats
#     #         print('[{}] Task [{}/{}], Train Loss: {:.7f}, Train Accuracy: {:.7f}'
#     #                         .format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),iteration + 1, total_iterations, torch.Tensor(losses).mean().item(),
#     #                                 torch.Tensor(train_accuracies).mean().item()))
#     #         train_accuracies = []
#     #         losses = []

#         # if ((iteration + 1) % VALIDATION_FREQUENCY == 0) and (iteration + 1) != total_iterations:
#         #     # validate
#         #     accuracy_dict = self.validate(session)
#         #     self.validation_accuracies.print(self.logfile, accuracy_dict)
#         #     # save the model if validation is the best so far
#         #     if self.validation_accuracies.is_better(accuracy_dict):
#         #         self.validation_accuracies.replace(accuracy_dict)
#         #         torch.save(self.model.state_dict(), self.checkpoint_path_validation)
#         #         print_and_log(self.logfile, 'Best validation model was updated.')
#         #         print_and_log(self.logfile, '')

#     # #save the final model
#     # torch.save(model.state_dict(), self.checkpoint_path_final)


        



# #     def validate(self, session):
# #         with torch.no_grad():
# #             accuracy_dict ={}
# #             for item in self.validation_set:
# #                 accuracies = []
# #                 for _ in range(NUM_VALIDATION_TASKS):
# #                     task_dict = self.metadataset.get_validation_task(item)
# #                     context_images, target_images, context_labels, target_labels = self.prepare_task(task_dict)
# #                     target_logits = self.model(context_images, context_labels, target_images)
# #                     accuracy = self.accuracy_fn(target_logits, target_labels)
# #                     accuracies.append(accuracy.item())
# #                     del target_logits

# #                 accuracy = np.array(accuracies).mean() * 100.0
# #                 confidence = (196.0 * np.array(accuracies).std()) / np.sqrt(len(accuracies))

# #                 accuracy_dict[item] = {"accuracy": accuracy, "confidence": confidence}

# #         return accuracy_dict

# #     def test(self, path, session):
# #         self.model = self.init_model()
# #         self.model.load_state_dict(torch.load(path))
# #         print_and_log(self.logfile, "")  # add a blank line
# #         print_and_log(self.logfile, 'Testing model {0:}: '.format(path))

# #         with torch.no_grad():
# #             for item in self.test_set:
# #                 accuracies = []
# #                 for _ in range(NUM_TEST_TASKS):
# #                     task_dict = self.metadataset.get_test_task(item)
# #                     context_images, target_images, context_labels, target_labels = self.prepare_task(task_dict)
# #                     target_logits = self.model(context_images, context_labels, target_images)
# #                     accuracy = self.accuracy_fn(target_logits, target_labels)
# #                     accuracies.append(accuracy.item())
# #                     del target_logits

# #                 accuracy = np.array(accuracies).mean() * 100.0
# #                 accuracy_confidence = (196.0 * np.array(accuracies).std()) / np.sqrt(len(accuracies))

# #                 print_and_log(self.logfile, '{0:}: {1:3.1f}+/-{2:2.1f}'.format(item, accuracy, accuracy_confidence))


# %%
