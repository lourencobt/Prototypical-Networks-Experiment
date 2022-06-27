from datetime import datetime
import torch
import numpy as np
import argparse
import os

from tqdm import tqdm
from paths import META_RECORDS_ROOT
from utils import ValidationAccuracies, get_checkpoint_files, device
from models.proto_net import PrototypicalNetwork
from data.meta_dataset_reader import MetaDatasetEpisodeReader
from torchvision import  models, transforms

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Quiet TensorFlow warnings

NUM_TRAIN_TASKS = 110000
NUM_VALIDATION_TASKS = 200
NUM_TEST_TASKS = 600
VALIDATION_FREQUENCY = 10000

def main():
    learner = Learner()
    learner.run()

class Learner:
    def __init__(self):
        self.args = self.parse_command_line()

        self.checkpoint_dir, self.checkpoint_path_validation, self.checkpoint_path_final \
            = get_checkpoint_files(self.args.checkpoint_dir, self.args.mode)

        print("Options: %s\n" % self.args)
        print("Checkpoint Directory: %s\n" % self.checkpoint_dir)

        self.model, self.encoder = self.init_model()

        #Load Data
        if "train" in self.args.mode:
            self.train_loader = MetaDatasetEpisodeReader(META_RECORDS_ROOT, "train", self.args.shuffle)
            self.val_loader = MetaDatasetEpisodeReader(META_RECORDS_ROOT, "val", self.args.shuffle)

            self.validation_accuracies = ValidationAccuracies(self.val_loader.val_datasets)
            
        elif "test" in self.args.mode:
            print("Helloooo")
            self.test_loader = MetaDatasetEpisodeReader(META_RECORDS_ROOT, "test", self.args.shuffle)

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.learning_rate, momentum=self.args.momentum,          weight_decay=self.args.weight_decay)
        self.optimizer.zero_grad()

    """
    Command line parser
    """
    def parse_command_line(self):
        parser = argparse.ArgumentParser()
        
        parser.add_argument("--mode", choices=["train", "test", "train_test"], default="train_test",
                            help="Whether to run training only, testing only, or both training and testing.")
        parser.add_argument("--learning_rate", "-lr", type=float, default=5e-4, help="Learning rate.")
        parser.add_argument("--momentum", "-mt", type=float, default=0, help="Momentum.")
        parser.add_argument("--weight_decay", "-wd", type=float, default=0, help="Weight Decay")
        parser.add_argument("--matching_fn", "-mf", type=str, default='l2', 
                            help="Distance metric/similarity score to compute between prototypes and query embeddings. Can be 'l2', 'cosine' and 'dot'")
        parser.add_argument("--tasks_per_batch", type=int, default=16,
                            help="Number of tasks between parameter optimizations.")
        parser.add_argument("--checkpoint_dir", "-c", default='./checkpoints', help="Directory to save checkpoint to.")
        parser.add_argument("--test_model_path", "-m", default=None, help="Path to model to load and test.")
        parser.add_argument("--shuffle", action='store_true',
                            help="As per default, shuffles composition and sampling of episodes. Set False to research purposes.")

        args = parser.parse_args()

        return args

    def init_model(self):
        #Model 
        encoder = models.resnet18(pretrained=True)
        model = PrototypicalNetwork(encoder, self.args.matching_fn).to(device)

        return model, encoder

    def run(self):
        if self.args.mode == 'train' or self.args.mode == 'train_test':
            self.train()

        if self.args.mode == 'train_test':
            self.test(self.checkpoint_path_final)
            self.test(self.checkpoint_path_validation)

        if self.args.mode == 'test':
            self.test(self.args.test_model_path)

    def train(self):
        # ! Think about a better way to do this
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

        losses = []
        train_accuracies = []
        total_iterations = NUM_TRAIN_TASKS

        self.model.train()
        for iteration, (task, source_id) in tqdm(get_iterator(self.train_loader, NUM_TRAIN_TASKS, T)):
            support_images=task['support_images'].to(device)
            support_labels=task['support_labels'].to(device)
            query_images=task['query_images'].to(device)
            query_labels=task['query_labels'].to(device)

            # Compute prediction error
            target_logits = self.model(support_images, support_labels, query_images)
            task_loss = self.loss_fn(target_logits, query_labels)
            task_accuracy = self.model.accuracy(target_logits, query_labels)

            train_accuracies.append(task_accuracy)
            losses.append(task_loss)

            # ! Optimize per batch?
            # Backpropagation
            task_loss.backward()
            self.optimizer.zero_grad()
            self.optimizer.step()    

            if (iteration + 1) % 1000 == 0:
                # print training stats
                print('[{}] Task [{}/{}], Train Loss: {:.7f}, Train Accuracy: {:.7f}'
                                .format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),iteration + 1, total_iterations, torch.Tensor(losses).mean().item(),
                                        torch.Tensor(train_accuracies).mean().item()))
                train_accuracies = []
                losses = []
            
            if ((iteration + 1) % VALIDATION_FREQUENCY == 0) and (iteration + 1) != total_iterations:
                accuracy_dict = self.validate()
                self.validation_accuracies.print(accuracy_dict)
                # save the model if validation is the best so far
                if self.validation_accuracies.is_better(accuracy_dict):
                    self.validation_accuracies.replace(accuracy_dict)
                    torch.save(self.model.state_dict(), self.checkpoint_path_validation)
                    print('Best validation model was updated.')
                    print('')
                self.model.train()

        #save the final model
        torch.save(self.model.state_dict(), self.checkpoint_path_final)

    def validate(self):
        # ! Think about a better way to do this
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

        self.model.eval()
        with torch.no_grad():
            accuracy_dict = {}

            for dataset in self.val_loader.val_datasets:

                accuracies = []
                for iteration, (task, source_id) in get_iterator(self.val_loader, dataset, NUM_VALIDATION_TASKS, T):
                    support_images=task['support_images'].to(device)
                    support_labels=task['support_labels'].to(device)
                    query_images=task['query_images'].to(device)
                    query_labels=task['query_labels'].to(device)

                    target_logits = self.model(support_images, support_labels, query_images)
                    task_accuracy = self.model.accuracy(target_logits, query_labels)

                    accuracies.append(task_accuracy.cpu())
                    accuracies = accuracies
                    # ! Is this needed?
                    del target_logits
            
                accuracy = np.array(accuracies).mean() * 100.0
                accuracy_confidence = (196.0 * np.array(accuracies).std()) / np.sqrt(len(accuracies))

                accuracy_dict[dataset] = {"accuracy": accuracy, "confidence": accuracy_confidence}
        
        return accuracy_dict

    def test(self, path):
        # ! Think about a better way to do this
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

        self.model = self.init_model()
        self.model.load_state_dict(torch.load(path))

        print("")  # add a blank line
        print('Testing model {0:}: '.format(path))

        with torch.no_grad():
            for dataset in self.test_loader.test_datasets:
                accuracies = []
                for iteration, (task, source_id) in get_iterator(self.test_loader, dataset, NUM_TEST_TASKS, T):
                    support_images=task['support_images'].to(device)
                    support_labels=task['support_labels'].to(device)
                    query_images=task['query_images'].to(device)
                    query_labels=task['query_labels'].to(device)

                    target_logits = self.model(support_images, support_labels, query_images)
                    task_accuracy = self.model.accuracy(target_logits, query_labels)

                    accuracies.append(task_accuracy.cpu())
                    accuracies = accuracies
                    # ! Is this needed?
                    del target_logits

                accuracy = np.array(accuracies).mean() * 100.0
                accuracy_confidence = (196.0 * np.array(accuracies).std()) / np.sqrt(len(accuracies))

                print('{0:}: {1:3.1f}+/-{2:2.1f}'.format(dataset, accuracy, accuracy_confidence))

if __name__ == "__main__":
    main()
