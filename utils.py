import math
import os
from datetime import datetime
import torch
from functools import partial

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ValidationAccuracies:
    """
    Determines if an evaluation on the validation set is better than the best so far.
    In particular, this handles the case for meta-dataset where we validate on multiple datasets and we deem
    the evaluation to be better if more than half of the validation accuracies on the individual validation datsets
    are better than the previous best.
    """

    def __init__(self, validation_datasets):
        self.datasets = validation_datasets
        self.dataset_count = len(self.datasets)
        self.current_best_accuracy_dict = {}
        for dataset in self.datasets:
            self.current_best_accuracy_dict[dataset] = {"accuracy": 0.0, "confidence": 0.0}

    def is_better(self, accuracies_dict):
        is_better = False
        is_better_count = 0
        for i, dataset in enumerate(self.datasets):
            if accuracies_dict[dataset]["accuracy"] > self.current_best_accuracy_dict[dataset]["accuracy"]:
                is_better_count += 1

        if is_better_count >= int(math.ceil(self.dataset_count / 2.0)):
            is_better = True

        return is_better

    def replace(self, accuracies_dict):
        self.current_best_accuracy_dict = accuracies_dict

    def print(self, accuracy_dict):
        print("")  # add a blank line
        print("Validation Accuracies:")
        for dataset in self.datasets:
            print("{0:}: {1:.1f}+/-{2:.1f}".format(dataset, accuracy_dict[dataset]["accuracy"],
                                                                    accuracy_dict[dataset]["confidence"]))
        print("")  # add a blank line


def get_checkpoint_files(checkpoint_dir, experiment_name):
    """
    Function that takes a path to a checkpoint directory and returns a reference to paths to the
    fully trained model and the model with the best validation score.
    """
    unique_checkpoint_dir = os.path.join(checkpoint_dir, experiment_name+"-"+datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    if not os.path.exists(unique_checkpoint_dir):
        os.makedirs(unique_checkpoint_dir)
    checkpoint_path_validation = os.path.join(unique_checkpoint_dir, 'best_validation.pt')
    checkpoint_path_final = os.path.join(unique_checkpoint_dir, 'fully_trained.pt')

    return unique_checkpoint_dir, checkpoint_path_validation, checkpoint_path_final