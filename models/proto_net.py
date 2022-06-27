# Implementation of Prototypical Network
import torch
from torch import nn
from config import EPSILON

# From https://github.com/google-research/meta-dataset/blob/13fd952585ca0928ba313fe56874081fa7fff7f0/meta_dataset/learner.py#L50
def compute_prototypes(embeddings, labels):
        """Computes class prototypes over the last dimension of embeddings.
        Args:
            embeddings: Tensor of examples of shape [num_examples, embedding_size].
            labels: Tensor of one-hot encoded labels of shape [num_examples,
            num_classes].
        Returns:
            prototypes: Tensor of class prototypes of shape [num_classes,
            embedding_size].
        """
        # [num examples, 1, embedding size].
        embeddings = embeddings.unsqueeze(1)

        # [num examples, num classes, 1].
        labels = labels.unsqueeze(2)

        # Sums each class' embeddings. [num classes, embedding size].
        class_sums = torch.sum(labels*embeddings, 0)

        # The prototype of each class is the averaged embedding of its examples.
        class_num_images = labels.sum(0) # [way].
        prototypes = class_sums / class_num_images

        return prototypes

# From https://github.com/oscarknagg/few-shot/blob/672de83a853cc2d5e9fe304dc100b4a735c10c15/few_shot/utils.py#L45
def pairwise_distances(x: torch.Tensor,
                       y: torch.Tensor,
                       matching_fn: str) -> torch.Tensor:
    """Efficiently calculate pairwise distances (or other similarity scores) between
    two sets of samples.
    # Arguments
        x: Query samples. A tensor of shape (n_x, d) where d is the embedding dimension
        y: Class prototypes. A tensor of shape (n_y, d) where d is the embedding dimension
        matching_fn: Distance metric/similarity score to compute between samples
    """
    n_x = x.shape[0]
    n_y = y.shape[0]

    if matching_fn == 'l2':
        expanded_x = x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = y.unsqueeze(0).expand(n_x, n_y, -1)
        distances = ( expanded_x - expanded_y ).pow(2).sum(dim=2)
        return distances
    elif matching_fn == 'cosine':
        normalised_x = x / (x.pow(2).sum(dim=1, keepdim=True).sqrt() + EPSILON)
        normalised_y = y / (y.pow(2).sum(dim=1, keepdim=True).sqrt() + EPSILON)

        expanded_x = normalised_x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = normalised_y.unsqueeze(0).expand(n_x, n_y, -1)

        cosine_similarities = (expanded_x * expanded_y).sum(dim=2)
        return 1 - cosine_similarities
    elif matching_fn == 'dot':
        expanded_x = x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = y.unsqueeze(0).expand(n_x, n_y, -1)

        return -(expanded_x * expanded_y).sum(dim=2)
    else:
        raise(ValueError('Unsupported similarity function'))

# Based on https://github.com/google-research/meta-dataset/blob/13fd952585ca0928ba313fe56874081fa7fff7f0/meta_dataset/learner.py#L1194
class PrototypicalNetwork(nn.Module):
    def __init__(self, encoder, matching_fn) -> None:
        super(PrototypicalNetwork, self).__init__()

        self.encoder = encoder
        self.matching_fn = matching_fn

    def forward(self, support_images, support_labels, query_images):
        self.support_images = support_images
        self.support_labels = support_labels
        self.query_images = query_images

        # Embedding the images
        support_embeddings = self.encoder(self.support_images)
        query_embeddings = self.encoder(self.query_images)

        # Computing Prototypes 
        support_labels_one_hot = nn.functional.one_hot(self.support_labels, -1)
        prototypes = compute_prototypes(support_embeddings, support_labels_one_hot)

        # Compute logits (in Prototypical Networls logits are the pairwise distances between the prototypes and the query embeddings)
        logits = self.compute_logits(prototypes, query_embeddings)

        return logits

    def compute_logits(self, prototypes, query_embeddings):
        distances = pairwise_distances(query_embeddings, prototypes, self.matching_fn)

        logits = -distances
        return logits
    
    def accuracy(self, logits, query_labels):
        query_labels_pred = logits.argmax(1)
        correct = torch.eq(query_labels_pred, query_labels)
        
        return correct.float().mean()