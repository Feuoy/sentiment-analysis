import torch.nn.functional as F


def binary_crossentropy_loss(output, target):
    # Function that measures Binary Cross Entropy between target and output logits.
    return F.binary_cross_entropy_with_logits(output, target)
