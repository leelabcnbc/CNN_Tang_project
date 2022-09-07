"""
Assorted loss functions for use in optimization.
All take in at least three arguments: the real output, 
the target output, and the image (for regularization), 
but don't necessary use all of these.

Partial application can be used to change the default
arguments beyond those three.

Most default arguments,  come from the paper
"Understanding Deep Image Representations by Inverting Them".
"""
import torch

def alpha_norm(output, target, image, alpha=6):
    """
    Takes the alpha-norm of the mean-subtracted
    image, but with the mean, not the sum, to better
    account for variable-size images without changing
    hyperparameters.
    """
    return torch.norm(image - image.mean(), alpha)


def tv_norm(output, target, image, beta=2):
    """
    Takes the total variation norm of the image.
    """
    col_shift = torch.empty(image.shape, requires_grad=False).cuda()
    row_shift = torch.empty(image.shape, requires_grad=False).cuda()

    row_shift[:, :, 1:, :] = (image[:, :, 1:, :] - image[:, :, :-1, :]) ** 2
    col_shift[:, :, :, 1:] = (image[:, :, :, 1:] - image[:, :, :, :-1]) ** 2

    return torch.norm(row_shift + col_shift, beta / 2)


def output_loss(output, target, image):
    """
    Euclidean distance between the output and target.
    """
    return torch.norm(output - target, 2)

def specific_output_loss(output, target, image, idx):
    """
    Euclidean distance between the output and target
    at a specific slice of the tensors.
    """
    relevant_output = output[idx]

    return torch.norm(relevant_output - target, 2)

def maximization_loss(output, target, image, idx):
    """
    Loss intended to maximize a particular output index.
    """
    return -output[idx]

def standard_loss(output, target, image,
        lambda_a=1, lambda_b=1,
        alpha=6, beta=2):
    """
    WARNING: I'm not sure this works yet, and the default
    parameter values are not at all optimized and possible
    unreasonable.

    Combine output loss, alpha-norm loss, and total
    variation loss, as in the representation-inversion
    paper.
    """
    normed_output = output_loss(output, target, image) / torch.norm(target, 2)
    # leaving out scaling factor inside the output loss from
    # paper, for convenience
    # TODO: maybe implement this

    return normed_output + lambda_a * alpha_norm(output, target, image, alpha) \
            + lambda_b * tv_norm(output, target, image, beta)
