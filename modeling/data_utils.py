### various utilities for processing data
### with the intent of using it for modeling purposes
### downstream of the data_utils in analysis
import numpy as np

from skimage.transform import downscale_local_mean
from skimage.util import pad
from sklearn.decomposition import PCA
from analysis.data_utils import *

IMGNET_MEAN = np.array([[0.485, 0.456, 0.406]])[..., np.newaxis, np.newaxis]
IMGNET_STD = np.array([[0.229, 0.224, 0.225]])[..., np.newaxis, np.newaxis]

def get_images(dataset, downsample=1, torch_format=True,
        normalize=True):
    """
    Returns the images corresponding to the dataset
    (tang, googim, or both), in a fixed order. Downsamples
    them by the specified factor (default none), puts
    them in PyTorch NCHW format (default true), and normalizes
    them to zero mean and unit variance (default true).

    Returns a numpy array.
    """
    assert dataset in {'tang', 'googim', 'both'}
    if dataset == 'googim': dataset = 'google-imagenet'

    if dataset != 'both':
        images = np.load(DATA_DIR + f'{dataset}/images/all_imgs.npy')
    else:
        tang_images = np.load(DATA_DIR + f'tang/images/all_imgs.npy')
        googim_images = np.load(DATA_DIR + f'google-imagenet/images/all_imgs.npy')

        # googim images are 251x251 -- annoying!
        # but padding one pixel won't throw off RFs much
        googim_images = pad(googim_images, ((0, 0), (0, 1), (0, 1)),
                mode='edge')
        images = np.concatenate([tang_images, googim_images])

    # add channels
    images = images[..., np.newaxis]

    if downsample > 1:
        images = downscale_local_mean(images, (1, downsample, downsample, 1))

    if torch_format:
        images = images.transpose(0, 3, 1, 2)

    if normalize:
        img_mean = np.mean(images)
        img_std = np.std(images)
        images = (images - img_mean) / img_std

    return images

def train_val_test_split(total_size, train_size, val_size,
        deterministic=True):
    """
    Return indices for a training set, validation set, and test
    set of total_size. Indices are randomly selected, but this
    random selection will be done with a fixed seed if deterministic (default).

    Note that if deterministic is true, the numpy RNG will be randomly
    re-seeded at the end, so if it was seeded outside of this function,
    it must be re-seeded after calling.
    """
    if deterministic:
        np.random.seed(0)

    # kinda messy, but it works
    train_val_indices = np.random.choice(total_size, size=(train_size+val_size), replace=False)

    train_only_indices = np.random.choice((train_size + val_size), train_size, replace=False)
    train_indices = train_val_indices[train_only_indices]

    val_mask = np.ones((train_size + val_size,), dtype=bool)
    val_mask[train_only_indices] = 0
    val_indices = train_val_indices[val_mask]

    test_mask = np.ones((total_size,), dtype=bool)
    test_mask[train_val_indices] = 0
    test_indices = np.array(range(total_size))[test_mask]

    # set seed back to random
    if deterministic:
        np.random.seed()

    return train_indices, val_indices, test_indices

def pca_noise_adjust(trial_data, components_used=2):
    """
    Adjusts the data by subtracting out its projection onto
    the first few PCs of the trial-to-trial variability.

    Won't work on data with a temporal dimension.
    """
    means = [np.mean(condition, 0) for condition in trial_data]
    noise_data = [condition - mean for condition, mean in
            zip(trial_data, means)]
    noise_data = np.concatenate(noise_data)

    pca = PCA(n_components=components_used).fit(noise_data)

    projections = [pca.inverse_transform(pca.transform(condition))
            for condition in trial_data]
    adjusted = [condition - projection for condition, projection
            in zip(trial_data, projections)]

    return adjusted

def torchvision_normalize(images):
    """
    Normalize according to Torchvision standards.
    Assuming 0-255 pixel values, and numpy arrays.
    """
    images = images / 255.
    images = (images - IMGNET_MEAN) / IMGNET_STD

    return images
