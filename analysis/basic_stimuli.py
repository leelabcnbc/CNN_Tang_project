### stimulus generation functions for 'physiological' experiments
### TODO: this is really messy. split up bar generation at least
### include rotating
import numpy as np

from skimage.draw import rectangle, disk # needed to draw bars
from skimage.transform import rotate, warp, AffineTransform
from skimage.filters import gaussian
from skimage.util import crop
from math import pi

# TODO: make all this more clear
# these are for the surround suprression experiment
# basically generate a grating, then put it behind an aperture
# scaling of pixels is kind of opaque though
def sine_wave_grating(height, width, freq, contrast, phase):
    """
    generate a sine-wave grating image
    """
    idxs = np.arange(0, 2 * pi, (2 * pi) / width)
    # go from 0-contrast, hopefully
    row = (np.sin((phase * 2 * pi) + idxs * freq) * (contrast / 2)) + (contrast / 2)
    img = np.stack([row] * height, axis=0)
    return img

def partial_grating(grating, center, radius, gray=0.5,
        background=None):
    """
    put an aperture over a sine-wave grating image
    """
    # outside aperture will be gray
    # unless it's something like a diff orientation
    if background is None:
        new_img = np.full(grating.shape, fill_value=gray)
    else:
        new_img = background

    # copy the corresponding circle of grating onto it
    rr, cc = disk((center[0], center[1]), radius, shape=grating.shape)
    new_img[rr, cc] = grating[rr, cc]

    return new_img

def smooth_perimeter(grating, center, radius, sigma):
    """
    smooth the perimeter of a sine-wave grating
    """
    # little tricky because default smooothing doesn't
    # accept certain indices as an argument
    new_img = grating.copy()

    # this lets us know where to apply the smoothing
    # where it's more than 5%
    perimeter = np.zeros(grating.shape)
    rr, cc = disk((center[0], center[1]), radius, shape=perimeter.shape)
    perimeter[rr, cc] = 1
    perimeter = gaussian(perimeter, sigma)
    idxs = perimeter > 0.05

    # now apply the smoothing
    # at the spots where perimeter is  significant
    smoothed = gaussian(new_img, sigma)
    new_img[idxs] = smoothed[idxs]

    return new_img

# this puts all the above together
# for a bunch of stimuli for surround suppression
def generate_stimuli(freq, orientation, center, sigma, H, radii, phases, gray=0.5):
    """
    generate a range of sine-wave grating stimuli of
    increasing radii
    """
    all_stimuli = []

    transformer = AffineTransform(translation=(center[0] - H // 2, center[1] - H // 2))
    for radius in radii:
        stimuli = []
        for phase in phases:
            # 1 contrast and 0 phase
            grating = sine_wave_grating(2 * H, 2 * H, freq, 1.0, phase)
            # radii are no longer ratios
            grating = partial_grating(grating, (H, H),  radius, gray=gray)
            grating = smooth_perimeter(grating, (H, H),  radius, sigma)
            grating = rotate(grating, orientation, mode='constant', cval=0)
            grating = crop(grating, H // 2)
            grating = warp(grating, transformer, cval=0)
            #stimuli.append(grating)
            stimuli.append(grating[np.newaxis, ...])
            #stimuli.append(np.stack([grating, grating, grating]))
        all_stimuli.append(np.array(stimuli))
    return np.array(all_stimuli)

def generate_center_black_grating(freq,orientation,center,sigma,H,radius,phase, center_radii, gray=0.5):
    all_stimuli = []

    transformer = AffineTransform(translation=(center[0] - H // 2, center[1] - H // 2))
    for r in center_radii:
        # 1 contrast and 0 phase
        grating = sine_wave_grating(2 * H, 2 * H, freq, 1.0, phase)
        # radii are no longer ratios
        grating = partial_grating(grating, (H, H), radius, gray=gray)
        grating = smooth_perimeter(grating, (H, H), radius, sigma)
        grating = rotate(grating, orientation, mode='constant', cval=0)
        grating = crop(grating, H // 2)
        grating = warp(grating, transformer, cval=0)
        rr, cc = disk((center[0], center[1]), r, shape=grating.shape)
        grating[rr,cc] = 0.5
        # stimuli.append(grating)
        all_stimuli.append(grating[np.newaxis, ...])
        # stimuli.append(np.stack([grating, grating, grating]))
    return np.array(all_stimuli)

GRID_SIZE = 10
GRID_COUNT = 10
IMG_SIZE = 10
LENGTH = 10
WIDTH = 10
# TODO: simplify
# this generates the 3-bar stimuli for association field experiments
# but should be split up into simpler functions
def generate_grid_stimulus(x_loc, y_loc, orientation, rf_center,
        grid_size=GRID_SIZE, grid_count=GRID_COUNT, img_size=IMG_SIZE,
        bar_length=LENGTH, bar_width=WIDTH, center_stim=True,
        surround_stim=True):
    """
    generate a single stimulus according to the 9x9 grid used by K&G

    x_loc and y_loc specify the surround location in the 
    top-left half of the grid, the mirror-symmetric bar is always added

    default values set in the file
    """
    # grid size -- K&G use 9x9 but that is inconveniently large
    # should generally be an odd number of grid squares though, to center
    # needs to fit in the total image size (determined in rf_loc_mapping)
    total_grid = grid_count * grid_size
    # this breaks for Gaya cause of small images -- could be a problem...
    #assert total_grid <= img_size

    # rf_center is in pixels, determined from rf_loc_map previously
    rf_x, rf_y = rf_center # tuple

    # create the image
    image = np.full((img_size, img_size, 1), 1, dtype=np.float32)

    # create the center stimulus at the image center
    # rotation and translation at the end, after surround added
    if center_stim:
        bar_rr, bar_cc = rectangle(start=(img_size // 2 - bar_width // 2, 
            img_size // 2 - bar_length // 2),
                extent=(bar_width, bar_length), shape=(img_size, img_size))
        image[bar_rr, bar_cc, :] = 0

    # create the surround stimuli according to the grid location
    if surround_stim:
        x_diff = grid_size * ((grid_count // 2) - x_loc)
        y_diff = grid_size * ((grid_count // 2) - y_loc)

        # top left side first
        x_pos = (img_size // 2) - x_diff
        y_pos = (img_size // 2) - y_diff

        bar_rr, bar_cc = rectangle(start=(x_pos - bar_width // 2, 
            y_pos - bar_length // 2), 
                extent=(bar_width, bar_length), shape=(img_size, img_size))
        image[bar_rr, bar_cc, :] = 0

        # now the mirror image
        x_pos = (img_size // 2) + x_diff
        y_pos = (img_size // 2) + y_diff

        bar_rr, bar_cc = rectangle(start=(x_pos - bar_width // 2, 
            y_pos - bar_length // 2), 
                extent=(bar_width, bar_length), shape=(img_size, img_size))
        image[bar_rr, bar_cc, :] = 0

    # now need to rotate to proper orientation
    # the whole grid rotates, as it should (?)
    image = rotate(image, orientation * (180 / rotations),
            mode='constant', cval=1)

    # and translate according to the RF center
    x_offset = rf_x - (img_size // 2)
    y_offset = rf_y - (img_size // 2)
    transformer = AffineTransform(translation = (x_offset, y_offset))
    image = warp(image, transformer, cval=1)

    return image

def apply_contrast(stimuli, contrast):
    return