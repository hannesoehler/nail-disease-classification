import numpy as np


def dice_coefficient(mask1, mask2):

    """Calculate dice coefficient for two segmentation masks

    Args:
        mask1: numpy array, segmentation mask
        mask2: numpy array, segmentation mask

    Returns:
        dice: float, dice coefficient
    """

    mask1[mask1 > 0] = 1  # bc mask was 255
    mask2[mask2 > 0] = 1  # bc mask was 255
    intersect = np.sum(mask1 * mask2)
    fsum = np.sum(mask1)
    ssum = np.sum(mask2)
    dice = (2 * intersect) / (fsum + ssum)
    dice = np.mean(dice)
    dice = round(dice, 3)  # for easy reading
    return dice
