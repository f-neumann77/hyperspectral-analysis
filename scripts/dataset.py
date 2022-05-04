import scipy.io as io
import numpy as np
import seaborn as sns
from typing import List


def get_dataset(dataset_path: str, img_name: str, gt_name: str, label_values: List) -> tuple:
    """ return data from .mat files in tuple

    :param dataset_path: path to directory of dataset
    :param img_name: name of hyperspectral image
    :param gt_name: name of mask image
    :param label_values: list of name of classes in hyperspectral image
    :return:
    """
    img = io.loadmat(f'{dataset_path}/{img_name}')['image']
    gt = io.loadmat(f'{dataset_path}/{gt_name}')["img"]
    ignored_labels = [0]
    img = np.asarray(img, dtype="float32")
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    palette = {0: (0, 0, 0)}
    for k, color in enumerate(sns.color_palette("hls", len(label_values) - 1)):
        palette[k + 1] = tuple(np.asarray(255 * np.array(color), dtype="uint8"))
    return img, gt, label_values, ignored_labels, palette
