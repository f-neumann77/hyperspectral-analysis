import scipy.io as io
import numpy as np
import seaborn as sns

def get_dataset(dataset_path: str, img_name: str, gt_name: str) -> tuple[np.array, np.array, list, list, dict]:
    """
    return data from .mat files in tuple

    Parameters
    ----------
    dataset_path: str
        path to directory of dataset
    img_name: str
        name of hyperspectral image
    gt_name: str
        name of mask image
    label_values: list
        names of classes in hyperspectral image

    Returns
    ----------
    img : np.array
        hyperspectral image
    gt : np.array
        mask of hyperspectral image
    pallete : dict
        pallete for colorizing  predicted image
    """

    img = io.loadmat(f'{dataset_path}/{img_name}')['image']
    label_values = io.loadmat(f'{dataset_path}/{img_name}')['labels']

    if gt_name:
        gt = io.loadmat(f'{dataset_path}/{gt_name}')["mask"]
    else:
        gt = None

    ignored_labels = [0]
    img = img.astype("float32")
    img = (img - np.min(img)) / (np.max(img) - np.min(img))

    palette = {0: (0, 0, 0)}
    for k, color in enumerate(sns.color_palette("hls", len(label_values) - 1)):
        palette[k + 1] = tuple(np.asarray(255 * np.array(color), dtype="uint8"))

    return img, gt, ignored_labels, label_values, palette
