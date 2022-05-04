from scripts.dataset import get_dataset
from scripts.newModel import get_model, test
from scripts.utils import convert_to_color_
import torch
import numpy as np
from typing import List, Dict, Tuple

def test_model(dataset_path: str,
               img_name: str,
               gt_name: str,
               label_values: List,
               hyperparams: Dict,
               weights_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    img, gt, LABEL_VALUES, IGNORED_LABELS, palette = get_dataset(dataset_path, img_name, gt_name, label_values)
    hyperparams['patch_size'] = 7
    hyperparams['batch_size'] = 40
    hyperparams['n_classes'] = len(LABEL_VALUES)
    hyperparams['n_bands'] = img.shape[-1]
    hyperparams['ignored_labels'] = IGNORED_LABELS
    hyperparams['learning_rate'] = 0.01
    hyperparams['test_stride'] = 1

    model, optimizer, loss, hyperparams = get_model(hyperparams)
    model.load_state_dict(torch.load(weights_path))
    model.eval()

    probabilities = test(model, img, hyperparams)
    prediction = np.argmax(probabilities, axis=-1)

    mask = np.zeros(gt.shape, dtype="bool")
    for l in IGNORED_LABELS:
        mask[gt == l] = True

    color_prediction = convert_to_color_(prediction, palette)

    return gt, prediction, color_prediction

