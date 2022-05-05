from scripts.dataset import get_dataset
from scripts.newModel import get_model, test
from scripts.utils import convert_to_color_
import torch
import numpy as np

def test_model(dataset_path: str,
               LABEL_VALUES: list,
               hyperparams: dict,
               weights_path: str,
               img_name: str,
               gt_name: str=None,
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    img, gt, IGNORED_LABELS, palette = get_dataset(dataset_path, img_name, gt_name, LABEL_VALUES)
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

    color_prediction = convert_to_color_(prediction, palette)

    return gt, prediction, color_prediction

