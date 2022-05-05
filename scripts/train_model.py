from scripts.HyperX import HyperX
from scripts.dataset import get_dataset
from scripts.newModel import get_model, train
from scripts.utils import sample_gt
import numpy as np
import torch
import torch.utils.data as data

#from typing import List, Dict

def create_loader(img: np.array,
                  gt: np.array,
                  hyperparams: dict,
                  shuffle: bool = False):
    dataset = HyperX(img, gt, **hyperparams)
    return data.DataLoader(dataset, batch_size=hyperparams["batch_size"], shuffle=shuffle)


def train_model(dataset_path: str,
                img_name: str,
                gt_name: str,
                LABEL_VALUES: list,
                hyperparams: dict,
                sample_percentage: float = 0.5,
                weights_path=None):

    img, gt, IGNORED_LABELS, palette = get_dataset(dataset_path, img_name, gt_name, LABEL_VALUES)
    hyperparams['patch_size'] = 7
    hyperparams['batch_size'] = 40
    hyperparams['learning_rate'] = 0.01
    hyperparams['n_bands'] = img.shape[-1]
    hyperparams['ignored_labels'] = IGNORED_LABELS

    model, optimizer, loss, hyperparams = get_model(hyperparams)

    if weights_path:
        model.load_state_dict(torch.load(weights_path))

    train_gt, _ = sample_gt(gt, sample_percentage, mode='random')

    train_gt, val_gt = sample_gt(train_gt, 0.95, mode="random")

    # Generate the dataset
    train_loader = create_loader(img, train_gt, hyperparams, shuffle=True)

    val_loader = create_loader(img, val_gt, hyperparams)

    train(
        model,
        optimizer,
        loss,
        train_loader,
        hyperparams["epoch"],
        scheduler=hyperparams["scheduler"],
        device=hyperparams["device"],
        supervision=hyperparams["supervision"],
        val_loader=val_loader,
    )
