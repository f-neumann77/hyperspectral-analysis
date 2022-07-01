from scripts.DataLoader import DataLoader
from scripts.dataset import get_dataset
from scripts.Models.model import get_model
from scripts.model_tvts import train
from scripts.utils import sample_gt
import numpy as np
import torch
import torch.utils.data as data


def create_loader(img: np.array,
                  gt: np.array,
                  hyperparams: dict,
                  shuffle: bool = False):
    dataset = DataLoader(img, gt, **hyperparams)
    return data.DataLoader(dataset, batch_size=hyperparams["batch_size"], shuffle=shuffle)


def train_model(dataset_path: str,
                img_name: str,
                gt_name: str,
                hyperparams: dict,
                sample_percentage: float = 0.5,
                weights_path=None):

    img, gt, IGNORED_LABELS, LABEL_VALUES, palette = get_dataset(dataset_path, img_name, gt_name)

    hyperparams['patch_size'] = 7
    hyperparams['batch_size'] = 40
    hyperparams['learning_rate'] = 0.01
    hyperparams['n_bands'] = img.shape[-1]
    hyperparams['ignored_labels'] = IGNORED_LABELS
    hyperparams['net_name'] = 'he'

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
        val_loader=val_loader,
    )
