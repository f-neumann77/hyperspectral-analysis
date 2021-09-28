import torch
from matplotlib import pyplot as plt
from utils import (
    metrics,
    convert_to_color_,
    convert_from_color_,
    display_dataset,
    display_predictions,
    explore_spectrums,
    plot_spectrums,
    sample_gt,
    build_dataset,
    show_results,
    compute_imf_weights,
    get_device,
)
from datasets import get_dataset, HyperX, open_file, DATASETS_CONFIG
from models import get_model, train, test, save_model
import seaborn as sns
import numpy as np
import cv2
palette = None



def convert_to_color(x):
    return convert_to_color_(x, palette=palette)


img, gt, LABEL_VALUES, IGNORED_LABELS, RGB_BANDS, palette = get_dataset('corn_3', './Datasets/')

N_CLASSES = len(LABEL_VALUES)
N_BANDS = img.shape[-1]

if palette is None:
    # Generate color palette
    palette = {0: (0, 0, 0)}
    for k, color in enumerate(sns.color_palette("hls", len(LABEL_VALUES) - 1)):
        palette[k + 1] = tuple(np.asarray(255 * np.array(color), dtype="uint8"))
invert_palette = {v: k for k, v in palette.items()}


hyperparams = {

'patch_size': 7,
'center_pixel':True,
'batch_size': 40,
'device': torch.device("cuda"),
'n_classes': N_CLASSES,
'n_bands': N_BANDS,
'ignored_labels': IGNORED_LABELS,
'test_stride':1
}


model, optimizer, loss, hyperparams = get_model('he_bn', **hyperparams)

model.load_state_dict(torch.load('/workdir/checkpoints/he_et_al_bn/corn_triple/2021_08_31_20_05_03_epoch5_0.98.pth'))
model.eval()

probabilities = test(model, img, hyperparams)
prediction = np.argmax(probabilities, axis=-1)

mask = np.zeros(gt.shape, dtype="bool")
for l in IGNORED_LABELS:
    mask[gt == l] = True


color_prediction = convert_to_color(prediction)

cv2.imwrite('predict_corn_3_he_bn_new.png', color_prediction)

for i in range(N_CLASSES):
    print(palette[i], LABEL_VALUES[i])
