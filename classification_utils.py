from typing import List, Tuple
import re
import os
import io
from collections import Counter
from .common_utils import _get_timestamp_string
from .common_config import OUTPUT_DIRECTORY


def make_classification_model_directory(model_name):
    ts = _get_timestamp_string()
    name = f"{model_name}_cls_{ts}_tile_448_448"
    model_dir = os.path.join(OUTPUT_DIRECTORY, name)

    os.makedirs(model_dir)
    return model_dir


def load_model(model_path, device="cuda"):
    import torch

    if torch.cuda.is_available():
        def map_location(storage, loc): return storage.cuda()
    else:
        map_location = "cpu"

    print("[INFO] loading the model...")
    model = torch.load(model_path, map_location=map_location)
    model.to(device)
    model.eval()
    return model


def get_computed_metrics(actual, predicted, labels) -> str:
    """
        Compute metrics for classification (confusion matrix, classification report)
        :param actual: list of actual labels
        :param predicted: list of predicted labels
        :param labels: list of labels
        :return: str object with metrics
    """

    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report
    from tabulate import tabulate
    import numpy as np

    labels_np = np.array(labels)
    p = labels_np[predicted]
    a = labels_np[actual]

    confusion_m = confusion_matrix(a, p, labels=labels)

    adapted_matrix = []

    for i, cls in enumerate(labels):
        adapted_matrix.append([cls, *confusion_m[i]])

    tabulated_matrix = tabulate(adapted_matrix, headers=['Class', *labels])
    out = io.StringIO()
    print('Confusion matrix: \n', tabulated_matrix, file=out)
    report = classification_report(a, p, labels=labels)
    print('Classification report:\n', report, file=out)

    return out.getvalue()


def get_dataset_metrics(root_dir) -> Tuple[dict, list]:
    from torchvision import datasets

    ds = datasets.ImageFolder(root=root_dir)
    rm_idx_num = re.compile(r'_\d+.*')
    rm_tile = re.compile(r'^tile_')

    site_counter_dict = {}
    for (fname, label_idx) in ds.samples:
        _, image_name = os.path.split(fname)

        image_name = rm_idx_num.sub('', image_name, 1)
        site_name = rm_tile.sub('', image_name, 1)

        if site_name not in site_counter_dict:
            site_counter_dict[site_name] = Counter()

        site_counter_dict[site_name].update([label_idx])

    return site_counter_dict, ds.classes


def get_printable_dataset_metrics(root_dir) -> str:
    site_counter_dict, classes = get_dataset_metrics(root_dir)
    out_fd = io.StringIO()
    class_counter = Counter()
    for (site, site_counter) in site_counter_dict.items():
        print(site, file=out_fd)
        for label_idx in site_counter:
            class_counter.update([label_idx])
            print('|-', '{0:04}'.format(site_counter[label_idx]),
                  classes[label_idx], file=out_fd)
        print(file=out_fd)

    print('Classes:', file=out_fd)
    for label_idx in class_counter:
        print('|-', '{0:04}'.format(class_counter[label_idx]),
              classes[label_idx], file=out_fd)

    return out_fd.getvalue()
