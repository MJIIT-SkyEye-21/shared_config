import os
from .common_utils import _get_timestamp_string
from .common_config import OUTPUT_DIRECTORY


def make_classification_model_directory(model_name):
    ts = _get_timestamp_string()
    name = f"{model_name}_cls_{ts}_tile_448_448"
    model_dir = os.path.join(OUTPUT_DIRECTORY, name)

    os.makedirs(model_dir)
    return model_dir


def print_metrics(confusion_m, classification_report, labels):
    from tabulate import tabulate
    adapted_matrix = []

    for i, cls in enumerate(labels):
        adapted_matrix.append([cls, *confusion_m[i]])

    tabulated_matrix = tabulate(adapted_matrix, headers=['Class', *labels])

    print('Confusion matrix: \n', tabulated_matrix)
    print('Classification report : \n', classification_report)


def compute_metrics(actual, predicted, labels):
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report
    import numpy as np

    labels_np = np.array(labels)
    p = labels_np[predicted]
    a = labels_np[actual]

    confusion_m = confusion_matrix(a, p, labels=labels)
    report = classification_report(a, p, labels=labels)

    return confusion_m, report
