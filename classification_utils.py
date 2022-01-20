import os
import io
from numpy import number
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


def get_computed_metrics(actual, predicted, labels) -> io.StringIO:
    """
        Compute metrics for classification (confusion matrix, classification report)
        :param actual: list of actual labels
        :param predicted: list of predicted labels
        :param labels: list of labels
        :return: io.StringIO object with metrics
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

    return out
