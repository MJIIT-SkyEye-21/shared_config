import os
from .common_utils import _get_timestamp_string
from .common_config import OUTPUT_DIRECTORY


def make_classification_model_directory(model_name):
    ts = _get_timestamp_string()
    name = f"{model_name}_cls_{ts}_tile_448_448"
    model_dir = os.path.join(OUTPUT_DIRECTORY, name)

    os.makedirs(model_dir)
    return model_dir
