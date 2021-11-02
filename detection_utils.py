import os
from common_utils import _get_timestamp_string
from common_config import OUTPUT_DIRECTORY
from detection_config import TILE_SIDE_HEIGHT, TILE_SIDE_WIDTH, TILE_STRIDE_VERTICAL, TILE_STRIDE_HORIZONTAL


def make_new_model_directory(model_name):
    ts = _get_timestamp_string()
    name = f"{model_name}_{ts}"\
        f"_tile_{TILE_SIDE_HEIGHT}_{TILE_SIDE_WIDTH}"\
        f"_stride_{TILE_STRIDE_VERTICAL}_{TILE_STRIDE_HORIZONTAL}"

    model_dir = os.path.join(OUTPUT_DIRECTORY, name)

    os.makedirs(model_dir)
    return model_dir
