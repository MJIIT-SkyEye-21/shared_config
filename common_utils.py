import cv2
import random
import numpy as np

from common_config import _CLASS_COLORS
from datetime import datetime, timedelta

RANDOM_SEED = 1024


def fixed_random_seed():
    random.seed(RANDOM_SEED)


def fixed_numpy_random_seed():
    np.random.seed(RANDOM_SEED)


def _get_timestamp_string():
    return (datetime.now() + timedelta(hours=8)).strftime('%Y_%m_%d_%H_%M')


def make_new_model_name(model_name):
    return f"{model_name}_{_get_timestamp_string()}"


def xywh_to_xyxy(i_crop):
    # https://github.com/BloodAxe/pytorch-toolbelt/blob/a8ca678c5d2e86085a609b2fe7f7f34ede6bd207/pytorch_toolbelt/inference/tiles.py#L306
    x, y, tile_width, tile_height = i_crop
    return [x, y, x+tile_width, y+tile_height]


def bbox_color_bgr(class_name):
    r, g, b = _CLASS_COLORS[class_name]
    return (b, g, r)


def bbox_color_rgb(class_name):
    return _CLASS_COLORS[class_name]


def draw_text(img, text,
              font=cv2.FONT_HERSHEY_PLAIN,
              pos=(0, 0),
              font_scale=2,
              font_thickness=2,
              text_color=(0, 255, 0),
              text_color_bg=(0, 0, 0)
              ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font,
                font_scale, text_color, font_thickness, cv2.LINE_AA)

    return text_size
