MODEL_BASE = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"

TARGET_CLASSES = [
    'rru1',
    'rru2',
    'rru3',
    'rod',
    'microwave',
    'workplatform',
    'rfpanel',
    'boom',
    'emptyboom',
    'bandpass',
    'moss',
    'other-defect',
    'dangle',
    'aol',
    'rru',
    'tower',
    'peel'
]

# TILE_SIDE_HEIGHT = 800
# TILE_SIDE_WIDTH = 800
TILE_SIDE_HEIGHT = 1920
TILE_SIDE_WIDTH = 1920
# TILE_SIDE_HEIGHT = 2240
# TILE_SIDE_WIDTH = 2240

# Initialy it's 448
# Move slow vertically and fast horizontally
TILE_STRIDE_VERTICAL = TILE_SIDE_HEIGHT // 4
TILE_STRIDE_HORIZONTAL = TILE_SIDE_WIDTH // 4

INTERSECTION_IOU_THRESH = 10
MODEL_SCORE_THRESH_TEST = 0.55

model_name = MODEL_BASE.split("/")[1].replace(".yaml", "")
ITER_COUNT = 100000
