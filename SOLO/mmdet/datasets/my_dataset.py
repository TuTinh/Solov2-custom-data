from .coco import CocoDataset
from .registry import DATASETS
@DATASETS.register_module
class MyDataset(CocoDataset):
    CLASSES = ['_background_', 'road_crack']