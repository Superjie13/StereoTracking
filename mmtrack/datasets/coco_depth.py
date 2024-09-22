# This script is modified from CocoDataset of mmdet,
# for adapting multi-modal data collection (rgb and depth).
import copy
import os.path as osp
from typing import List, Union, Any

from mmtrack.registry import DATASETS
from mmdet.datasets.api_wrappers import COCO
from mmdet.datasets.coco import CocoDataset


@DATASETS.register_module()
class CocoMMDataset(CocoDataset):
    """Multi-Modal dataset for COCO."""

    def __init__(self,
                 disparity_dir_name: str = 'disparity',
                 *args,
                 **kwargs
                 ):
        self.disparity_dir_name = disparity_dir_name
        super().__init__(*args, **kwargs)

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format.
        Add 'disp_path' key to the data_info dict.

        Args:
            raw_data_info (dict): Raw data information load from ``ann_file``

        Returns:
            Union[dict, List[dict]]: Parsed annotation.
        """
        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info['raw_ann_info']
        data_info = {}

        if self.data_prefix.get('img_path', None) is not None:
            img_path = osp.join(self.data_prefix['img_path'],
                                img_info['file_name'])
        else:
            img_path = img_info['file_name']
        data_info['img_path'] = img_path

        disp_file_name = img_info['file_name'].replace('img2', self.disparity_dir_name)
        data_info['disp_path'] = img_path.replace(img_info['file_name'], disp_file_name)

        if self.data_prefix.get('seg_path', None):
            seg_map_path = osp.join(
                self.data_prefix['seg_path'],
                img_info['file_name'].rsplit('.', 1)[0] + self.seg_map_suffix)
        else:
            seg_map_path = None

        data_info['img_id'] = img_info['img_id']
        data_info['seg_map_path'] = seg_map_path
        data_info['height'] = img_info['height']
        data_info['width'] = img_info['width']

        instances = []
        for i, ann in enumerate(ann_info):
            instance = {}

            if ann.get('ignore', False):
                continue

            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]

            if ann.get('iscrowd', False):
                instance['ignore_flag'] = 1
            else:
                instance['ignore_flag'] = 0
            instance['bbox'] = bbox
            instance['bbox_label'] = self.cat2label[ann['category_id']]

            if ann.get('segmentation', None):
                instance['mask'] = ann['segmentation']

            instances.append(instance)
        data_info['instances'] = instances
        return data_info

    def prepare_data(self, idx: int) -> Any:
        """Pass the dataset to the pipeline during training to support mixed
        data augmentation, such as Mosaic and MixUp."""
        if self.test_mode is False:
            data_info = self.get_data_info(idx)
            data_info['dataset'] = self
            return self.pipeline(data_info)
        else:
            return super().prepare_data(idx)
