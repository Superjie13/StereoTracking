# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import List, Union

from mmengine.dataset import force_full_init
from mmtrack.registry import DATASETS
from .base_video_dataset import BaseVideoDataset


@DATASETS.register_module()
class MOTDispDataset(BaseVideoDataset):
    """Dataset for MOTKitti.

    Args:
        occluded_thr (int, optional): Integer (0,1,2,3) indicating occlusion state:
                     0 = fully visible, 1 = partly occluded
                     2 = largely occluded, 3 = unknown. Default: 2
        detection_file (str, optional): The path of the public
            detection file. Default to None.
        disparity_dir_name (str, optional): dir name to access the disparity map
    """

    METAINFO = {
        'CLASSES':
        ('drone',)
    }

    def __init__(self,
                 detection_file: str = None,
                 disparity_dir_name: str='disparity',
                 *args,
                 **kwargs):
        self.detection_file = detection_file
        self.disparity_dir_name = disparity_dir_name
        super().__init__(*args, **kwargs)

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format.

        Args:
            raw_data_info (dict): Raw data information load from ``ann_file``

        Returns:
            Union[dict, List[dict]]: Parsed annotation.
        """
        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info['raw_ann_info']
        data_info = {}

        data_info.update(img_info)
        if self.data_prefix.get('img_path', None) is not None:
            img_path = osp.join(self.data_prefix['img_path'],
                                img_info['file_name'])
        else:
            img_path = img_info['file_name']
        data_info['img_path'] = img_path

        disp_file_name = img_info['file_name'].replace('left', self.disparity_dir_name)
        data_info['disp_path'] = img_path.replace(img_info['file_name'], disp_file_name)

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
            instance['instance_id'] = ann['instance_id']
            instance['category_id'] = ann['category_id']
            instance['bbox_label'] = self.cat2label[ann['category_id']]
            instance['bbox'] = bbox  # bbox (left, top, right, bottom)
            instance['location'] = ann['location']  # location (x, y, z)
            instance['mot_conf'] = ann['mot_conf']
            instance['visibility'] = ann['visibility']
            if len(instance) > 0:
                instances.append(instance)
        data_info['instances'] = instances
        return data_info

    @force_full_init
    def get_data_info(self, idx: int) -> dict:
        data_info = super().get_data_info(idx)
        data_info.update({'cat2label': self.cat2label})

        return data_info

