# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import List, Union, Any

from mmengine.dataset import force_full_init
from mmtrack.registry import DATASETS
from mmdet.datasets.coco import CocoDataset


@DATASETS.register_module()
class SELMA_Coco_Dataset(CocoDataset):
    """Dataset for SELMA. (only RGB, disparity and depth are available)
    """

    def __init__(self,
                 disparity_dir_name: str='disparity',
                 depth_dir_name: str='DEPTHCAM_FRONT_LEFT',
                 *args,
                 **kwargs):
        self.depth_dir_name = depth_dir_name
        self.disparity_dir_name = disparity_dir_name
        super().__init__(*args, **kwargs)

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        """Parse file path to target format.

        Returns:
            Union[dict, List[dict]]: Parsed path.
        """
        img_info = raw_data_info['raw_img_info']
        # ann_info = raw_data_info['raw_ann_info']
        data_info = {}

        data_info.update(img_info)
        if self.data_prefix.get('img_path', None) is not None:
            img_path = osp.join(self.data_prefix['img_path'],
                                img_info['file_name'])
        else:
            img_path = img_info['file_name']
        data_info['img_path'] = img_path

        disp_file_name = img_info['file_name'].replace('CAM_FRONT_LEFT', self.disparity_dir_name)
        data_info['disp_path'] = img_path.replace(img_info['file_name'], disp_file_name).replace('.jpg', '.png')

        depth_file_name = img_info['file_name'].replace('CAM_FRONT_LEFT', self.depth_dir_name)
        data_info['depth_path'] = img_path.replace(img_info['file_name'], depth_file_name).replace('.jpg', '.png')

        instances = []
        # for i, ann in enumerate(ann_info):
        #     instance = {}
        #
        #     if (not self.test_mode) and (ann['occluded'] <
        #                                  self.occluded_thr):
        #         continue
        #     if ann.get('ignore', False):
        #         continue
        #     x1, y1, w, h = ann['bbox']
        #     inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
        #     inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
        #     if inter_w * inter_h == 0:
        #         continue
        #     if ann['area'] <= 0 or w < 1 or h < 1:
        #         continue
        #     if ann['category_id'] not in self.cat_ids:
        #         continue
        #     bbox = [x1, y1, x1 + w, y1 + h]
        #
        #     if ann.get('iscrowd', False):
        #         instance['ignore_flag'] = 1
        #     else:
        #         instance['ignore_flag'] = 0
        #     instance['instance_id'] = ann['instance_id']
        #     instance['category_id'] = ann['category_id']
        #     instance['bbox_label'] = self.cat2label[ann['category_id']]
        #     instance['truncated'] = ann['truncated']
        #     instance['occluded'] = ann['occluded']
        #     instance['alpha'] = ann['alpha']
        #     instance['bbox'] = bbox  # bbox (left, top, right, bottom)
        #     instance['dim'] = ann['dim']  # dimension (height, width, length)
        #     instance['location'] = ann['location']  # location (x, y, z)
        #     instance['rotation_y'] = ann['rotation_y']
        #     instance['mot_conf'] = ann['mot_conf']
        #     instance['visibility'] = ann['visibility']
        #     if len(instance) > 0:
        #         instances.append(instance)
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
