# This script is adapted from mmlab formatting.py to support format disparity data.
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from mmcv.transforms import to_tensor
from mmcv.transforms.base import BaseTransform
from mmengine.structures import InstanceData, PixelData
from mmdet.structures.mask import BitmapMasks

from mmtrack.registry import TRANSFORMS
from mmtrack.structures import TrackDataSample
from mmdet.structures import DetDataSample
from mmdet.structures.bbox import BaseBoxes


@TRANSFORMS.register_module()
class PackTrackInputs_Disparity(BaseTransform):
    """Pack the inputs data for the video object detection / multi object tracking.

    For each value (``list`` type) in the input dict, we concat the first
    `num_key_frames` elements to the first dict with a new key, and the rest
    of elements are concated to the second dict with a new key.
    All the information of images are packed to ``inputs``.
    All the information except images are packed to ``data_samples``.

    Args:
        ref_prefix (str): The prefix of key added to the `reference` frames.
            Defaults to 'ref'.
        num_key_frames (int) The number of key frames. Defaults to 1.
        pack_single_img (bool, optional): Whether to only pack single image. If
            True, pack the data as a list additionally. Defaults to False.
        meta_keys (Sequence[str]): Meta keys to be collected in
            ``data_sample.metainfo``. Defaults to None.
        default_meta_keys (tuple): Default meta keys. Defaults to ('img_id`,
            `img_path`, `ori_shape`, `img_shape`, `scale_factor`,
            `flip`, `flip_direction`, `frame_id`, `is_video_data`,
            `video_id`, `video_length`, `instances`, 'num_left_ref_imgs',
            'frame_stride','cat2label').
    """
    mapping_table = {
        'gt_bboxes': 'bboxes',
        'gt_bboxes_labels': 'labels',
        'gt_masks': 'masks',
        'gt_instances_id': 'instances_id'
    }

    def __init__(self,
                 ref_prefix: str = 'ref',
                 num_key_frames: int = 1,
                 pack_single_img: Optional[bool] = False,
                 meta_keys: Optional[dict] = None,
                 default_meta_keys: tuple = ('img_id', 'img_path', 'ori_shape',
                                             'img_shape', 'scale_factor',
                                             'flip', 'flip_direction',
                                             'frame_id', 'is_video_data',
                                             'video_id', 'video_length',
                                             'instances', 'num_left_ref_imgs',
                                             'frame_stride', 'cat2label')):
        self.ref_prefix = ref_prefix
        self.num_key_frames = num_key_frames
        self.meta_keys = default_meta_keys
        if meta_keys is not None:
            if isinstance(meta_keys, str):
                meta_keys = (meta_keys, )
            else:
                assert isinstance(meta_keys, tuple), \
                    'meta_keys must be str or tuple'
            self.meta_keys += meta_keys

        self.pack_single_img = pack_single_img

    def _cat_same_type_data(self,
                            data: Union[List, int],
                            return_ndarray: bool = True,
                            axis: int = 0,
                            stack: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Concatenate data with the same type.

        Args:
            data (Union[List, int]): Input data.
            return_ndarray (bool, optional): Whether to return ``np.ndarray``.
                Defaults to True.
            axis (int, optional): The axis that concatenating along. Defaults
                to 0.
            stack (bool, optional): Whether to stack all the data. If not,
                using the concatenating operation. Defaults to False.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The first element is the
                concatenated data of key frames, and the second element is the
                concatenated data of reference frames.
        """
        if self.pack_single_img:
            data = [data]
        key_data = data[:self.num_key_frames]
        ref_data = data[self.num_key_frames:] if len(
            data) > self.num_key_frames else None
        
        if return_ndarray:
            if stack:
                key_data = np.stack(key_data, axis=axis)
                if ref_data is not None:
                    ref_data = np.stack(ref_data, axis=axis)
            else:
                key_data = np.concatenate(key_data, axis=axis)
                if ref_data is not None:
                    ref_data = np.concatenate(ref_data, axis=axis)
        
        return key_data, ref_data
    
    def _get_img_idx_map(self, anns: List) -> Tuple[np.ndarray, np.ndarray]:
        """Get the index of images for the annotations. The multiple instances
        in one image need to be denoted the image index when concatenating
        multiple images.
        
        Args:
            anns (List): Input annotations.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: The first element is the
                concatenated indexes of key frames, and the second element is
                the concatenated indexes of reference frames.
        """
        if self.pack_single_img:
            anns = [anns]
        key_img_idx_map = []
        for img_idx, ann in enumerate(anns[:self.num_key_frames]):
            key_img_idx_map.extend([img_idx] * len(ann))
        key_img_idx_map = np.array(key_img_idx_map, dtype=np.int32)
        if len(anns) > self.num_key_frames:
            ref_img_idx_map = []
            for img_idx, ann in enumerate(anns[self.num_key_frames:]):
                ref_img_idx_map.extend([img_idx] * len(ann))
            ref_img_idx_map = np.array(ref_img_idx_map, dtype=np.int32)
        else:
            ref_img_idx_map = None
        return key_img_idx_map, ref_img_idx_map
    
    def transform(self, results: dict) -> Optional[dict]:
        """Method to pack the input data.
        
        Args:
            results (dict): Result dict from the data pipeline.
            
        Returns:
            dict:
            - `inputs` (dict[Tensor]): The forward data of models.
            - `data_samples` (obj:`TrackDataSample`): The annotation info of
                the samples
        """
        packed_results = dict()
        packed_results['inputs'] = dict()
        
        # 1. Pack images
        if 'img' in results:
            imgs = results['img']
            key_imgs, ref_imgs = self._cat_same_type_data(imgs, stack=True)
            key_imgs = key_imgs.transpose(0, 3, 1, 2)
            packed_results['inputs']['img'] = to_tensor(key_imgs)
            
            if ref_imgs is not None:
                ref_imgs = ref_imgs.transpose(0, 3, 1, 2)
                packed_results['inputs'][f'{self.ref_prefix}_img'] = to_tensor(
                    ref_imgs)
        
        # 2. Pack disparity maps
        if 'disp_postp' in results:
            disp_postps = results['disp_postp']
            key_disp_postps, ref_disp_postps = self._cat_same_type_data(disp_postps, stack=True)
            key_disp_postps = key_disp_postps.transpose(0, 3, 1, 2)
            packed_results['inputs']['disp_postp'] = to_tensor(key_disp_postps)
            
            if ref_disp_postps is not None:
                ref_disp_postps = ref_disp_postps.transpose(0, 3, 1, 2)
                packed_results['inputs'][f'{self.ref_prefix}_disp_postp'] = to_tensor(
                    ref_disp_postps)

        # 3. Pack disparity masks
        if 'disp_mask' in results:
            disp_masks = results['disp_mask']
            key_disp_masks, ref_disp_masks = self._cat_same_type_data(disp_masks, stack=True)
            key_disp_masks = key_disp_masks.transpose(0, 3, 1, 2)
            packed_results['inputs']['disp_mask'] = to_tensor(key_disp_masks)

            if ref_disp_masks is not None:
                ref_disp_masks = ref_disp_masks.transpose(0, 3, 1, 2)
                packed_results['inputs'][f'{self.ref_prefix}_disp_mask'] = to_tensor(
                    ref_disp_masks)
        if 'disp_cut_mask' in results:
            disp_cut_masks = results['disp_cut_mask']
            key_disp_cut_masks, ref_disp_cut_masks = self._cat_same_type_data(disp_cut_masks, stack=True)
            key_disp_cut_masks = key_disp_cut_masks.transpose(0, 3, 1, 2)
            packed_results['inputs']['disp_cut_mask'] = to_tensor(key_disp_cut_masks)

            if ref_disp_cut_masks is not None:
                ref_disp_cut_masks = ref_disp_cut_masks.transpose(0, 3, 1, 2)
                packed_results['inputs'][f'{self.ref_prefix}_disp_cut_mask'] = to_tensor(
                    ref_disp_cut_masks)

        # 4. Pack depth maps
        if 'depth_postp' in results:
            depth_postps = results['depth_postp']
            key_depth_postps, ref_depth_postps = self._cat_same_type_data(depth_postps, stack=True)
            key_depth_postps = key_depth_postps.transpose(0, 3, 1, 2)
            packed_results['inputs']['depth_postp'] = to_tensor(key_depth_postps)

            if ref_depth_postps is not None:
                ref_depth_postps = ref_depth_postps.transpose(0, 3, 1, 2)
                packed_results['inputs'][f'{self.ref_prefix}_depth_postp'] = to_tensor(
                    ref_depth_postps)

        data_sample = TrackDataSample()

        # 5. Pack InstanceData
        if 'gt_ignore_flags' in results:
            gt_ignore_flags = results['gt_ignore_flags']
            (key_gt_ignore_flags,
             ref_gt_ignore_flags) = self._cat_same_type_data(gt_ignore_flags)
            key_valid_idx = key_gt_ignore_flags == 0
            if ref_gt_ignore_flags is not None:
                ref_valid_idx = ref_gt_ignore_flags == 0

        instance_data = InstanceData()
        ignore_instance_data = InstanceData()
        ref_instance_data = InstanceData()
        ref_ignore_instance_data = InstanceData()
        
        # Flag that whether have recorded the image index
        img_idx_map_flag = False
        for key in self.mapping_table.keys():
            if key not in results:
                continue
            if key == 'gt_masks':
                gt_masks = results[key]
                gt_masks_ndarray = [
                    mask.to_ndarray() for mask in gt_masks
                ] if isinstance(gt_masks, list) else gt_masks.to_ndarray()
                key_gt_masks, ref_gt_masks = self._cat_same_type_data(
                    gt_masks_ndarray)

                mapped_key = self.mapping_table[key]
                if 'gt_ignore_flags' in results:
                    instance_data[mapped_key] = BitmapMasks(
                        key_gt_masks[key_valid_idx], *key_gt_masks.shape[-2:])
                    ignore_instance_data[mapped_key] = BitmapMasks(
                        key_gt_masks[~key_valid_idx], *key_gt_masks.shape[-2:])

                    if ref_gt_masks is not None:
                        ref_instance_data[mapped_key] = BitmapMasks(
                            ref_gt_masks[ref_valid_idx],
                            *key_gt_masks.shape[-2:])
                        ref_ignore_instance_data[mapped_key] = BitmapMasks(
                            ref_gt_masks[~ref_valid_idx],
                            *key_gt_masks.shape[-2:])
                else:
                    instance_data[mapped_key] = BitmapMasks(
                        key_gt_masks, *key_gt_masks.shape[-2:])
                    if ref_gt_masks is not None:
                        ref_instance_data[mapped_key] = BitmapMasks(
                            ref_gt_masks, *ref_gt_masks.shape[-2:])

            else:
                anns = results[key]
                key_anns, ref_anns = self._cat_same_type_data(anns)

                if not img_idx_map_flag:
                    # The multiple instances in one image need to be
                    # denoted the image index when concatenating multiple
                    # images.
                    key_img_idx_map, ref_img_idx_map = self._get_img_idx_map(
                        anns)
                    img_idx_map_flag = True

                mapped_key = self.mapping_table[key]
                if 'gt_ignore_flags' in results:
                    instance_data[mapped_key] = to_tensor(
                        key_anns[key_valid_idx])
                    ignore_instance_data[mapped_key] = to_tensor(
                        key_anns[~key_valid_idx])
                    instance_data['map_instances_to_img_idx'] = to_tensor(
                        key_img_idx_map[key_valid_idx])
                    ignore_instance_data[
                        'map_instances_to_img_idx'] = to_tensor(
                        key_img_idx_map[~key_valid_idx])

                    if ref_anns is not None:
                        ref_instance_data[mapped_key] = to_tensor(
                            ref_anns[ref_valid_idx])
                        ref_ignore_instance_data[mapped_key] = to_tensor(
                            ref_anns[~ref_valid_idx])
                        ref_instance_data[
                            'map_instances_to_img_idx'] = to_tensor(
                            ref_img_idx_map[ref_valid_idx])
                        ref_ignore_instance_data[
                            'map_instances_to_img_idx'] = to_tensor(
                            ref_img_idx_map[~ref_valid_idx])
                else:
                    instance_data[mapped_key] = to_tensor(key_anns)
                    instance_data['map_instances_to_img_idx'] = to_tensor(
                        key_img_idx_map)
                    if ref_anns is not None:
                        ref_instance_data[mapped_key] = to_tensor(ref_anns)
                        ref_instance_data[
                            'map_instances_to_img_idx'] = to_tensor(
                            ref_img_idx_map)

        data_sample.gt_instances = instance_data
        data_sample.ignored_instances = ignore_instance_data
        setattr(data_sample, f'{self.ref_prefix}_gt_instances',
                ref_instance_data)
        setattr(data_sample, f'{self.ref_prefix}_ignored_instances',
                ref_ignore_instance_data)

        # 6. Pack metainfo
        new_img_metas = {}
        for key in self.meta_keys:
            if key not in results:
                continue
            img_metas = results[key]
            key_img_metas, ref_img_metas = self._cat_same_type_data(
                img_metas, return_ndarray=False)
            # To compatible the interface of ``MMDet``, we don't use
            # the fotmat of list when the length of meta information is
            # equal to 1.
            if len(key_img_metas) > 1:
                new_img_metas[key] = key_img_metas
            else:
                new_img_metas[key] = key_img_metas[0]
            if ref_img_metas is not None:
                if len(ref_img_metas) > 1:
                    new_img_metas[f'{self.ref_prefix}_{key}'] = ref_img_metas
                else:
                    new_img_metas[f'{self.ref_prefix}_{key}'] = ref_img_metas[
                        0]

        data_sample.set_metainfo(new_img_metas)
        packed_results['data_samples'] = data_sample
        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(ref_prefix={self.ref_prefix}, '
        repr_str += f'meta_keys={self.meta_keys}, '
        repr_str += f'default_meta_keys={self.default_meta_keys})'
        return repr_str
                    