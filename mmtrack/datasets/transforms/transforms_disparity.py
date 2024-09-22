# This script modify transforms to adapt disparity.
import math
from typing import Dict, List, Optional, Tuple, Union

import cv2
import mmcv
import numpy as np
from mmcv.transforms import BaseTransform
from mmcv.transforms.utils import cache_randomness
from mmengine.logging import print_log

from mmcv.image.geometric import _scale_size
from mmdet.datasets.transforms import Resize as MMDET_Resize
from mmdet.datasets.transforms import Pad as MMDET_Pad
from mmdet.datasets.transforms import RandomFlip as MMDET_RandomFlip
from mmdet.structures.bbox import HorizontalBoxes, autocast_box_type

from mmtrack.registry import TRANSFORMS
from mmtrack.utils import crop_image


@TRANSFORMS.register_module()
class Resize_Disparity(MMDET_Resize):
    """Resize images & bbox & seg & disparity.

    This transform resizes the input image according to ``scale`` or
    ``scale_factor``. Bboxes, masks, seg map and disparity map are then resized
    with the same scale factor.
    if ``scale`` and ``scale_factor`` are both set, it will use ``scale`` to
    resize.

    Required Keys:

    - img
    - disp_postp
    - disp_mask
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_masks (BitmapMasks | PolygonMasks) (optional)
    - gt_seg_map (np.uint8) (optional)

    Modified Keys:

    - img
    - img_shape
    - disp_postp
    - disp_mask
    - gt_bboxes
    - gt_masks
    - gt_seg_map
    """

    def _resize_disp_postp(self, results: dict) -> None:
        """Resize disparity map with ``results['scale']``."""
        if results.get('disp_postp', None) is not None:
            if self.keep_ratio:
                disp_postp = mmcv.imrescale(
                    results['disp_postp'],
                    results['scale'],
                    interpolation='nearest',
                    backend=self.backend)
            else:
                disp_postp = mmcv.imresize(
                    results['disp_postp'],
                    results['scale'],
                    interpolation='nearest',
                    backend=self.backend)
            if np.ndim(disp_postp) == 2 and \
                    np.ndim(disp_postp) < np.ndim(results['disp_postp']):
                disp_postp = disp_postp[:, :, None]
            results['disp_postp'] = disp_postp
            
    def _resize_depth_postp(self, results: dict) -> None:
        """Resize depth map with ``results['scale']``."""
        if results.get('depth_postp', None) is not None:
            if self.keep_ratio:
                depth_postp = mmcv.imrescale(
                    results['depth_postp'],
                    results['scale'],
                    interpolation='nearest',
                    backend=self.backend)
            else:
                depth_postp = mmcv.imresize(
                    results['depth_postp'],
                    results['scale'],
                    interpolation='nearest',
                    backend=self.backend)
            if np.ndim(depth_postp) == 2 and \
                    np.ndim(depth_postp) < np.ndim(results['depth_postp']):
                depth_postp = depth_postp[:, :, None]
            results['depth_postp'] = depth_postp

    def _resize_disp_mask(self, results: dict) -> None:
        """Resize disparity mask with ``results['scale']``."""
        if results.get('disp_mask', None) is not None:
            if self.keep_ratio:
                disp_mask = mmcv.imrescale(
                    results['disp_mask'],
                    results['scale'],
                    interpolation='nearest',
                    backend=self.backend)
            else:
                disp_mask = mmcv.imresize(
                    results['disp_mask'],
                    results['scale'],
                    interpolation='nearest',
                    backend=self.backend)
            if np.ndim(disp_mask) == 2 and \
                    np.ndim(disp_mask) < np.ndim(results['disp_mask']):
                disp_mask = disp_mask[:, :, None]
            results['disp_mask'] = disp_mask

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        """Transform function to resize images, bounding boxes and semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Resized results, 'img', 'gt_bboxes', 'gt_seg_map',
            'scale', 'scale_factor', 'height', 'width', and 'keep_ratio' keys
            are updated in result dict.
        """
        if self.scale:
            results['scale'] = self.scale
        else:
            img_shape = results['img'].shape[:2]
            results['scale'] = _scale_size(img_shape[::-1], self.scale_factor)
        self._resize_img(results)
        self._resize_disp_postp(results)
        self._resize_disp_mask(results)
        self._resize_depth_postp(results)
        self._resize_bboxes(results)
        self._resize_masks(results)
        self._resize_seg(results)
        self._record_homography_matrix(results)
        return results

@TRANSFORMS.register_module()
class Pad_Disparity(MMDET_Pad):
    """Pad the image & segmentation map & disparity map.

    There are three padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number. and (3)pad to square. Also,
    pad to square and pad to the minimum size can be used as the same time.

    Required Keys:

    - img
    - disp_postp
    - disp_mask
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_masks (BitmapMasks | PolygonMasks) (optional)
    - gt_seg_map (np.uint8) (optional)

    Modified Keys:

    - img
    - disp_postp
    - disp_mask
    - img_shape
    - gt_masks
    - gt_seg_map
    """

    def __init__(self,
                 size: Optional[Tuple[int, int]] = None,
                 size_divisor: Optional[int] = None,
                 pad_to_square: bool = False,
                 pad_val: Union[Union[int, float], dict] = dict(img=0, seg=255, disp=0, disp_mask=0),
                 padding_mode: str = 'constant') -> None:

        if isinstance(pad_val, int):
            pad_val = dict(img=pad_val, seg=255, disp=0, disp_mask=0)
        super().__init__(size, size_divisor, pad_to_square, pad_val, padding_mode)

    def _pad_disp_postp(self, results: dict) -> None:
        """Pad post-processed disparity map according to
        ``results['pad_shape']``."""
        if results.get('disp_postp', None) is not None:
            pad_val = self.pad_val.get('disp_postp', 0)
            if isinstance(pad_val, int) and results['disp_postp'].ndim == 3:
                pad_val = tuple(
                    pad_val for _ in range(results['disp_postp'].shape[2]))
            disp_postp = mmcv.impad(
                results['disp_postp'],
                shape=results['pad_shape'][:2],
                pad_val=pad_val,
                padding_mode=self.padding_mode)

            if np.ndim(disp_postp) == 2 and \
                    np.ndim(disp_postp) < np.ndim(results['disp_postp']):
                disp_postp = disp_postp[:, :, None]
            results['disp_postp'] = disp_postp
            
    def _pad_depth_postp(self, results: dict) -> None:
        """Pad post-processed depth map according to
        ``results['pad_shape']``."""
        if results.get('depth_postp', None) is not None:
            pad_val = self.pad_val.get('depth_postp', 0)
            if isinstance(pad_val, int) and results['depth_postp'].ndim == 3:
                pad_val = tuple(
                    pad_val for _ in range(results['depth_postp'].shape[2]))
            depth_postp = mmcv.impad(
                results['depth_postp'],
                shape=results['pad_shape'][:2],
                pad_val=pad_val,
                padding_mode=self.padding_mode)

            if np.ndim(depth_postp) == 2 and \
                    np.ndim(depth_postp) < np.ndim(results['depth_postp']):
                depth_postp = depth_postp[:, :, None]
            results['depth_postp'] = depth_postp

    def _pad_disp_mask(self, results: dict) -> None:
        """Pad post-processed disparity map according to
        ``results['pad_shape']``."""
        if results.get('disp_mask', None) is not None:
            pad_val = self.pad_val.get('disp_mask', 0)
            if isinstance(pad_val, int) and results['disp_mask'].ndim == 3:
                pad_val = tuple(
                    pad_val for _ in range(results['disp_mask'].shape[2]))
            disp_mask = mmcv.impad(
                results['disp_mask'],
                shape=results['pad_shape'][:2],
                pad_val=pad_val,
                padding_mode=self.padding_mode)

            if np.ndim(disp_mask) == 2 and \
                    np.ndim(disp_mask) < np.ndim(results['disp_mask']):
                disp_mask = disp_mask[:, :, None]
            results['disp_mask'] = disp_mask

    def transform(self, results: dict) -> dict:
        """Call function to pad images, masks, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        self._pad_disp_postp(results)
        self._pad_disp_mask(results)
        self._pad_depth_postp(results)
        self._pad_seg(results)
        self._pad_masks(results)
        return results


@TRANSFORMS.register_module()
class RandomFlip_Disparity(MMDET_RandomFlip):
    """Flip the image & bbox & mask & segmentation map & disparity map.

    Required Keys:

    - img
    - disp_postp
    - disp_mask
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_masks (BitmapMasks | PolygonMasks) (optional)
    - gt_seg_map (np.uint8) (optional)

    Modified Keys:

    - img
    - disp_postp
    - disp_mask
    - gt_bboxes
    - gt_masks
    - gt_seg_map
    """

    @autocast_box_type()
    def _flip(self, results: dict) -> None:
        """Flip images, bounding boxes, and semantic segmentation map."""
        # flip image
        results['img'] = mmcv.imflip(
            results['img'], direction=results['flip_direction'])

        # flip post-processed disparity
        results['disp_postp'] = mmcv.imflip(
            results['disp_postp'], direction=results['flip_direction'])

        # flip disparity mask
        results['disp_mask'] = mmcv.imflip(
            results['disp_mask'], direction=results['flip_direction'])

        img_shape = results['img'].shape[:2]

        # flip post-processed depth
        if results.get('depth_postp', None) is not None:
            results['depth_postp'] = mmcv.imflip(
                results['depth_postp'], direction=results['flip_direction'])

        # flip bboxes
        if results.get('gt_bboxes', None) is not None:
            results['gt_bboxes'].flip_(img_shape, results['flip_direction'])

        # flip masks
        if results.get('gt_masks', None) is not None:
            results['gt_masks'] = results['gt_masks'].flip(
                results['flip_direction'])

        # flip segs
        if results.get('gt_seg_map', None) is not None:
            results['gt_seg_map'] = mmcv.imflip(
                results['gt_seg_map'], direction=results['flip_direction'])

        # record homography matrix for flip
        self._record_homography_matrix(results)


@TRANSFORMS.register_module()
class RandCutout(BaseTransform):
    """Randomly mask out patches from an image.
    (In a mask: 0 refers to pixels masked out, 1 refers to remained pixels)

    Args:
        patch_size (int, tuple): size of patches, (h, w).
        cut_ratio (float | tuple[float], [0~1)): percentage of patches masked out.
        prob: rate of running RandCutout.

    Required keys:
    - img

    Modified Keys:
    - img_cut_mask (optional)
    - disp_cut_mask (optional)
    """

    def __init__(self, patch_size, cut_ratio,
                 keys=['disp'], prob=1.):
        assert isinstance(patch_size, (tuple, list))

        self.patch_size = patch_size
        self.cut_ratio = cut_ratio
        self.keys = keys
        self.prob = prob

    def transform(self, results: dict) -> dict:
        assert results.get('img_shape', None) is not None
        img_shape = results['img_shape']
        num_h = img_shape[0] // self.patch_size[0]
        num_w = img_shape[1] // self.patch_size[1]
        num_patches = num_h * num_w

        if np.random.uniform() > self.prob:
            for key in self.keys:
                mask = np.ones(img_shape, np.uint8)
                mask = mask[:, :, None]
                results[key + '_cut_mask'] = mask

        else:
            for key in self.keys:
                if isinstance(self.cut_ratio, (tuple, list)):
                    ratio = np.random.uniform(self.cut_ratio[0], self.cut_ratio[1])
                else:
                    ratio = self.cut_ratio
                num_mask = int(ratio * num_patches)
                idx_mask = np.hstack([
                    np.ones(num_patches - num_mask),
                    np.zeros(num_mask)])
                np.random.shuffle(idx_mask)
                idx_mask = idx_mask.reshape(num_h, num_w)
                _mask = mmcv.imresize(
                    idx_mask,
                    (num_w * self.patch_size[1], num_h * self.patch_size[0]),
                    interpolation='nearest')
                mask = np.ones(img_shape, np.uint8)
                mask[: _mask.shape[0], : _mask.shape[1]] = _mask
                mask = mask[:, :, None]
                results[key + '_cut_mask'] = mask

        return results