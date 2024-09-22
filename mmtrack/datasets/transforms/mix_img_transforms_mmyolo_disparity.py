# This script inherited from mmyolo mix_img_transforms.py for processing disparity data.
import collections
import copy
from typing import Optional, Sequence, Tuple, Union

import mmcv
import numpy as np
from numpy import random
from mmyolo.datasets.transforms.mix_img_transforms import BaseMixImageTransform
from mmyolo.datasets.transforms.mix_img_transforms import Mosaic, YOLOXMixUp
from mmdet.structures.bbox import autocast_box_type, HorizontalBoxes

from mmtrack.registry import TRANSFORMS


@TRANSFORMS.register_module()
class Mosaic_Disparity(Mosaic):
    """ Mosaic augmentation for rgb+disparity

    Required Keys:

    - img
    - disp_postp
    - disp_mask
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_bboxes_labels (np.int64) (optional)
    - gt_ignore_flags (np.bool) (optional)
    - mix_results (List[dict])


    Modified Keys:

    - img
    - disp_postp
    - disp_mask
    - img_shape
    - gt_bboxes (optional)
    - gt_bboxes_labels (optional)
    - gt_ignore_flags (optional)
    """

    def mix_img_transform(self, results: dict) -> dict:
        """Mixed image data transformation.

        Args:
            results (dict): Result dict.

        Returns:
            results (dict): Updated result dict.
        """
        assert 'mix_results' in results
        assert 'disp_postp' in results, 'disparity does not existing!'
        mosaic_bboxes = []
        mosaic_bboxes_labels = []
        mosaic_ignore_flags = []

        if len(results['img'].shape) == 3:
            mosaic_img = np.full(
                (int(self.img_scale[0] * 2), int(self.img_scale[1] * 2), 3),
                self.pad_val,
                dtype=results['img'].dtype)
        else:
            mosaic_img = np.full(
                (int(self.img_scale[0] * 2), int(self.img_scale[1] * 2)),
                self.pad_val,
                dtype=results['img'].dtype)

        if results['disp_postp'].shape[-1] == 3:
            mosaic_disp_postp = np.full(
                (int(self.img_scale[0] * 2), int(self.img_scale[1] * 2), 3),
                0.,
                dtype=results['disp_postp'].dtype)
        elif results['disp_postp'].shape[-1] == 1:
            mosaic_disp_postp = np.full(
                (int(self.img_scale[0] * 2), int(self.img_scale[1] * 2), 1),
                0.,
                dtype=results['disp_postp'].dtype)
        else:
            raise NotImplementedError('Mosaic_Disparity only support disparity channel = 3 or 1')

        mosaic_disp_mask = np.full(
            (int(self.img_scale[0] * 2), int(self.img_scale[1] * 2), 1),
            0.,
            dtype=results['disp_mask'].dtype)

        # mosaic center x, y
        center_x = int(
            random.uniform(*self.center_ratio_range) * self.img_scale[1])
        center_y = int(
            random.uniform(*self.center_ratio_range) * self.img_scale[0])
        center_position = (center_x, center_y)

        loc_strs = ('top_left', 'top_right', 'bottom_left', 'bottom_right')
        for i, loc in enumerate(loc_strs):
            if loc == 'top_left':
                results_patch = results
            else:
                results_patch = results['mix_results'][i - 1]

            img_i = results_patch['img']
            disp_postp_i = results_patch['disp_postp']
            disp_mask_i = results_patch['disp_mask']
            h_i, w_i = img_i.shape[:2]
            # keep_ratio resize
            scale_ratio_i = min(self.img_scale[0] / h_i,
                                self.img_scale[1] / w_i)
            img_i = mmcv.imresize(
                img_i, (int(w_i * scale_ratio_i), int(h_i * scale_ratio_i)))
            disp_postp_i = mmcv.imresize(
                disp_postp_i, (int(w_i * scale_ratio_i), int(h_i * scale_ratio_i)))

            # compute the combine parameters
            paste_coord, crop_coord = self._mosaic_combine(
                loc, center_position, img_i.shape[:2][::-1])
            x1_p, y1_p, x2_p, y2_p = paste_coord
            x1_c, y1_c, x2_c, y2_c = crop_coord

            # crop and paste image
            mosaic_img[y1_p:y2_p, x1_p:x2_p] = img_i[y1_c:y2_c, x1_c:x2_c]
            mosaic_disp_postp[y1_p:y2_p, x1_p:x2_p] = disp_postp_i[y1_c:y2_c, x1_c:x2_c]
            mosaic_disp_mask[y1_p:y2_p, x1_p:x2_p] = disp_mask_i[y1_c:y2_c, x1_c:x2_c]

            # adjust coordinate
            gt_bboxes_i = results_patch['gt_bboxes']
            gt_bboxes_labels_i = results_patch['gt_bboxes_labels']
            gt_ignore_flags_i = results_patch['gt_ignore_flags']

            padw = x1_p - x1_c
            padh = y1_p - y1_c
            gt_bboxes_i.rescale_([scale_ratio_i, scale_ratio_i])
            gt_bboxes_i.translate_([padw, padh])
            mosaic_bboxes.append(gt_bboxes_i)
            mosaic_bboxes_labels.append(gt_bboxes_labels_i)
            mosaic_ignore_flags.append(gt_ignore_flags_i)

        mosaic_bboxes = mosaic_bboxes[0].cat(mosaic_bboxes, 0)
        mosaic_bboxes_labels = np.concatenate(mosaic_bboxes_labels, 0)
        mosaic_ignore_flags = np.concatenate(mosaic_ignore_flags, 0)

        if self.bbox_clip_border:
            mosaic_bboxes.clip_([2 * self.img_scale[0], 2 * self.img_scale[1]])
        else:
            # remove outside bboxes
            inside_inds = mosaic_bboxes.is_inside(
                [2 * self.img_scale[0], 2 * self.img_scale[1]]).numpy()
            mosaic_bboxes = mosaic_bboxes[inside_inds]
            mosaic_bboxes_labels = mosaic_bboxes_labels[inside_inds]
            mosaic_ignore_flags = mosaic_ignore_flags[inside_inds]

        results['img'] = mosaic_img
        results['disp_postp'] = mosaic_disp_postp
        results['disp_mask'] = mosaic_disp_mask
        results['img_shape'] = mosaic_img.shape
        results['gt_bboxes'] = mosaic_bboxes
        results['gt_bboxes_labels'] = mosaic_bboxes_labels
        results['gt_ignore_flags'] = mosaic_ignore_flags
        return results


@TRANSFORMS.register_module()
class YOLOXMixUp_Disparity(YOLOXMixUp):
    """MixUp data augmentation for YOLOX for disparity data.

    Required Keys:

    - img
    - diap_postp
    - disp_mask
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_bboxes_labels (np.int64) (optional)
    - gt_ignore_flags (np.bool) (optional)
    - mix_results (List[dict])


    Modified Keys:

    - img
    - diap_postp
    - disp_mask
    - img_shape
    - gt_bboxes (optional)
    - gt_bboxes_labels (optional)
    - gt_ignore_flags (optional)
    """

    def mix_img_transform(self, results: dict) -> dict:
        assert 'mix_results' in results
        assert 'disp_postp' in results, 'disparity does not existing!'
        assert len(
            results['mix_results']) == 1, 'MixUp only support 2 images now !'

        if results['mix_results'][0]['gt_bboxes'].shape[0] == 0:
            # empty bbox
            return results

        retrieve_results = results['mix_results'][0]
        retrieve_results['gt_bboxes'] = HorizontalBoxes(retrieve_results['gt_bboxes'], clone=False)  # convert the np.ndarray to HorizontalBoxes
        retrieve_img = retrieve_results['img']
        retrieve_disp_postp = retrieve_results['disp_postp']
        retrieve_disp_mask = retrieve_results['disp_mask']

        jit_factor = random.uniform(*self.ratio_range)
        is_filp = random.uniform(0, 1) > self.flip_ratio

        if len(retrieve_img.shape) == 3:
            out_img = np.ones((self.img_scale[0], self.img_scale[1], 3),
                              dtype=retrieve_img.dtype) * self.pad_val
        else:
            out_img = np.ones(
                self.img_scale, dtype=retrieve_img.dtype) * self.pad_val

        if retrieve_disp_postp.shape[-1] == 3:
            out_disp_postp = np.zeros((self.img_scale[0], self.img_scale[1], 3),
                                      dtype=retrieve_disp_postp.dtype)
        elif retrieve_disp_postp.shape[-1] == 1:
            out_disp_postp = np.zeros((self.img_scale[0], self.img_scale[1], 1),
                                      dtype=retrieve_disp_postp.dtype)
        else:
            raise NotImplementedError('only support disparity channel = 3 or 1')

        out_disp_mask = np.zeros((self.img_scale[0], self.img_scale[1], 1),
                                 dtype=retrieve_disp_postp.dtype)

        # 1. keep_ratio resize
        scale_ratio = min(self.img_scale[0] / retrieve_img.shape[0],
                          self.img_scale[1] / retrieve_img.shape[1])
        retrieve_img = mmcv.imresize(
            retrieve_img, (int(retrieve_img.shape[1] * scale_ratio),
                           int(retrieve_img.shape[0] * scale_ratio)))
        retrieve_disp_postp = mmcv.imresize(
            retrieve_disp_postp, (int(retrieve_disp_postp.shape[1] * scale_ratio),
                                  int(retrieve_disp_postp.shape[0] * scale_ratio)))
        retrieve_disp_mask = mmcv.imresize(
            retrieve_disp_mask, (int(retrieve_disp_mask.shape[1] * scale_ratio),
                                 int(retrieve_disp_mask.shape[0] * scale_ratio)))[:, :, None]

        # 2. paste
        out_img[:retrieve_img.shape[0], :retrieve_img.shape[1]] = retrieve_img
        out_disp_postp[:retrieve_disp_postp.shape[0], :retrieve_disp_postp.shape[1]] = retrieve_disp_postp
        out_disp_mask[:retrieve_disp_mask.shape[0], :retrieve_disp_mask.shape[1]] = retrieve_disp_mask

        # 3. scale jit
        scale_ratio *= jit_factor
        out_img = mmcv.imresize(out_img, (int(out_img.shape[1] * jit_factor),
                                          int(out_img.shape[0] * jit_factor)))
        out_disp_postp = mmcv.imresize(out_disp_postp, (int(out_disp_postp.shape[1] * jit_factor),
                                                        int(out_disp_postp.shape[0] * jit_factor)))
        out_disp_mask = mmcv.imresize(out_disp_mask, (int(out_disp_mask.shape[1] * jit_factor),
                                                      int(out_disp_mask.shape[0] * jit_factor)))[:, :, None]

        # 4. flip
        if is_filp:
            out_img = out_img[:, ::-1, :]
            out_disp_postp = out_disp_postp[:, ::-1, :]
            out_disp_mask = out_disp_mask[:, ::-1, :]

        # 5. random crop
        ori_img = results['img']
        ori_disp_postp = results['disp_postp']
        ori_disp_mask = results['disp_mask']
        origin_h, origin_w = out_img.shape[:2]
        target_h, target_w = ori_img.shape[:2]

        padded_img = np.ones((max(origin_h, target_h), max(
            origin_w, target_w), 3)) * self.pad_val
        padded_img = padded_img.astype(np.uint8)
        padded_img[:origin_h, :origin_w] = out_img

        padded_disp_postp = np.zeros((max(origin_h, target_h), max(
            origin_w, target_w), 3))
        padded_disp_postp = padded_disp_postp.astype(np.uint8)
        padded_disp_postp[:origin_h, :origin_w] = out_disp_postp

        padded_disp_mask = np.zeros((max(origin_h, target_h), max(
            origin_w, target_w), 1))
        padded_disp_mask = padded_disp_mask.astype(np.uint8)
        padded_disp_mask[:origin_h, :origin_w] = out_disp_mask

        x_offset, y_offset = 0, 0
        if padded_img.shape[0] > target_h:
            y_offset = random.randint(0, padded_img.shape[0] - target_h)
        if padded_img.shape[1] > target_w:
            x_offset = random.randint(0, padded_img.shape[1] - target_w)
        padded_cropped_img = padded_img[y_offset:y_offset + target_h,
                             x_offset:x_offset + target_w]
        padded_cropped_disp_postp = padded_disp_postp[y_offset:y_offset + target_h,
                             x_offset:x_offset + target_w]
        padded_cropped_disp_mask = padded_disp_mask[y_offset:y_offset + target_h,
                             x_offset:x_offset + target_w]

        # 6. adjust bbox
        retrieve_gt_bboxes = retrieve_results['gt_bboxes']
        retrieve_gt_bboxes.rescale_([scale_ratio, scale_ratio])
        if self.bbox_clip_border:
            retrieve_gt_bboxes.clip_([origin_h, origin_w])

        if is_filp:
            retrieve_gt_bboxes.flip_([origin_h, origin_w],
                                     direction='horizontal')

        # 7. filter
        cp_retrieve_gt_bboxes = retrieve_gt_bboxes.clone()
        cp_retrieve_gt_bboxes.translate_([-x_offset, -y_offset])
        if self.bbox_clip_border:
            cp_retrieve_gt_bboxes.clip_([target_h, target_w])

        # 8. mix up
        mixup_img = 0.5 * ori_img + 0.5 * padded_cropped_img
        mixup_disp_postp = 0.5 * ori_disp_postp + 0.5 * padded_cropped_disp_postp
        mixup_disp_mask = ori_disp_mask | padded_cropped_disp_mask

        retrieve_gt_bboxes_labels = retrieve_results['gt_bboxes_labels']
        retrieve_gt_ignore_flags = retrieve_results['gt_ignore_flags']

        mixup_gt_bboxes = cp_retrieve_gt_bboxes.cat(
            (results['gt_bboxes'], cp_retrieve_gt_bboxes), dim=0)
        mixup_gt_bboxes_labels = np.concatenate(
            (results['gt_bboxes_labels'], retrieve_gt_bboxes_labels), axis=0)
        mixup_gt_ignore_flags = np.concatenate(
            (results['gt_ignore_flags'], retrieve_gt_ignore_flags), axis=0)

        if not self.bbox_clip_border:
            # remove outside bbox
            inside_inds = mixup_gt_bboxes.is_inside([target_h,
                                                     target_w]).numpy()
            mixup_gt_bboxes = mixup_gt_bboxes[inside_inds]
            mixup_gt_bboxes_labels = mixup_gt_bboxes_labels[inside_inds]
            mixup_gt_ignore_flags = mixup_gt_ignore_flags[inside_inds]

        results['img'] = mixup_img.astype(np.uint8)
        results['disp_postp'] = mixup_disp_postp.astype(np.uint8)
        results['disp_mask'] = mixup_disp_mask.astype(np.uint8)
        results['img_shape'] = mixup_img.shape
        results['gt_bboxes'] = mixup_gt_bboxes
        results['gt_bboxes_labels'] = mixup_gt_bboxes_labels
        results['gt_ignore_flags'] = mixup_gt_ignore_flags

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(img_scale={self.img_scale}, '
        repr_str += f'ratio_range={self.ratio_range}, '
        repr_str += f'flip_ratio={self.flip_ratio}, '
        repr_str += f'pad_val={self.pad_val}, '
        repr_str += f'max_refetch={self.max_refetch}, '
        repr_str += f'bbox_clip_border={self.bbox_clip_border})'
        return repr_str