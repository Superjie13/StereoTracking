# loading depth related data as input
import numpy as np

from typing import Optional

import mmengine
import mmcv
from mmcv.transforms import BaseTransform
from mmtrack.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadDisparityFromFile(BaseTransform):
    """Load a disparity map from file.

    Required Keys:

    - disp_path

    Modified Keys:

    - disp
    - disp_postp: Post-processed disparity map.
    - disp_mask: Filter out the valuable pixels.
    - img_shape: If not defined in somewhere else, it will be defined here.
    - ori_shape: If not defined in somewhere else, it will be defined here.

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint16 array.
            Defaults to False.
        to_3channel (bool): Whether to repeat channel 3 times to get h,w,3.
        post_processing (dict): Arguments to post-process the disparity map.
            Defaults to ``dict(disp_thr_h=1200, disp_thr_l=10)``
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:`mmcv.imfrombytes`.
            See :func:`mmcv.imfrombytes` for details.
            Defaults to 'cv2'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmengine.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        ignore_empty (bool): Whether to allow loading empty image or file path
            not existent. Defaults to False.
    """

    def __init__(self,
                 to_float32: bool = False,
                 to_3channel: bool = False,
                 post_processing: dict = dict(disp_thr_h=1200, disp_thr_l=10),
                 imdecode_backend: str = 'cv2',
                 file_client_args: dict = dict(backend='disk'),
                 ignore_empty: bool = False) -> None:
        self.ignore_empty = ignore_empty
        self.to_float32 = to_float32
        self.to_3channel = to_3channel
        self.post_processing = post_processing
        self.imdecode_backend = imdecode_backend
        self.file_client_args = file_client_args.copy()
        self.file_client = mmengine.FileClient(**self.file_client_args)
    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load disparity.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        filename = results['disp_path']
        try:
            disp_bytes = self.file_client.get(filename)
            disp = mmcv.imfrombytes(
                disp_bytes, flag='unchanged', backend=self.imdecode_backend)
        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e

        disp_mask = (disp < 65535).astype(np.uint8)
        results['disp_mask'] = disp_mask[:, :, None]  # H, W, C

        if self.to_3channel:
            disp = np.repeat(disp[:, :, None], 3, axis=-1)
            assert np.ndim(disp) == 3, f'Shape of disparity map must be 3 dim, i.e., (h, w, 3), ' \
                                       f'but got {np.ndim(disp)} dim'
        else:
            disp = disp[:, :, None]
        if self.to_float32:
            disp = disp.astype(np.float32)

        results['disp'] = disp
        if results.get('img_shape', None) is None:
            results['img_shape'] = disp.shape[: 2]
            results['ori_shape'] = disp.shape[: 2]

        if self.post_processing is not None:
            results['disp_postp'] = self._post_processing_v2(results)
        else:
            results['disp_postp'] = disp

        return results

    def _post_processing(self, results: dict):
        """Private function to post-processing disparity map.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains post-processed disparity map ('disp_postp').
        """
        disp_thr_h = self.post_processing['disp_thr_h']
        disp_thr_l = self.post_processing['disp_thr_l']

        disp_postp = results['disp']
        disp_postp[disp_postp > disp_thr_h] = 0
        disp_postp[disp_postp < disp_thr_l] = 0
        disp_postp = np.clip((disp_postp / disp_thr_h * 255), 0, 255).astype(np.uint8)

        if self.to_float32:
            disp_postp = disp_postp.astype(np.float32)

        return disp_postp

    def _post_processing_v2(self, results: dict):
        disp_postp = results['disp']
        disp_postp[disp_postp == 65535] = 0
        disp_postp = disp_postp.astype(np.float32) / 16.

        return disp_postp

    def __repr__(self) -> str:
        repr_str = (f'{self.__class__.__name__}('
                    f'ignore_empty={self.ignore_empty}, '
                    f'to_float32={self.to_float32}, '
                    f"to_3channel='{self.to_3channel}', "
                    f"post_processing={self.post_processing}, "
                    f"imdecode_backend='{self.imdecode_backend}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str

@TRANSFORMS.register_module()
class LoadDepthFromFile(BaseTransform):
    """Load a depth map from file.

    Required Keys:

    - depth_path

    Modified Keys:

    - depth
    - depth_postp: Filter out the valuable pixels.

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint16 array.
            Defaults to False.
        to_3channel (bool): Whether to repeat channel 3 times to get h,w,3.
        post_processing (dict): Arguments to post-process the disparity map.
            Defaults to ``dict(disp_thr_h=1200, disp_thr_l=10)``
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:`mmcv.imfrombytes`.
            See :func:`mmcv.imfrombytes` for details.
            Defaults to 'cv2'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmengine.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        ignore_empty (bool): Whether to allow loading empty image or file path
            not existent. Defaults to False.
    """

    def __init__(self,
                 to_float32: bool = False,
                 to_3channel: bool = False,
                 post_processing: dict = dict(thr_h=1000, thr_l=0),
                 inv_depth: bool = False,
                 imdecode_backend: str = 'cv2',
                 file_client_args: dict = dict(backend='disk'),
                 ignore_empty: bool = False) -> None:
        self.ignore_empty = ignore_empty
        self.to_float32 = to_float32
        self.to_3channel = to_3channel
        self.post_processing = post_processing
        self.inv_depth = inv_depth  # no use
        self.imdecode_backend = imdecode_backend
        self.file_client_args = file_client_args.copy()
        self.file_client = mmengine.FileClient(**self.file_client_args)

    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load disparity.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        filename = results['depth_path']
        try:
            depth_bytes = self.file_client.get(filename)
            depth = mmcv.imfrombytes(
                depth_bytes, flag='unchanged', backend=self.imdecode_backend)
        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e

        if 'selma' in filename.lower():
            depth = depth.astype(np.float32)
            depth = np.dot(depth, [65535.0, 256.0, 1.0])  # decoder to normalized depth
            depth = depth / (256 * 256 * 256 - 1)  # normalized to (0, 1)
            depth = 1. / (depth + 1e-6)
        elif 'airsim' in filename.lower():
            # import cv2
            # # depth = np.clip(depth, 1., 1000., )
            # # depth = 1000. / depth
            # mask = depth > 50000
            # # depth = np.clip(depth, 1., 20000., )
            # # depth = depth.astype(np.float32)
            # depth = 50000. / (depth + 1e-6) * 2
            # depth[mask] = 0
            # depth = np.clip(depth, 0, 255).astype(np.uint8)
            # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            # depth = cv2.morphologyEx(depth, cv2.MORPH_DILATE, kernel)
            depth = depth / 100.

        else:
            raise NotImplementedError('Currently only support loading depth from SELMA dataset.')

        if self.to_3channel:
            depth = np.repeat(depth[:, :, None], 3, axis=-1)
            assert np.ndim(depth) == 3, f'Shape of disparity map must be 3 dim, i.e., (h, w, 3), ' \
                                       f'but got {np.ndim(depth)} dim'
        else:
            depth = depth[:, :, None]
        if self.to_float32:
            depth = depth.astype(np.float32)

        results['depth'] = depth

        if self.post_processing is not None:
            results['depth_postp'] = self._post_processing(results)
        else:
            results['depth_postp'] = depth

        return results

    def _post_processing(self, results: dict):
        """Private function to post-processing disparity map.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains post-processed disparity map ('disp_postp').
        """
        thr_h = self.post_processing['thr_h']
        thr_l = self.post_processing['thr_l']

        depth_postp = results['depth']
        depth_postp = np.clip(depth_postp, thr_l, thr_h, )
        depth_postp = np.clip((depth_postp / np.max(depth_postp) * 255), 0, 255).astype(np.uint8)
        if self.to_float32:
            depth_postp = depth_postp.astype(np.float32)

        return depth_postp

    def __repr__(self) -> str:
        repr_str = (f'{self.__class__.__name__}('
                    f'ignore_empty={self.ignore_empty}, '
                    f'to_float32={self.to_float32}, '
                    f"to_3channel={self.to_3channel}, "
                    f'inv_depth={self.inv_depth}, '
                    f"post_processing={self.post_processing}, "
                    f"imdecode_backend='{self.imdecode_backend}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str


@TRANSFORMS.register_module()
class Disp2ColorImg(BaseTransform):
    """Simply see disparity map as color image"""

    def transform(self, results: dict) -> Optional[dict]:
        if results.get('disp_postp', None) is not None:
            disp = results['disp_postp']
        else:
            disp = results['disp']

        if disp.shape[-1] == 1:
            disp = np.repeat(disp, 3, axis=-1)
        results['img'] = disp

        return results
