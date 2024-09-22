# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import shutil
import tempfile
from collections import defaultdict
from glob import glob
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
import trackeval
from mmengine.dist import (all_gather_object, barrier, broadcast,
                           broadcast_object_list, get_dist_info,
                           is_main_process)
from mmengine.logging import MMLogger

from mmtrack.registry import METRICS, TASK_UTILS
from .base_video_metrics import BaseVideoMetric
from ..functional.kitti_2d_box import Kitti2DBox_MOT


def get_tmpdir() -> str:
    """return the same tmpdir for all processes."""
    rank, world_size = get_dist_info()
    MAX_LEN = 512
    # 32 is whitespace
    dir_tensor = torch.full((MAX_LEN, ), 32, dtype=torch.uint8)
    if rank == 0:
        tmpdir = tempfile.mkdtemp()
        tmpdir = torch.tensor(bytearray(tmpdir.encode()), dtype=torch.uint8)
        dir_tensor[:len(tmpdir)] = tmpdir
    broadcast(dir_tensor, 0)
    tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    return tmpdir


@METRICS.register_module()
class MOTKittiMetrics(BaseVideoMetric):
    """Evaluation metrics for MOT Kitti.

    Args:
        metric (str | list[str]): Metrics to be evaluated. Options are
            'HOTA', 'CLEAR', 'Identity'.
            Defaults to ['HOTA', 'CLEAR', 'Identity'].
        outfile_prefix (str, optional): Path to save the formatted results.
            Defaults to None.
        track_iou_thr (float): IoU threshold for tracking evaluation.
            Defaults to 0.5.
        benchmark (str): Benchmark to be evaluated. Defaults to 'KITTI'.
        format_only (bool): If True, only formatting the results to the
            official format and not performing evaluation. Defaults to False.
        postprocess_tracklet_cfg (List[dict], optional): configs for tracklets
            postprocessing methods. `AppearanceFreeLink` and
            `InterpolateTracklets` are supported. Defaults to [].
            - AppearanceFreeLink:
                - checkpoint (str): Checkpoint path.
                - temporal_threshold (tuple, optional): The temporal constraint
                    for tracklets association. Defaults to (0, 30).
                - spatial_threshold (int, optional): The spatial constraint for
                    tracklets association. Defaults to 75.
                - confidence_threshold (float, optional): The minimum
                    confidence threshold for tracklets association.
                    Defaults to 0.95.
            - InterpolateTracklets:
                - min_num_frames (int, optional): The minimum length of a
                    track that will be interpolated. Defaults to 5.
                - max_num_frames (int, optional): The maximum disconnected
                    length in a track. Defaults to 20.
                - use_gsi (bool, optional): Whether to use the GSI (Gaussian-
                    smoothed interpolation) method. Defaults to False.
                - smooth_tau (int, optional): smoothing parameter in GSI.
                    Defaults to 10.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Default: None
    Returns:
    """
    TRACKER = 'default-tracker'
    allowed_metrics = ['HOTA', 'CLEAR', 'Identity']
    allowed_benchmarks = ['KITTI']
    default_prefix: Optional[str] = 'motkitti-metric'
    class_name_to_class_id = {'car': 1, 'van': 2, 'truck': 3, 'pedestrian': 4, 'person': 5,  # person sitting
                              'cyclist': 6, 'tram': 7, 'misc': 8, 'dontcare': 9}

    def __init__(self,
                 metric: Union[str, List[str]] = ['HOTA', 'CLEAR', 'Identity'],
                 outfile_prefix: Optional[str] = None,
                 classes_eval: List[str] = ['car', 'pedestrian'],
                 track_iou_thr: float = 0.5,
                 benchmark: str = 'KITTI',
                 cls_id2name: bool = True,
                 format_only: bool = False,
                 postprocess_tracklet_cfg: Optional[List[dict]] = [],
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        if isinstance(metric, list):
            metrics = metric
        elif isinstance(metric, str):
            metrics = [metric]
        else:
            raise TypeError('metric must be a list or a str.')
        for metric in metrics:
            if metric not in self.allowed_metrics:
                raise KeyError(f'metric {metric} is not supported.')
        self.metrics = metrics
        self.classes_eval = list(classes_eval)
        self.format_only = format_only
        self.postprocess_tracklet_cfg = postprocess_tracklet_cfg.copy()
        self.postprocess_tracklet_methods = [
            TASK_UTILS.build(cfg) for cfg in self.postprocess_tracklet_cfg
        ]
        assert benchmark in self.allowed_benchmarks
        self.benchmark = benchmark
        self.track_iou_thr = track_iou_thr
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_dir.name = get_tmpdir()
        self.seq_info = defaultdict(
            lambda: dict(seq_length=-1, gt_tracks=[], pred_tracks=[]))
        self.gt_dir = self._get_gt_dir()
        self.pred_dir = self._get_pred_dir(outfile_prefix)
        self.seqmap = osp.join(self.pred_dir, 'videoseq.txt')
        with open(self.seqmap, 'w') as f:
            f.write('name\n')
        self.cls_id2name = cls_id2name

    def __del__(self):
        # To avoid tmpdir being cleaned up too early, because in multiple
        # consecutive ValLoops, the value of `self.tmp_dir.name` is unchanged,
        # and calling `tmp_dir.cleanup()` in compute_metrics will cause errors.
        self.tmp_dir.cleanup()

    def _get_pred_dir(self, outfile_prefix):
        """Get directory to save the prediction results."""
        logger: MMLogger = MMLogger.get_current_instance()

        if outfile_prefix is None:
            outfile_prefix = self.tmp_dir.name
        else:
            if osp.exists(outfile_prefix) and is_main_process():
                logger.info('remove previous results.')
                shutil.rmtree(outfile_prefix)
        pred_dir = osp.join(outfile_prefix, self.TRACKER)
        os.makedirs(pred_dir, exist_ok=True)
        return pred_dir

    def _get_gt_dir(self):
        """Get directory to save the gt files."""
        output_dir = osp.join(self.tmp_dir.name, 'gt')
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of data samples that
                contain annotations and predictions.
        """
        for data_sample in data_samples:
            # load basic info
            frame_id = data_sample['frame_id']
            video_length = data_sample['video_length']
            video = data_sample['img_path'].split(os.sep)[-3]
            if self.seq_info[video]['seq_length'] == -1:
                self.seq_info[video]['seq_length'] = video_length

            if 'instances' in data_sample:
                gt_instances = data_sample['instances']
                gt_tracks = [
                    np.array([
                        frame_id,  # frame
                        gt_instances[i]['instance_id'],  # track_id
                        gt_instances[i]['category_id'],  # type
                        gt_instances[i]['truncated'],  # truncated
                        gt_instances[i]['occluded'],  # occluded
                        gt_instances[i]['alpha'],  # alpha
                        gt_instances[i]['bbox'][0],  # bbox l
                        gt_instances[i]['bbox'][1],  # bbox t
                        gt_instances[i]['bbox'][2],  # bbox r
                        gt_instances[i]['bbox'][3],  # bbox b
                        gt_instances[i]['dim'][0],  # dimension height
                        gt_instances[i]['dim'][1],  # dimension width
                        gt_instances[i]['dim'][2],  # dimension length
                        gt_instances[i]['location'][0],  # location x
                        gt_instances[i]['location'][1],  # location y
                        gt_instances[i]['location'][2],  # location z
                        gt_instances[i]['rotation_y'],  # rotation_y
                        gt_instances[i]['mot_conf'],
                        gt_instances[i]['visibility']
                    ]) for i in range(len(gt_instances))
                ]
                self.seq_info[video]['gt_tracks'].extend(gt_tracks)

            # load predictions
            assert 'pred_track_instances' in data_sample
            pred_instances = data_sample['pred_track_instances']
            assert 'cat2label' in data_sample
            cat2label = data_sample['cat2label']
            label2cat = {v: k for k, v in cat2label.items()}
            pred_tracks = [
                np.array([
                    frame_id,  # frame
                    pred_instances['instances_id'][i].cpu(),  # track_id
                    label2cat[pred_instances['labels'][i].item()],  # type i.e., category_id
                    -1,  # truncated
                    -1,  # occluded
                    -1,  # alpha
                    pred_instances['bboxes'][i][0].cpu(),  # bbox l
                    pred_instances['bboxes'][i][1].cpu(),  # bbox t
                    pred_instances['bboxes'][i][2].cpu(),  # bbox r
                    pred_instances['bboxes'][i][3].cpu(),  # bbox b
                    -1,  # dimension height
                    -1,  # dimension width
                    -1,  # dimension length
                    -1,  # location x
                    -1,  # location y
                    -1,  # location z
                    -1,  # rotation_y
                    pred_instances['scores'][i].cpu()
                ]) for i in range(len(pred_instances['instances_id']))
            ]
            self.seq_info[video]['pred_tracks'].extend(pred_tracks)

            if frame_id == video_length - 1:
                # postprocessing
                if self.postprocess_tracklet_cfg:
                    info = self.seq_info[video]
                    pred_tracks = np.array(info['pred_tracks'])
                    for postprocess_tracklet_methods in \
                            self.postprocess_tracklet_methods:
                        pred_tracks = postprocess_tracklet_methods\
                            .forward(pred_tracks)
                    info['pred_tracks'] = pred_tracks
                self._save_one_video_gts_preds(video)
                break

    def _save_one_video_gts_preds(self, seq: str) -> None:
        """Save the gt and prediction results."""
        info = self.seq_info[seq]
        # save predictions
        pred_file = osp.join(self.pred_dir, seq + '.txt')

        pred_tracks = np.array(info['pred_tracks'])

        # load gts
        if self.cls_id2name:
            id_map = {v: k for k, v in self.class_name_to_class_id.items()}
        else:
            id_map = {v: v for k, v in self.class_name_to_class_id.items()}

        with open(pred_file, 'wt') as f:
            for tracks in pred_tracks:
                line = f'{int(tracks[0])},{int(tracks[1])},{id_map[int(tracks[2])]},{int(tracks[3])},' \
                       f'{int(tracks[4])},{tracks[5]:.6f},{tracks[6]:.6f},{tracks[7]:.6f},{tracks[8]:.6f},' \
                       f'{tracks[9]:.6f},{tracks[10]:.6f},{tracks[11]:.6f},{tracks[12]:.6f},{tracks[13]:.6f},' \
                       f'{tracks[14]:.6f},{tracks[15]:.6f},{tracks[16]:.6f},{tracks[17]:.6f}\n'
                f.writelines(line)

        info['pred_tracks'] = []
        # save gts
        if info['gt_tracks']:
            gt_file = osp.join(self.gt_dir, seq + '.txt')
            with open(gt_file, 'wt') as f:
                for tracks in info['gt_tracks']:
                    line = f'{int(tracks[0])},{int(tracks[1])},{id_map[int(tracks[2])]},{int(tracks[3])},' \
                           f'{int(tracks[4])},{tracks[5]:.6f},{tracks[6]:.6f},{tracks[7]:.6f},{tracks[8]:.6f},' \
                           f'{tracks[9]:.6f},{tracks[10]:.6f},{tracks[11]:.6f},{tracks[12]:.6f},{tracks[13]:.6f},' \
                           f'{tracks[14]:.6f},{tracks[15]:.6f},{tracks[16]:.6f}\n'
                    f.writelines(line)
            info['gt_tracks'].clear()
        # save seq info
        with open(self.seqmap, 'a') as f:
            f.write(seq + '\n')
            f.close()

    def compute_metrics(self, results: list = None) -> dict:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.
                Defaults to None.

        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        # NOTICE: don't access `self.results` from the method.
        eval_results = dict()

        if self.format_only:
            return eval_results

        eval_config = trackeval.Evaluator.get_default_eval_config()

        # need to split out the tracker name
        # caused by the implementation of TrackEval
        pred_dir_tmp = self.pred_dir.rsplit(osp.sep, 1)[0]
        dataset_config = self.get_dataset_cfg(self.gt_dir, pred_dir_tmp, self.classes_eval)

        evaluator = trackeval.Evaluator(eval_config)
        dataset = [Kitti2DBox_MOT(dataset_config)]
        metrics = [
            getattr(trackeval.metrics,
                    metric)(dict(METRICS=[metric], THRESHOLD=0.5))
            for metric in self.metrics
        ]
        output_res, _ = evaluator.evaluate(dataset, metrics)
        for cls in dataset_config['CLASSES_TO_EVAL']:
            _output_res = output_res['Kitti2DBox_MOT'][
                self.TRACKER]['COMBINED_SEQ'][cls]

            self.collect_results(eval_results, _output_res, logger, prefix=f'_{cls}')

        # clean all txt file
        for txt_name in glob(osp.join(self.tmp_dir.name, '*.txt')):
            os.remove(txt_name)
        return eval_results

    def collect_results(self, eval_results, output_res, logger, prefix=''):
        if 'HOTA' in self.metrics:
            logger.info('Evaluating HOTA Metrics...')
            eval_results[f'HOTA{prefix}'] = np.average(output_res['HOTA']['HOTA'])
            eval_results[f'AssA{prefix}'] = np.average(output_res['HOTA']['AssA'])
            eval_results[f'DetA{prefix}'] = np.average(output_res['HOTA']['DetA'])

        if 'CLEAR' in self.metrics:
            logger.info('Evaluating CLEAR Metrics...')
            eval_results[f'MOTA{prefix}'] = np.average(output_res['CLEAR']['MOTA'])
            eval_results[f'MOTP{prefix}'] = np.average(output_res['CLEAR']['MOTP'])
            eval_results[f'IDSW{prefix}'] = np.average(output_res['CLEAR']['IDSW'])
            eval_results[f'TP{prefix}'] = np.average(output_res['CLEAR']['CLR_TP'])
            eval_results[f'FP{prefix}'] = np.average(output_res['CLEAR']['CLR_FP'])
            eval_results[f'FN{prefix}'] = np.average(output_res['CLEAR']['CLR_FN'])
            eval_results[f'Frag{prefix}'] = np.average(output_res['CLEAR']['Frag'])
            eval_results[f'MT{prefix}'] = np.average(output_res['CLEAR']['MT'])
            eval_results[f'ML{prefix}'] = np.average(output_res['CLEAR']['ML'])

        if 'Identity' in self.metrics:
            logger.info('Evaluating Identity Metrics...')
            eval_results[f'IDF1{prefix}'] = np.average(output_res['Identity']['IDF1'])
            eval_results[f'IDTP{prefix}'] = np.average(output_res['Identity']['IDTP'])
            eval_results[f'IDFN{prefix}'] = np.average(output_res['Identity']['IDFN'])
            eval_results[f'IDFP{prefix}'] = np.average(output_res['Identity']['IDFP'])
            eval_results[f'IDP{prefix}'] = np.average(output_res['Identity']['IDP'])
            eval_results[f'IDR{prefix}'] = np.average(output_res['Identity']['IDR'])

        return eval_results

    def evaluate(self, size: int = None) -> dict:
        """Evaluate the model performance of the whole dataset after processing
        all batches.

        Args:
            size (int): Length of the entire validation dataset.
                Defaults to None.

        Returns:
            dict: Evaluation metrics dict on the val dataset. The keys are the
            names of the metrics, and the values are corresponding results.
        """
        # wait for all processes to complete prediction.
        barrier()

        # gather seq_info and convert the list of dict to a dict.
        # convert self.seq_info to dict first to make it picklable.
        gathered_seq_info = all_gather_object(dict(self.seq_info))
        all_seq_info = dict()
        for _seq_info in gathered_seq_info:
            all_seq_info.update(_seq_info)
        self.seq_info = all_seq_info

        if is_main_process():
            _metrics = self.compute_metrics()  # type: ignore
            # Add prefix to metric names
            if self.prefix:
                _metrics = {
                    '/'.join((self.prefix, k)): v
                    for k, v in _metrics.items()
                }
            metrics = [_metrics]
        else:
            metrics = [None]  # type: ignore

        broadcast_object_list(metrics)

        # reset the results list
        self.results.clear()
        return metrics[0]

    def get_dataset_cfg(self, gt_folder: str, tracker_folder: str, classes_eval: List[str]):
        """Get default configs for trackeval.datasets.MotChallenge2DBox.

        Args:
            gt_folder (str): the name of the GT folder
            tracker_folder (str): the name of the tracker folder

        Returns:
            Dataset Configs for MotChallenge2DBox.
        """
        dataset_config = dict(
            # Location of GT data
            GT_FOLDER=gt_folder,
            # Trackers location
            TRACKERS_FOLDER=tracker_folder,
            # Where to save eval results
            # (if None, same as TRACKERS_FOLDER)
            OUTPUT_FOLDER=None,
            # Use self.TRACKER as the default tracker
            TRACKERS_TO_EVAL=[self.TRACKER],
            # Option values: ['car', 'pedestrian']
            CLASSES_TO_EVAL=classes_eval,
            # Option Values: 'train', 'test'
            SPLIT_TO_EVAL='train',
            # Whether tracker input files are zipped
            INPUT_AS_ZIP=False,
            # Whether to print current config
            PRINT_CONFIG=True,
            # Tracker files are in
            # TRACKER_FOLDER/tracker_name/TRACKER_SUB_FOLDER
            TRACKER_SUB_FOLDER='',
            # Output files are saved in
            # OUTPUT_FOLDER/tracker_name/OUTPUT_SUB_FOLDER
            OUTPUT_SUB_FOLDER='',
            # Names of trackers to display
            # (if None: TRACKERS_TO_EVAL)
            TRACKER_DISPLAY_NAMES=None,
            # Where seqmaps are found
            # (if None: GT_FOLDER/seqmaps)
            SEQMAP_FOLDER=None,
            # Directly specify seqmap file
            # (if none use seqmap_folder/benchmark-split_to_eval)
            SEQMAP_FILE=self.seqmap,
            # If not None, specify sequences to eval
            # and their number of timesteps
            SEQ_INFO={
                seq: info['seq_length']
                for seq, info in self.seq_info.items()
            },
            # '{gt_folder}/{seq}.txt'
            GT_LOC_FORMAT='{gt_folder}/{seq}.txt',
            CLASS_NAME_TO_CLASS_ID = self.class_name_to_class_id
        )

        return dataset_config
