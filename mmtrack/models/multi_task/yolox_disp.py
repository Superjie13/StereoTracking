from mmengine.logging import print_log, MMLogger

from mmtrack.registry import MODELS
from mmtrack.utils import ConfigType, OptConfigType, OptMultiConfig
from .single_stage import SingleStageDetector_DispCompletion


@MODELS.register_module()
class YOLOX_DISP(SingleStageDetector_DispCompletion):
    r"""Implementation of `YOLOX: Exceeding YOLO Series in 2021
    <https://arxiv.org/abs/2107.08430>`_ plus a disparity completion head

    Args:
        backbone (:obj:`ConfigDict` or dict): The backbone config.
        neck (:obj:`ConfigDict` or dict): The neck config.
        bbox_head (:obj:`ConfigDict` or dict): The bbox head config.
        disparity_head (:obj:`ConfigDict` or dict): The disparity completion head config.
        train_cfg (:obj:`ConfigDict` or dict, optional): The training config
            of YOLOX. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): The testing config
            of YOLOX. Defaults to None.
        data_preprocessor (:obj:`ConfigDict` or dict, optional): Config of
            :class:`DetDataPreprocessor` to process the input data.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 disparity_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            disparity_head=disparity_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

    def init_weights(self):
        if self.init_cfg is not None:
            if self.init_cfg.get('type', None) == 'ColorPretrained':
                from mmengine.runner.checkpoint import _load_checkpoint, load_state_dict
                logger = MMLogger.get_instance('mmengine')

                pretrained = self.init_cfg.get('checkpoint')
                print_log(f"load model from: {pretrained}", logger=logger)
                checkpoint = _load_checkpoint(pretrained, logger=logger, map_location='cpu')
                state_dict = checkpoint['state_dict']

                print_log(f"update stem conv for disparity: `new stem`", logger=logger)

                disp_branch = dict()
                for name, param in state_dict.items():
                    if 'stem' in name:
                        new_name = name.replace('stem', 'disp_stem')
                        disp_branch.update({new_name: param})
                    if 'stage1' in name:
                        new_name = name.replace('stage1', 'disp_stage1')
                        disp_branch.update({new_name: param})
                state_dict.update(disp_branch)
                load_state_dict(self, state_dict, strict=False, logger=logger)

