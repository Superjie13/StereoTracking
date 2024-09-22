# Copyright (c) OpenMMLab. All rights reserved.
# from mmdet.engine import YOLOXModeSwitchHook as _YOLOXModeSwitchHook
from mmyolo.engine import YOLOXModeSwitchHook as _YOLOXModeSwitchHook
import copy

from mmengine.model import is_model_wrapper
from mmengine.runner import Runner

from mmtrack.registry import HOOKS


@HOOKS.register_module()
class YOLOXModeSwitchHook_mmyolox(_YOLOXModeSwitchHook):
    """Switch the mode of YOLOX during training.

    This hook turns off the mosaic and mixup data augmentation and switches
    to use L1 loss in bbox_head.

    The difference between this class and the class in mmyolo is that the
    class in mmdet use `model.bbox_head.use_bbox_aux=True` to switch mode, while
    this class will check whether there is a detector module in the model
    firstly, then use `model.detector.bbox_head.use_bbox_aux=True` or
    `model.bbox_head.use_bbox_aux=True` to switch mode.

    Args:
        num_last_epochs (int): The number of latter epochs in the end of the
            training to close the data augmentation and switch to L1 loss.
            Defaults to 15.
    """

    def before_train_epoch(self, runner):
        """Close mosaic and mixup augmentation and switches to use L1 loss."""
        epoch = runner.epoch
        model = runner.model
        if is_model_wrapper(model):
            model = model.module

        if (epoch + 1) == runner.max_epochs - self.num_last_epochs:
            runner.logger.info(f'New Pipeline: {self.new_train_pipeline_cfg}')
            train_dataloader_cfg = copy.deepcopy(runner.cfg.train_dataloader)
            train_dataloader_cfg.dataset.pipeline = self.new_train_pipeline_cfg
            # Note: Why rebuild the dataset?
            # When build_dataloader will make a deep copy of the dataset,
            # it will lead to potential risks, such as the global instance
            # object FileClient data is disordered.
            # This problem needs to be solved in the future.
            new_train_dataloader = Runner.build_dataloader(
                train_dataloader_cfg)
            runner.train_loop.dataloader = new_train_dataloader

            runner.logger.info('recreate the dataloader!')
            runner.logger.info('Add additional bbox reg loss now!')

            if hasattr(model, 'detector'):
                model.detector.bbox_head.use_bbox_aux = True
            else:
                model.bbox_head.use_bbox_aux = True