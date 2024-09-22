from typing import Any, Mapping, Sequence
from mmengine.dataset import COLLATE_FUNCTIONS, pseudo_collate

@COLLATE_FUNCTIONS.register_module()
def multi_dataset_collate(data_batch: Sequence) -> Any:
    """Convert list of data sampled from dataset into a batch of data, of which
    type consistent with the type of each data_itement in ``data_batch``.

    The default behavior of dataloader is to merge a list of samples to form
    a mini-batch of Tensor(s). However, in MMEngine, ``pseudo_collate``
    will not stack tensors to batch tensors, and convert int, float, ndarray to
    tensors.

    This code is referenced from:
    `Pytorch default_collate <https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py>`_.  # noqa: E501
    Args:
        data_batch (Sequence): Batch of data from dataloader.

    Returns:
        Any: Transversed Data in the same format as the data_itement of
        ``data_batch``.
    """
    # Group data
    src_data = []
    tar_data = []
    for data_item in data_batch:
        source_disp = data_item['inputs']['source_disp']
        if source_disp is not None:
            src_data.append({'inputs': dict(source_disp=source_disp),
                             'data_samples': dict(source_disp=data_item['data_samples']['source_disp'])})

        target_disp = data_item['inputs']['target_disp']
        target_sup_det = data_item['inputs']['target_sup_det']
        if target_disp is not None and target_sup_det is not None:
            tar_data.append({'inputs':
                                 dict(target_disp=target_disp,
                                      target_sup_det=target_sup_det),
                             'data_samples': dict(target_disp=data_item['data_samples']['target_disp'],
                                                  target_sup_det=data_item['data_samples']['target_sup_det'])})

    if len(src_data) > 0:
        src_data = pseudo_collate(src_data)
    if len(tar_data) > 0:
        tar_data = pseudo_collate(tar_data)

    return {'src': src_data, 'tar': tar_data}

    # data_item = data_batch[0]
    # data_item_type = type(data_item)
    # if isinstance(data_item, (str, bytes)):
    #     return data_batch
    # elif isinstance(data_item, tuple) and hasattr(data_item, '_fields'):
    #     # named tuple
    #     return data_item_type(*(multi_dataset_collate(samples)
    #                             for samples in zip(*data_batch)))
    # elif isinstance(data_item, Sequence):
    #     # check to make sure that the data_itements in batch have
    #     # consistent size
    #     it = iter(data_batch)
    #     data_item_size = len(next(it))
    #     if not all(len(data_item) == data_item_size for data_item in it):
    #         raise RuntimeError(
    #             'each data_itement in list of batch should be of equal size')
    #     transposed = list(zip(*data_batch))
    #
    #     if isinstance(data_item, tuple):
    #         return [multi_dataset_collate(samples)
    #                 for samples in transposed]  # Compat with Pytorch.
    #     else:
    #         try:
    #             return data_item_type(
    #                 [multi_dataset_collate(samples) for samples in transposed])
    #         except TypeError:
    #             # The sequence type may not support `__init__(iterable)`
    #             # (e.g., `range`).
    #             return [multi_dataset_collate(samples) for samples in transposed]
    # elif isinstance(data_item, Mapping):
    #     return data_item_type({
    #         key: multi_dataset_collate([d[key] for d in data_batch])
    #         for key in data_item
    #     })
    # else:
    #     return data_batch