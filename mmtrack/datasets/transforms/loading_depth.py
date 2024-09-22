# loading depth related data as input
import numpy as np

from mmtrack.registry import TRANSFORMS


@TRANSFORMS.register_module()
class Load