# Copyright (c) OpenMMLab. All rights reserved.
from .fpn import FPN
from .gap_neck import GlobalAveragePooling
from .posewarper_neck import PoseWarperNeck
from .tcformer_mta_neck import MTA
from .channel_mapper import ChannelMapper

__all__ = ['GlobalAveragePooling', 'PoseWarperNeck', 'FPN', 'MTA']
