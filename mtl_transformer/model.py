import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from models.enhanced_feature_encoder import EnhancedFeatureEncoder

# from .config import Config # Config is used by the models, but not directly in this file anymore

import logging
logger = logging.getLogger(__name__)

# All class definitions have been moved to their respective files in the 'models' directory.
# The UBTModel alias is also imported from universal_behavioral_transformer.py

__all__ = [
    'EnhancedFeatureEncoder',
]
