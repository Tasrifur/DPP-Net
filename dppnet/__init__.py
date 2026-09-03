"""DPP-Net: a dynamic part-aware framework for few-shot cross-condition gait recognition."""

__version__ = "1.0.0"

from .model import (DynamicProtoNetV3, PartAwareGaitEncoder, build_model,
                    load_model)

__all__ = ["PartAwareGaitEncoder", "DynamicProtoNetV3", "build_model", "load_model"]
