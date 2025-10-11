from .attention import AttentionAccess
from .base import ModelAccess, make_tanh_classifier_head
from .logreg import LogRegAccess
from .self_attention import SelfAttentionAccess
from .temporal import SimpleTemporalPoolingClassifier, TemporalAccess  # kept for completeness

__all__ = [
    "ModelAccess",
    "make_tanh_classifier_head",
    "LogRegAccess",
    "AttentionAccess",
    "SelfAttentionAccess",
    "TemporalAccess",
    "SimpleTemporalPoolingClassifier",
]
