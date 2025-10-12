from typing import Dict

from . import (
    AttentionAccess,
    LogRegAccess,
    ModelAccess,
    SelfAttentionAccess,
    SelfAttentionQKVAccess,
    TemporalAccess,
)


def build_models(sequence_length: int, n_features: int) -> Dict[str, ModelAccess]:
    models = [
        LogRegAccess(sequence_length=sequence_length, n_features=n_features),
        TemporalAccess(sequence_length=sequence_length, n_features=n_features),
        SelfAttentionAccess(n_features=n_features),
        SelfAttentionQKVAccess(n_features=n_features),
        AttentionAccess(n_features=n_features),
    ]
    return {m.name: m for m in models}

