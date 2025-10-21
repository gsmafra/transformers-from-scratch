from typing import Dict, Iterable

from . import (
    AttentionAccess,
    LogRegAccess,
    MLPAccess,
    ModelAccess,
    SelfAttentionAccess,
    SelfAttentionQKVAccess,
    SelfAttentionQKVPosAccess,
    TemporalAccess,
)


def build_models(sequence_length: int, n_features: int, *, only: Iterable[str] | None = None) -> Dict[str, ModelAccess]:
    """Construct model access wrappers.

    If `only` is provided, build exactly that subset by name.
    """
    builders = {
        "logreg": lambda: LogRegAccess(sequence_length=sequence_length, n_features=n_features),
        "mlp": lambda: MLPAccess(sequence_length=sequence_length, n_features=n_features),
        "temporal": lambda: TemporalAccess(sequence_length=sequence_length, n_features=n_features),
        "self_attention": lambda: SelfAttentionAccess(n_features=n_features),
        "self_attention_qkv": lambda: SelfAttentionQKVAccess(n_features=n_features),
        "self_attention_qkv_pos": lambda: SelfAttentionQKVPosAccess(n_features=n_features),
        "attention": lambda: AttentionAccess(n_features=n_features),
    }

    names = list(only) if only is not None else list(builders.keys())
    models = {name: builders[name]() for name in names if name in builders}
    return models
