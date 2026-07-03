from projector_study.models.projectors.linear import LinearProjector
from projector_study.models.projectors.mlp import MLPProjector
from projector_study.models.projectors.qformer import QFormerProjector
from projector_study.models.projectors.resampler import ResamplerProjector
from projector_study.models.projectors.c_abstractor import CAbstractorProjector
from projector_study.models.projectors.pixel_shuffle import PixelShuffleProjector

PROJECTOR_REGISTRY = {
    "linear":        LinearProjector,
    "mlp":           MLPProjector,
    "qformer":       QFormerProjector,
    "resampler":     ResamplerProjector,
    "c_abstractor":  CAbstractorProjector,
    "pixel_shuffle": PixelShuffleProjector,
}

__all__ = [
    "LinearProjector",
    "MLPProjector",
    "QFormerProjector",
    "ResamplerProjector",
    "CAbstractorProjector",
    "PixelShuffleProjector",
    "PROJECTOR_REGISTRY",
]
