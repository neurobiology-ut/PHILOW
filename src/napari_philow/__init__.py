
try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"


from ._annotation import AnnotationMode
from ._train import Trainer
from ._prediction import Predicter

__all__ = (
    "AnnotationMode",
    "Trainer",
    "Predicter",
)
