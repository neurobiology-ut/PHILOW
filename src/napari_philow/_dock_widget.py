from napari_plugin_engine import napari_hook_implementation

from src.napari_philow._annotation import AnnotationMode
from src.napari_philow._prediction import Predicter
from src.napari_philow._train import Trainer


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return [AnnotationMode, Trainer, Predicter]
