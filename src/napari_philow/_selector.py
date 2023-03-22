import os
from pathlib import Path

import dask.array as da
import numpy as np
from dask_image import ndmeasure
from magicgui import magicgui
from qtpy.QtWidgets import QWidget, QVBoxLayout, QPushButton, QSizePolicy, QLabel, QFileDialog, QCheckBox
from skimage import io

from napari_philow._utils import combine_blocks, load_images, load_saved_masks, save_masks


class Selector(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self._viewer = napari_viewer
        self.opath = ""
        self.modpath = ""
        self.select_path = ""
        self.btn1 = QPushButton('open', self)
        self.btn1.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn1.clicked.connect(self.show_dialog_o)
        self.btn2 = QPushButton('open', self)
        self.btn2.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn2.clicked.connect(self.show_dialog_mod)
        self.btn3 = QPushButton('open', self)
        self.btn3.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn3.clicked.connect(self.show_dialog_select)

        self.btn4 = QPushButton('launch napari', self)
        self.btn4.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn4.clicked.connect(self.launch_napari_selector)
        self.lbl = QLabel('original images dir', self)
        self.lbl2 = QLabel('original label dir', self)
        self.lbl3 = QLabel('selected label dir', self)
        self.build()

    def build(self):
        vbox = QVBoxLayout()
        vbox.addWidget(combine_blocks(self.btn1, self.lbl))
        vbox.addWidget(combine_blocks(self.btn2, self.lbl2))
        vbox.addWidget(combine_blocks(self.btn3, self.lbl3))
        vbox.addWidget(self.btn4)

        self.setLayout(vbox)
        self.show()

    def show_dialog_o(self):
        default_path = max(self.opath, self.modpath, self.select_path, os.path.expanduser('~'))
        f_name = QFileDialog.getExistingDirectory(self, 'Open Directory', default_path)
        if f_name:
            self.opath = f_name
            self.lbl.setText(self.opath)

    def show_dialog_mod(self):
        default_path = max(self.opath, self.modpath, self.select_path, os.path.expanduser('~'))
        f_name = QFileDialog.getExistingDirectory(self, 'Open Directory', default_path)
        if f_name:
            self.modpath = f_name
            self.lbl2.setText(self.modpath)

    def show_dialog_select(self):
        default_path = max(self.modpath, self.select_path, os.path.expanduser('~'))
        f_name = QFileDialog.getExistingDirectory(self, 'Open Directory', default_path)
        if f_name:
            self.select_path = f_name
            self.lbl3.setText(self.select_path)

    def launch_napari_selector(self):
        images = load_images(self.opath)
        labels, _ = load_saved_masks(self.modpath)
        if self.select_path == "":
            self.select_path = os.path.join(self.modpath + '_selected')
            os.makedirs(self.select_path, exist_ok=True)
        if len(os.listdir(self.select_path)) == 0:
            selects = da.zeros_like(images)
            filenames = [fn.name for fn in sorted(list(Path(self.opath).glob('./*png')))]
            for i in range(len(selects)):
                io.imsave(os.path.join(self.select_path, str(i).zfill(4) + '.png'), selects[i])
        else:
            selects, _ = load_saved_masks(self.select_path)

        self._viewer.window.remove_dock_widget(self)
        self.launch_selector(images, labels, selects)

    def launch_selector(self, original, label, select):
        global layer1
        global layer2
        global images_original
        global base_label
        global only_label

        images_original = original
        base_label = label
        only_label = select

        # remove duplicate area from base_label
        base_label = np.where(only_label > 0, 0, base_label)

        try:
            del layer1
            del layer2
        except NameError:
            pass
        image_layer = self._viewer.add_image(images_original, contrast_limits=[0, 255])
        self._viewer.add_labels(base_label, name='base', color={1: 'red', 2: 'blue', 3: 'green'})
        self._viewer.add_labels(only_label, name="selected_objects", color={1: 'blue', 2: 'green', 3: 'red'})

        # calc label
        labels_imgs, _ = ndmeasure.label((label == 1).astype(int))
        layer1 = self._viewer.layers[1]
        layer2 = self._viewer.layers[2]

        def select_target(layer, event):
            global base_label
            global only_label

            @layer1.bind_key('q', overwrite=True)
            def select_event(event):
                global base_label
                global only_label
                q_point = np.round(self._viewer.cursor.position).astype(int)
                print(q_point)
                print(labels_imgs.shape)
                target_label = labels_imgs[q_point[0]][q_point[1]][q_point[2]]
                print("select_label : ", target_label, "select_point : ", q_point)
                if target_label:
                    only_label = np.where(labels_imgs == target_label, 1, only_label)
                    base_label = np.where(labels_imgs == target_label, 0, base_label)
                    self._viewerlayers[1].data = base_label
                    self._viewer.layers[2].data = only_label
                    print('Selected!')
                return True

        @layer1.mouse_drag_callbacks.append
        def main(layer, event):
            select_target(layer, event)

        def deselect_target(layer, event):
            global base_label
            global only_label

            @layer2.bind_key('w', overwrite=True)
            def deselect_event(event):
                global base_label
                global only_label
                w_point = np.round(self._viewer.cursor.position.astype(int))
                print("select_label : ", labels_imgs[w_point[0], w_point[1], w_point[2]], "select_point : ", w_point)
                if labels_imgs[w_point[0], w_point[1], w_point[2]]:
                    target_label = labels_imgs[w_point[0], w_point[1], w_point[2]]
                    base_label = np.where(labels_imgs == target_label, 1, base_label)
                    only_label = np.where(labels_imgs == target_label, 0, only_label)
                    self._viewer.layers[1].data = base_label
                    self._viewer.layers[2].data = only_label
                    print('Deselected!')
                return True

        @layer2.mouse_drag_callbacks.append
        def main(layer, event):
            deselect_target(layer, event)

        @magicgui(dirname={"mode": "d"}, call_button=False)
        def mod_dirpicker(dirname=Path(self.modpath)):
            """Take a filename and do something with it."""
            print("The filename is:", dirname)
            return dirname

        self._viewer.window.add_dock_widget(mod_dirpicker, area='bottom')

        @magicgui(dirname={"mode": "d"}, call_button=False)
        def select_dirpicker(dirname=Path(self.select_path)):
            """Take a filename and do something with it."""
            print("The filename is:", dirname)
            return dirname

        self._viewer.window.add_dock_widget(select_dirpicker, area='bottom')

        @magicgui(call_button="save")
        def saver():
            print("The directory is:", self.select_path)
            return save_masks(layer1.data, self.mod_path), save_masks(layer2.data, self.select_path)

        # gui4 = saver.Gui(show=True)
        # view1.window.add_dock_widget(gui4, area='bottom')
        self._viewer.window.add_dock_widget(saver, area='bottom')
