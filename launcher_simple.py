import os
from pathlib import Path

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSizePolicy, QLabel, QFileDialog,
                             QTabWidget, QLineEdit, QCheckBox)
import numpy as np
from skimage import io
import napari
import pandas as pd

import utils
from napari_view_simple import launch_viewers
from train import train_unet


class Loader(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.master = parent
        self.opath = ""
        self.modpath = ""
        self.btn1 = QPushButton('open', self)
        self.btn1.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn1.clicked.connect(self.show_dialog_o)
        self.btn2 = QPushButton('open', self)
        self.btn2.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn2.clicked.connect(self.show_dialog_mod)

        self.textbox = QLineEdit(self)
        self.textbox.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self.checkBox = QCheckBox("create new dataset?")
        self.checkBox.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self.btn4 = QPushButton('launch napari', self)
        self.btn4.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn4.clicked.connect(self.launch_napari)
        self.btnb = QPushButton('back', self)
        self.btnb.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btnb.clicked.connect(self.back)
        self.lbl = QLabel('original dir', self)
        self.lbl2 = QLabel('mask dir', self)
        self.lbl4 = QLabel('model type (do not use word "train")', self)
        self.build()

    def build(self):
        vbox = QVBoxLayout()
        vbox.addWidget(combine_blocks(self.btn1, self.lbl))
        vbox.addWidget(combine_blocks(self.btn2, self.lbl2))
        vbox.addWidget(combine_blocks(self.textbox, self.lbl4))
        vbox.addWidget(self.checkBox)
        vbox.addWidget(self.btn4)
        vbox.addWidget(self.btnb)

        self.setLayout(vbox)
        self.show()

    def show_dialog_o(self):
        default_path = max(self.opath, self.modpath, os.path.expanduser('~'))
        f_name = QFileDialog.getExistingDirectory(self, 'Open Directory', default_path)
        if f_name:
            self.opath = f_name
            self.lbl.setText(self.opath)

    def show_dialog_mod(self):
        default_path = max(self.opath, self.modpath, os.path.expanduser('~'))
        f_name = QFileDialog.getExistingDirectory(self, 'Open Directory', default_path)
        if f_name:
            self.modpath = f_name
            self.lbl2.setText(self.modpath)

    def back(self):
        self.master.setCurrentIndex(0)

    def launch_napari(self):
        images = utils.load_images(self.opath)
        if self.modpath == "":
            labels = np.zeros_like(images.compute())
            self.modpath = os.path.join(os.path.dirname(self.opath), self.textbox.text())
            os.makedirs(self.modpath, exist_ok=True)
            filenames = [fn.name for fn in sorted(list(Path(self.opath).glob('./*png')))]
            for i in range(len(labels)):
                io.imsave(os.path.join(self.modpath, str(i).zfill(4) + '.png'), labels[i])
        else:
            labels = utils.load_saved_masks(self.modpath)
        try:
            labels_raw = utils.load_raw_masks(self.modpath + '_raw')
        except:
            labels_raw = None
        view1 = launch_viewers(images, labels, labels_raw, self.modpath, self.textbox.text(), self.checkBox.isChecked())
        global view_l
        view_l.close()
        return view1


class Trainer(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.master = parent
        self.opath = ""
        self.labelpath = ""
        self.btn1 = QPushButton('open', self)
        self.btn1.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn1.clicked.connect(self.show_dialog_o)
        self.btn2 = QPushButton('open', self)
        self.btn2.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn2.clicked.connect(self.show_dialog_label)

        self.btn4 = QPushButton('start training', self)
        self.btn4.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn4.clicked.connect(self.trainer)
        self.btnb = QPushButton('back', self)
        self.btnb.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btnb.clicked.connect(self.back)
        self.lbl = QLabel('original dir', self)
        self.lbl2 = QLabel('label dir', self)
        self.build()

    def build(self):
        vbox = QVBoxLayout()
        vbox.addWidget(combine_blocks(self.btn1, self.lbl))
        vbox.addWidget(combine_blocks(self.btn2, self.lbl2))
        vbox.addWidget(self.btn4)
        vbox.addWidget(self.btnb)

        self.setLayout(vbox)
        self.show()

    def show_dialog_o(self):
        default_path = max(self.opath, self.labelpath, os.path.expanduser('~'))
        f_name = QFileDialog.getExistingDirectory(self, 'Open Directory', default_path)
        if f_name:
            self.opath = f_name
            self.lbl.setText(self.opath)

    def show_dialog_label(self):
        default_path = max(self.opath, self.labelpath, os.path.expanduser('~'))
        f_name = QFileDialog.getExistingDirectory(self, 'Open Directory', default_path)
        if f_name:
            self.labelpath = f_name
            self.lbl2.setText(self.labelpath)

    def back(self):
        self.master.setCurrentIndex(0)

    def get_newest_csv(self):
        csvs = sorted(list(Path(self.labelpath).glob('./*csv')))
        csv = pd.read_csv(str(csvs[-1]), index_col=0)
        return csv

    def trainer(self):
        ori_imgs, ori_filenames = utils.load_X_gray(self.opath)
        label_imgs, label_filenames = utils.load_Y_gray(self.labelpath, normalize=False)
        train_csv = self.get_newest_csv(self.labelpath)
        train_ori_imgs, train_label_imgs = utils.select_train_data(
            dataframe=train_csv,
            ori_imgs=ori_imgs,
            label_imgs=label_imgs,
            ori_filenames=ori_filenames,
        )
        train_unet(
            X_train=train_ori_imgs,
            Y_train=train_label_imgs,
            csv_path="train_log.csv",
            model_path="demo.hdf5",
            input_shape=(512, 512, 1),
            num_classes=1
        )


class Predicter(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.master = parent
        self.opath = ""
        self.labelpath = ""
        self.btn1 = QPushButton('open', self)
        self.btn1.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn1.clicked.connect(self.show_dialog_o)
        self.btn2 = QPushButton('open', self)
        self.btn2.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn2.clicked.connect(self.show_dialog_label)

        self.btn4 = QPushButton('start training', self)
        self.btn4.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn4.clicked.connect(self.trainer)
        self.btnb = QPushButton('back', self)
        self.btnb.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btnb.clicked.connect(self.back)
        self.lbl = QLabel('original dir', self)
        self.lbl2 = QLabel('label dir', self)
        self.build()


class Entrance(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.master = parent
        self.btn1 = QPushButton('Loader', self)
        self.btn1.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn1.clicked.connect(self.move_to_predictions_loader)
        self.btn2 = QPushButton('Loader', self)
        self.btn2.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn2.clicked.connect(self.move_to_modifications_loader)
        self.btn3 = QPushButton('Trainer', self)
        self.btn3.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn3.clicked.connect(self.move_to_trainer)
        self.build()

    def build(self):
        vbox = QVBoxLayout()
        vbox.addWidget(self.btn1)
        vbox.addWidget(self.btn2)
        vbox.addWidget(self.btn3)

        self.setLayout(vbox)
        self.show()

    def move_to_predictions_loader(self):
        self.master.setCurrentIndex(1)

    def move_to_modifications_loader(self):
        self.master.setCurrentIndex(2)

    def move_to_trainer(self):
        self.master.setCurrentIndex(3)


class App(QTabWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("napari launcher")
        self.tab1 = Entrance(self)
        self.tab2 = Loader(self)
        self.tab3 = Loader(self)
        self.tab4 = Trainer(self)

        # add to tab page
        self.addTab(self.tab1, "Entrance")
        self.addTab(self.tab2, "PredictionsLoader")
        self.addTab(self.tab3, "Loader")
        self.addTab(self.tab4, "Trainer")

        self.setStyleSheet("QTabWidget::pane { border: 0; }")
        self.tabBar().hide()
        self.resize(500, 400)


def combine_blocks(block1, block2):
    temp_widget = QWidget()
    temp_layout = QHBoxLayout()
    temp_layout.addWidget(block1)
    temp_layout.addWidget(block2)
    temp_widget.setLayout(temp_layout)
    return temp_widget


if __name__ == '__main__':
    with napari.gui_qt():
        view_l = napari.Viewer()
        launcher = App()
        view_l.window.add_dock_widget(launcher, area='right')
