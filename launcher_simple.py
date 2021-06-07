import os
from pathlib import Path

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSizePolicy, QLabel, QFileDialog,
                             QTabWidget, QLineEdit, QCheckBox)
import numpy as np
from skimage import io
import napari

import utils
from napari_view_simple import launch_viewers


class PredictionsLoader(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.master = parent
        self.opath = ""
        self.erpath = ""
        self.mipath = ""
        self.btn1 = QPushButton('open', self)
        self.btn1.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn1.clicked.connect(self.show_dialog_o)

        self.btn2 = QPushButton('open', self)
        self.btn2.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn2.clicked.connect(self.show_dialog_m)

        self.btn3 = QPushButton('open', self)
        self.btn3.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn3.clicked.connect(self.show_dialog_e)

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
        self.lbl2 = QLabel('mitochondria mask dir', self)
        self.lbl3 = QLabel('er mask dir', self)
        self.lbl4 = QLabel('model type (do not use word "train")', self)

        self.build()

    def build(self):
        vbox = QVBoxLayout()
        vbox.addWidget(combine_blocks(self.btn1, self.lbl))
        vbox.addWidget(combine_blocks(self.btn2, self.lbl2))
        vbox.addWidget(combine_blocks(self.btn3, self.lbl3))

        vbox.addWidget(combine_blocks(self.textbox, self.lbl4))

        vbox.addWidget(self.checkBox)

        vbox.addWidget(self.btn4)

        vbox.addWidget(self.btnb)

        self.setLayout(vbox)
        self.show()

    def show_dialog_o(self):
        default_path = max(self.opath, self.erpath, self.mipath, os.path.expanduser('~'))
        f_name = QFileDialog.getExistingDirectory(self, 'Open Directory', default_path)
        if f_name:
            self.opath = f_name
            self.lbl.setText(self.opath)

    def show_dialog_m(self):
        default_path = max(self.opath, self.erpath, self.mipath, os.path.expanduser('~'))
        f_name = QFileDialog.getExistingDirectory(self, 'Open Directory', default_path)
        if f_name:
            self.mipath = f_name
            self.lbl2.setText(self.mipath)

    def show_dialog_e(self):
        default_path = max(self.opath, self.erpath, self.mipath, os.path.expanduser('~'))
        f_name = QFileDialog.getExistingDirectory(self, 'Open Directory', default_path)
        if f_name:
            self.erpath = f_name
            self.lbl3.setText(self.erpath)

    def launch_napari(self):
        images = utils.load_images(self.opath)
        if self.mipath == "":
            labels = np.zeros_like(images)
            view1 = launch_viewers(images, labels, os.path.dirname(os.path.dirname(self.opath)), self.textbox.text(),
                                   self.checkBox.isChecked())
        else:
            labels = utils.load_predicted_masks(self.mipath, self.erpath)
            view1 = launch_viewers(images, labels, os.path.dirname(os.path.dirname(self.mipath)), self.textbox.text(),
                                   self.checkBox.isChecked())
        global view_l
        view_l.close()
        return view1

    def back(self):
        self.master.setCurrentIndex(0)


class ModificationsLoader(QWidget):
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


class InitialLoader(QWidget):
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
        self.lbl2 = QLabel('label dir', self)
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

    def launch_napari(self):
        images = utils.load_images(self.opath)
        labels = np.zeros_like(images)
        utils.save_masks(labels, self.labelpath)
        view1 = launch_viewers(images, labels, self.labelpath, self.textbox.text(), self.checkBox.isChecked())
        global view_l
        view_l.close()
        return view1


class Entrance(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.master = parent
        self.btn1 = QPushButton('PredictionsLoader', self)
        self.btn1.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn1.clicked.connect(self.move_to_predictions_loader)
        self.btn2 = QPushButton('ModificationsLoader', self)
        self.btn2.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn2.clicked.connect(self.move_to_modifications_loader)
        self.btn3 = QPushButton('InitialLoader', self)
        self.btn3.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn3.clicked.connect(self.move_to_initial_loader)
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

    def move_to_initial_loader(self):
        self.master.setCurrentIndex(3)


class App(QTabWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("napari launcher")
        self.tab1 = Entrance(self)
        self.tab2 = PredictionsLoader(self)
        self.tab3 = ModificationsLoader(self)
        self.tab4 = InitialLoader(self)

        # add to tab page
        self.addTab(self.tab1, "Entrance")
        self.addTab(self.tab2, "PredictionsLoader")
        self.addTab(self.tab3, "ModificationsLoader")
        self.addTab(self.tab4, "InitialLoader")

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


