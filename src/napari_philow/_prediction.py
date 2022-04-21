import os
import shutil
from pathlib import Path

import pandas as pd
from napari_tools_menu import register_dock_widget
from qtpy.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QSizePolicy, QLabel, QFileDialog,
                            QCheckBox)

from napari_philow._models import get_nested_unet
from napari_philow._predict import predict_3ax, predict_1ax
from napari_philow._utils import combine_blocks, load_X_gray


@register_dock_widget(menu="PHILOW > Prediction mode")
class Predicter(QWidget):
    def __init__(self):
        super().__init__()
        self.opath = ""
        self.labelpath = ""
        self.modelpath = ""
        self.outpath = ""
        self.btn1 = QPushButton('open', self)
        self.btn1.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn1.clicked.connect(self.show_dialog_o)
        self.btn2 = QPushButton('open', self)
        self.btn2.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn2.clicked.connect(self.show_dialog_label)
        self.btn3 = QPushButton('open', self)
        self.btn3.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn3.clicked.connect(self.show_dialog_model)
        self.btn4 = QPushButton('open', self)
        self.btn4.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn4.clicked.connect(self.show_dialog_outdir)

        self.checkBox = QCheckBox("Check the box if you want to use TAP (Three-Axis-Prediction")
        self.checkBox.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.checkBox.toggle()

        self.btn5 = QPushButton('predict', self)
        self.btn5.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn5.clicked.connect(self.predicter)
        self.lbl = QLabel('original dir', self)
        self.lbl2 = QLabel('label dir', self)
        self.lbl3 = QLabel('model dir (contains model.hdf5)', self)
        self.lbl4 = QLabel('output dir', self)
        self.build()

        self.model = None
        self.worker_pred = None

    def build(self):
        vbox = QVBoxLayout()
        vbox.addWidget(combine_blocks(self.btn1, self.lbl))
        vbox.addWidget(combine_blocks(self.btn2, self.lbl2))
        vbox.addWidget(combine_blocks(self.btn3, self.lbl3))
        vbox.addWidget(combine_blocks(self.btn4, self.lbl4))
        vbox.addWidget(self.checkBox)
        vbox.addWidget(self.btn5)

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

    def show_dialog_model(self):
        default_path = max(self.opath, self.labelpath, os.path.expanduser('~'))
        f_name = QFileDialog.getExistingDirectory(self, 'Open Directory', default_path)
        if f_name:
            self.modelpath = f_name
            self.lbl3.setText(self.modelpath)

    def show_dialog_outdir(self):
        default_path = max(self.opath, self.labelpath, os.path.expanduser('~'))
        f_name = QFileDialog.getExistingDirectory(self, 'Open Directory', default_path)
        if f_name:
            self.outpath = f_name
            self.lbl4.setText(self.outpath)

    def get_newest_csv(self):
        csvs = sorted(list(Path(self.labelpath).glob('./*csv')))
        try:
            csv = pd.read_csv(str(csvs[-1]), index_col=0)
        except:
            csv = None
        return csv, str(csvs[-1])

    def predicter(self):
        ori_imgs, ori_filenames = load_X_gray(self.opath)
        print(ori_imgs.shape)
        input_shape = (512, 512, 1)
        num_classes = 1

        self.model = get_nested_unet(input_shape=input_shape, num_classes=num_classes)
        self.model.load_weights(os.path.join(self.modelpath, "model.hdf5"))

        self.btn5.setText('predicting')

        if self.checkBox.isChecked() is True:
            self.predict(ori_imgs, ori_filenames)
        else:
            self.predict_single(ori_imgs, ori_filenames)

    def predict(self, ori_imgs, filenames):
        try:
            predict_3ax(ori_imgs, self.model, self.outpath, filenames)
        except Exception as e:
            print(e)
        if self.labelpath != "":
            try:
                csv, csv_path = self.get_newest_csv()
                if csv is None:
                    pass
                else:
                    label_names = [node.filename for node in csv.itertuples() if node.train == "Checked"]
                    for ln in label_names:
                        shutil.copy(os.path.join(self.labelpath, ln), os.path.join(self.outpath, 'merged_prediction'))
                    shutil.copy(str(csv_path), os.path.join(self.outpath, 'merged_prediction'))
            except Exception as e:
                print(e)

        self.btn5.setText('predict')

    def predict_single(self, ori_imgs, filenames):
        try:
            predict_1ax(ori_imgs, self.model, self.outpath, filenames)
        except Exception as e:
            print(e)
        if self.labelpath != "":
            print('copy previous mask')
            try:
                csv, csv_path = self.get_newest_csv()
                print('find csv', csv_path)
                if csv is None:
                    pass
                else:
                    label_names = [node.filename for node in csv.itertuples() if node.train == "Checked"]
                    print(label_names)
                    for ln in label_names:
                        print('copy ln')
                        shutil.copy(os.path.join(self.labelpath, ln), os.path.join(self.outpath, 'merged_prediction'))
                    shutil.copy(str(csv_path), os.path.join(self.outpath, 'merged_prediction'))
                    print('csv copied')
            except Exception as e:
                print(e)

        self.btn5.setText('predict')
