import os
import shutil
from pathlib import Path

import pandas as pd
import torch
from qtpy.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QSizePolicy, QLabel, QFileDialog,
                            QCheckBox)
from segmentation_models_pytorch import UnetPlusPlus

from napari_philow._utils import combine_blocks

from napari_philow._predict import predict_3ax, predict_1ax


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
        self.lbl3 = QLabel('model path (.pth)', self)
        self.lbl4 = QLabel('output dir', self)
        self.build()

        self.net = None
        self.worker_pred = None
        self.ori_filenames = None
        self.device = None
        self.size = 512

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
        f_name, _ = QFileDialog.getOpenFileName(self, 'Select weight file', default_path)
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
        self.ori_filenames = sorted(list(Path(self.opath).glob('./*.png')))

        self.net = UnetPlusPlus(encoder_name="efficientnet-b0", encoder_weights="imagenet", in_channels=1, classes=1,
                           activation='sigmoid')
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(self.modelpath, map_location=torch.device(self.device))
        self.net.load_state_dict(state_dict)
        self.net.to(self.device)

        self.btn5.setText('predicting')

        if self.checkBox.isChecked() is True:

            self.predict()
        else:
            self.predict_single()

    def predict(self):
        try:
            predict_3ax(self.opath, self.net, self.outpath, self.size, self.device)
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


    def predict_single(self):
        try:
            predict_1ax(self.ori_filenames, self.net, self.outpath, self.size, self.device)
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
