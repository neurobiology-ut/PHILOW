import os
import shutil
import warnings
from pathlib import Path

import pandas as pd
import torch
from qtpy.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QSizePolicy, QLabel, QFileDialog,
                            QCheckBox)
from qtpy.QtCore import Qt
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
        self.mitopath = ""  # 追加: ミトコンドリアのラベルディレクトリを保持する変数
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

        self.checkBox_cristae = QCheckBox("Use cristae inference mode")  # 追加: クリステの推論モードを使うかどうかのチェックボックス
        self.checkBox_cristae.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.checkBox_cristae.stateChanged.connect(self.toggle_mito_dir)  # 追加: チェックボックスの状態が変更されたときのシグナル接続
        self.lbl_mito = QLabel('mitochondria label dir', self)  # 追加: ミトコンドリアのラベルディレクトリ選択用のラベル
        self.btn_mito = QPushButton('open', self)  # 追加: ミトコンドリアのラベルディレクトリ選択用のボタン
        self.btn_mito.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn_mito.clicked.connect(self.show_dialog_mito)
        self.btn_mito.setVisible(False)  # 追加: 初期状態では非表示
        self.lbl_mito.setVisible(False)  # 追加: 初期状態では非表示

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
        vbox.addWidget(self.checkBox_cristae)  # 追加: クリステの推論モードのチェックボックスをレイアウトに追加
        vbox.addWidget(combine_blocks(self.btn_mito, self.lbl_mito))  # 追加: ミトコンドリアのラベルディレクトリ選択用のボタンとラベルをレイアウトに追加
        vbox.addWidget(self.btn5)

        self.setLayout(vbox)
        self.show()

    def toggle_mito_dir(self, state):
        if state == Qt.Checked:
            self.btn_mito.setVisible(True)
            self.lbl_mito.setVisible(True)
        else:
            self.btn_mito.setVisible(False)
            self.lbl_mito.setVisible(False)

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

    def show_dialog_mito(self):
        default_path = max(self.opath, self.labelpath, os.path.expanduser('~'))
        f_name = QFileDialog.getExistingDirectory(self, 'Open Directory', default_path)
        if f_name:
            self.mitopath = f_name
            self.lbl_mito.setText(self.mitopath)

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
        if self.checkBox_cristae.isChecked() and not self.mitopath:
            warnings.warn('Please select mitochondria label directory for cristae inference mode.', UserWarning)
            return
        self.ori_filenames = sorted(list(Path(self.opath).glob('./*.png')))

        if self.checkBox_cristae.isChecked() is True:
            self.net = UnetPlusPlus(encoder_name="efficientnet-b0", encoder_weights="imagenet", in_channels=1, classes=3,
                                    activation='sigmoid')
        else:
            self.net = UnetPlusPlus(encoder_name="efficientnet-b0", encoder_weights="imagenet", in_channels=1, classes=1,
                                    activation='sigmoid')
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
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
            self.copy_previous_mask()
        self.btn5.setText('predict')

    def predict_single(self):
        try:
            if self.checkBox_cristae.isChecked() is True:
                predict_1ax(self.ori_filenames, self.net, self.outpath, self.size, self.device, mask_dir=self.mitopath, out_channel=[0])
            else:
                predict_1ax(self.ori_filenames, self.net, self.outpath, self.size, self.device)
        except Exception as e:
            print(e)
        if self.labelpath != "":
            self.copy_previous_mask()
        self.btn5.setText('predict')

    def copy_previous_mask(self):
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
