import io as IO
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from PIL import Image
from napari.qt.threading import create_worker
from qtpy.QtWidgets import QWidget, QPushButton, QSizePolicy, QLabel, QVBoxLayout, QFileDialog, QCheckBox, QSpinBox
from segmentation_models_pytorch import UnetPlusPlus
from torch import optim
from torch.utils import data

from napari_philow._utils import combine_blocks
from napari_philow.segmentation.dataset import PHILOWDataset, ImageTransform
from napari_philow.segmentation.loss import DiceBCELoss
from napari_philow.segmentation.train import train_model


class Trainer(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self._viewer = napari_viewer
        self.opath = ""
        self.labelpath = ""
        self.modelpath = ""
        self.btn1 = QPushButton('open', self)
        self.btn1.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn1.clicked.connect(self.show_dialog_o)
        self.btn2 = QPushButton('open', self)
        self.btn2.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn2.clicked.connect(self.show_dialog_label)
        self.btn3 = QPushButton('open', self)
        self.btn3.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn3.clicked.connect(self.show_dialog_model)
        self.btn4 = QPushButton('start training', self)
        self.btn4.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn4.clicked.connect(self.trainer)
        self.lbl = QLabel('original dir', self)
        self.lbl2 = QLabel('label dir', self)
        self.lbl3 = QLabel('model output dir', self)
        self.lbl4 = QLabel('epochs', self)
        self.epoch = QSpinBox(maximum=1000, value=400)
        self.checkBox = QCheckBox("Resize to 256x256?")
        self.checkBox_split = QCheckBox("Split and create validation data from training data?")
        self.checkBox.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        with plt.style.context('dark_background'):
            self.canvas = FigureCanvas(Figure(figsize=(3, 5)))
            self.canvas.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Maximum)
            self.axes = self.canvas.figure.subplots()

        self.build()

        self.model = None
        self.worker = None
        self.stop_training = False

        self.image_layer = None
        self.label_layer = None
        self.prediction_layer = None

        self.df = pd.DataFrame(columns=['epoch', 'train_loss', 'val_loss'])

    def build(self):
        vbox = QVBoxLayout()
        vbox.addWidget(combine_blocks(self.btn1, self.lbl))
        vbox.addWidget(combine_blocks(self.btn2, self.lbl2))
        vbox.addWidget(combine_blocks(self.btn3, self.lbl3))
        vbox.addWidget(combine_blocks(self.lbl4, self.epoch))
        # vbox.addWidget(self.checkBox)
        vbox.addWidget(self.checkBox_split)
        vbox.addWidget(self.btn4)
        vbox.addWidget(self.canvas)

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

    def get_newest_csv(self):
        csvs = sorted(list(Path(self.labelpath).glob('./*csv')))
        csv = pd.read_csv(str(csvs[-1]), index_col=0)
        return csv

    def update_layer(self, value):
        self.axes.clear()
        self.df.loc[len(self.df)] = {'epoch': value[0], 'train_loss': value[1], 'val_loss': value[2]}
        self.df.to_csv(os.path.join(self.modelpath, "train_log.csv"))
        self.axes.plot(list(self.df['epoch']), list(self.df['train_loss']), label='train_loss')
        if self.checkBox_split.isChecked():
            self.axes.plot(list(self.df['epoch']), list(self.df['val_loss']), label='val_loss')
        self.axes.set_xlim(0, len(self.df))
        # plt.ylim(0, 1)
        self.axes.legend()
        self.canvas.draw_idle()
        self.canvas.flush_events()

        image_mask_pred = value[3]

        if image_mask_pred is None:
            return

        if self.image_layer is None:
            self.image_layer = self._viewer.add_image(image_mask_pred[0], name='image')
        else:
            self.image_layer.data = image_mask_pred[0]
        if self.label_layer is None:
            if image_mask_pred[1] is not None:
                self.label_layer = self._viewer.add_labels(1*(image_mask_pred[1] > 0.5), name='label', color={1: 'green'}, blending='additive')
        else:
            if image_mask_pred[1] is not None:
                self.label_layer.data = 1*(image_mask_pred[1] > 0.5)
        if self.prediction_layer is None:
            if image_mask_pred[2] is not None:
                self.prediction_layer = self._viewer.add_labels(1*(image_mask_pred[2] > 0.5), name='prediction', color={1: 'magenta'}, blending='additive')
        else:
            if image_mask_pred[2] is not None:
                self.prediction_layer.data = 1*(image_mask_pred[2] > 0.5)

    def delete_worker(self):
        del self.worker
        self.worker = None
        self.btn4.setText('start training')

    def trainer(self):
        if self.worker:
            if self.worker.is_running:
                self.btn4.setText('stopping...')
                self.stop_training = True
                self.worker.send(self.stop_training)
            else:
                self.delete_worker()
        else:
            self.df = pd.DataFrame(columns=['epoch', 'train_loss', 'val_loss'])
            csv = self.get_newest_csv()
            names = list(csv[csv['train'] == 'Checked']['filename'])
            test_names = [name for name in list(csv[csv['train'] != 'Checked']['filename']) if
                          os.path.isfile(os.path.join(self.opath, name))]
            print(test_names)
            if self.checkBox_split.isChecked():
                np.random.shuffle(names)
                split_index = 9 * len(names) // 10
                train_names = names[0: split_index]
                val_names = names[split_index:]
            else:
                train_names = names

            w, h = Image.open(os.path.join(self.opath, names[0])).size

            batch_size = min(math.ceil(len(train_names) / 10), 4)
            print('batchsize = ', batch_size)

            train_dataset = PHILOWDataset(self.opath, self.labelpath, train_names, 'train', ImageTransform(512),
                                          multiplier=math.ceil(max(w, h) / 512))
            train_dataloader = data.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, num_workers=max(1, os.cpu_count() - 2))
            if len(test_names) != 0:
                test_dataset = PHILOWDataset(self.opath, self.labelpath, test_names, 'val', ImageTransform(512))
                test_dataloader = data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=1)
            else:
                test_dataloader = None

            if self.checkBox_split.isChecked():
                val_dataset = PHILOWDataset(self.opath, self.labelpath, val_names, 'val', ImageTransform(512),
                                            multiplier=math.ceil(max(w, h) / 512))
                val_dataloader = data.DataLoader(
                    val_dataset, batch_size=batch_size, shuffle=True, num_workers=max(1, os.cpu_count() - 2))
            else:
                val_dataloader = None

            # 辞書オブジェクトにまとめる
            dataloaders_dict = {"train": train_dataloader, "val": val_dataloader, "test": test_dataloader}

            print(dataloaders_dict)

            net = UnetPlusPlus(encoder_name="efficientnet-b0", encoder_weights="imagenet", in_channels=1, classes=1,
                               activation='sigmoid')

            optimizer = optim.AdamW(net.parameters())

            num_epochs = self.epoch.value()

            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

            criterion = DiceBCELoss()

            self.worker = create_worker(train_model, self.modelpath, net, dataloaders_dict, criterion, scheduler,
                                        optimizer,
                                        num_epochs=num_epochs)
            self.worker.started.connect(lambda: print("worker is running..."))
            self.worker.yielded.connect(self.update_layer)
            self.worker.finished.connect(self.delete_worker)

        if self.worker.is_running:
            self.btn4.setText('stopping...')
            self.stop_training = True
        else:
            self.worker.start()
            self.stop_training = False
            self.btn4.setText('stop')
