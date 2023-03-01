import io as IO
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from napari.qt.threading import create_worker
from qtpy.QtWidgets import QWidget, QPushButton, QSizePolicy, QLabel, QVBoxLayout, QFileDialog, QCheckBox
from segmentation_models_pytorch import UnetPlusPlus
from torch import optim
from torch.utils import data

from napari_philow._utils import combine_blocks
from napari_philow.segmentation.dataset import PHILOWDataset, ImageTransform
from napari_philow.segmentation.loss import DiceBCELoss


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
        self.checkBox = QCheckBox("resize to 256x256?")
        self.checkBox.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.build()

        self.model = None
        self.worker = None
        self.df = pd.DataFrame(columns=['epoch', 'dice_coeff', 'loss'])

    def build(self):
        vbox = QVBoxLayout()
        vbox.addWidget(combine_blocks(self.btn1, self.lbl))
        vbox.addWidget(combine_blocks(self.btn2, self.lbl2))
        vbox.addWidget(combine_blocks(self.btn3, self.lbl3))
        vbox.addWidget(self.btn4)

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
        self.df[len(self.df)] = {'epoch': value[0], 'dice_coeff': value[1], 'loss': value[2]}
        self.df.to_csv(os.path.join(self.modelpath, "train_log.csv"))
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(list(self.df['epoch']), list(self.df['dice_coeff']), label='dice_coeff')
        plt.xlim(0, 400)
        plt.ylim(0, 1)
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(list(self.df['epoch']), list(self.df['loss']), label='loss')
        plt.legend()
        plt.xlim(0, 400)
        buf = IO.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        im = Image.open(buf)
        im = np.array(im)
        buf.close()
        try:
            self._viewer.layers['result'].data = im
        except KeyError:
            self._viewer.add_image(
                im, name='result'
            )

    def trainer(self):
        if self.worker:
            if self.worker.is_running:
                pass
            else:
                self.worker.start()
                self.btn4.setText('stop')
        else:
            """
            ori_imgs, ori_filenames = load_X_gray(self.opath)
            label_imgs, label_filenames = load_Y_gray(self.labelpath, normalize=False)
            train_csv = self.get_newest_csv()
            train_ori_imgs, train_label_imgs = select_train_data(
                dataframe=train_csv,
                ori_imgs=ori_imgs,
                label_imgs=label_imgs,
                ori_filenames=ori_filenames,
            )
            devided_train_ori_imgs = divide_imgs(train_ori_imgs)
            devided_train_label_imgs = divide_imgs(train_label_imgs)
            devided_train_label_imgs = np.where(
                devided_train_label_imgs < 0,
                0,
                devided_train_label_imgs
            )

            self.model = get_nested_unet(input_shape=(512, 512, 1), num_classes=1)
            
            self.worker = self.train(devided_train_ori_imgs, devided_train_label_imgs, self.model)
            
            """

            df = self.get_newest_csv()
            names = list(df[df['train'] == 'Checked']['filename'])
            # if validation:
            # np.random.shuffle(names)
            # split_index = 9*len(names)//10
            # train_names = names[0: split_index]
            # val_names = names[split_index:]

            w, h = Image.open(os.path.join(self.opath), names[0]).size

            train_dataset = PHILOWDataset(self.opath, self.labelpath, names, 'train', ImageTransform(512),
                                          multiplier=math.ceil(max(w, h) / 512))

            batch_size = 4

            train_dataloader = data.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, num_workers=max(1, os.cpu_count() - 2))

            # 辞書オブジェクトにまとめる
            # dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}
            dataloaders_dict = {"train": train_dataloader}

            net = UnetPlusPlus(encoder_name="efficientnet-b0", encoder_weights="imagenet", in_channels=1, classes=1,
                               activation='sigmoid')

            optimizer = optim.AdamW(net.parameters())

            num_epochs = 400  # TODO create input form for specify epochs

            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

            criterion = DiceBCELoss()

            self.worker = create_worker(self.train_model, self.modelpath, net, dataloaders_dict, criterion, scheduler, optimizer,
                                           num_epochs=num_epochs)

            self.worker.started.connect(lambda: print("worker is running..."))
            self.worker.finished.connect(lambda: print("worker stopped"))
            self.worker.yielded.connect(self.update_layer)

        if self.worker.is_running:
            self.model.stop_training = True
            print("stop training requested")
            self.btn4.setText('start training')
            self.worker = None
        else:
            self.worker.start()
            self.btn4.setText('stop')