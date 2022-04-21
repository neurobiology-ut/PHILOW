import io as IO
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from napari.qt import thread_worker
from napari_tools_menu import register_dock_widget
from qtpy.QtWidgets import QWidget, QPushButton, QSizePolicy, QLabel, QVBoxLayout, QFileDialog
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from napari_philow._models import get_nested_unet
from napari_philow._utils import combine_blocks, load_X_gray, load_Y_gray, select_train_data, divide_imgs


@register_dock_widget(menu="PHILOW > Train mode")
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
        self.build()

        self.model = None
        self.worker = None
        self.worker2 = None

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

    def update_layer(self, df):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(list(df['epoch']), list(df['dice_coeff']), label='dice_coeff')
        plt.xlim(0, 400)
        plt.ylim(0, 1)
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(list(df['epoch']), list(df['loss']), label='loss')
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
            self.worker.started.connect(lambda: print("worker is running..."))
            self.worker.finished.connect(lambda: print("worker stopped"))
            self.worker2 = self.yield_csv()
            self.worker2.yielded.connect(self.update_layer)
            self.worker2.start()

        if self.worker.is_running:
            self.model.stop_training = True
            print("stop training requested")
            self.btn4.setText('start training')
            self.worker = None
        else:
            self.worker.start()
            self.btn4.setText('stop')

    @thread_worker
    def train(self, devided_train_ori_imgs, devided_train_label_imgs, model):
        train_unet(
            X_train=devided_train_ori_imgs,
            Y_train=devided_train_label_imgs,
            csv_path=os.path.join(self.modelpath, "train_log.csv"),
            model_path=os.path.join(self.modelpath, "model.hdf5"),
            model=model
        )

    @thread_worker
    def yield_csv(self):
        while True:
            try:
                df = pd.read_csv(os.path.join(self.modelpath, "train_log.csv"))
                df['epoch'] = df['epoch'] + 1
                yield df
            except Exception as e:
                print(e)
            time.sleep(30)


def train_unet(X_train, Y_train, csv_path, model_path, model):
    Y_train = Y_train
    X_train = X_train

    data_gen_args = dict(
        rotation_range=90.,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True
    )
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    # Provide the same seed and keyword arguments to the fit and flow methods
    seed = 1
    image_datagen.fit(X_train, augment=True, seed=seed)
    mask_datagen.fit(Y_train, augment=True, seed=seed)

    image_generator = image_datagen.flow(X_train, seed=seed, batch_size=8)
    mask_generator = mask_datagen.flow(Y_train, seed=seed, batch_size=8)

    # combine generators into one which yields image and masks
    train_generator = (pair for pair in zip(image_generator, mask_generator))

    BATCH_SIZE = 4
    NUM_EPOCH = 400

    callbacks = []
    callbacks.append(CSVLogger(csv_path))
    callbacks.append(ModelCheckpoint(model_path))
    history = model.fit(train_generator, steps_per_epoch=np.ceil(len(X_train / BATCH_SIZE)), epochs=NUM_EPOCH,
                        verbose=1, callbacks=callbacks)
    model.save_weights(model_path)
