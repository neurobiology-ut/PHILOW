import os
import shutil
from pathlib import Path

from qtpy.QtWidgets import (
    QButtonGroup,
    QWidget,
    QPushButton,
    QSlider,
    QCheckBox,
    QLabel,
    QSpinBox,
    QHBoxLayout,
    QVBoxLayout,
    QFileDialog,
    QComboBox,
    QGridLayout,
    QGroupBox,
)

from qtpy.QtCore import Qt
import pandas as pd


GUI_MAXIMUM_WIDTH = 225
GUI_MAXIMUM_HEIGHT = 350
GUI_MINIMUM_HEIGHT = 300


class Datamanager(QWidget):

    def __init__(self, *args, **kwargs):

        super(Datamanager, self).__init__(*args, **kwargs)

        layout = QVBoxLayout()

        # add some buttons
        self.button = QPushButton('1', self)
        self.button.clicked.connect(self.button_func)

        io_panel = QWidget()
        io_layout = QHBoxLayout()
        io_layout.addWidget(self.button)
        io_panel.setLayout(io_layout)
        io_panel.setMaximumWidth(GUI_MAXIMUM_WIDTH)
        layout.addWidget(io_panel)

        # set the layout
        #layout.setAlignment(Qt.AlignTop)
        #layout.setSpacing(4)

        self.setLayout(layout)
        #self.setMaximumHeight(GUI_MAXIMUM_HEIGHT)
        #self.setMaximumWidth(GUI_MAXIMUM_WIDTH)

        self.df = ""
        self.csv_path = ""
        self.slice_num = 0

    def prepare(self, label_dir, model_type, checkbox):
        """
        初期動作
        :param str label_dir:labelのpath
        :param str model_type:modelのtype
        :param bool checkbox:新しいdatasetを作るかどうか
        """
        self.df, self.csv_path = self.load_csv(label_dir, model_type, checkbox)
        print(self.csv_path, checkbox)
        #self.check_all_data_and_mod()
        self.update(0)
        # return df, train_data_dir, csv_path

    def load_csv(self, label_dir, model_type, checkbox):
        """
        csvをloadし、一番新しいcsvをloadするか、csvがなければ生成する
        :param str label_dir: labelのpath
        :param str model_type:modelのtype
        :param bool checkbox:新しいdatasetを作るかどうか
        :return: データフレーム、trainingデータの場所、csvのpath
        :rtype (pandas.DataFrame, str, str)
        """
        csvs = sorted(list(Path(label_dir).glob(f'{model_type}*.csv')))
        if len(csvs) == 0:
            df, train_data_dir, csv_path = self.create(label_dir, model_type)
        else:
            csv_path = str(csvs[-1])
            df = pd.read_csv(csv_path, index_col=0)
            if checkbox is True:
                csv_path = csv_path.split('_train')[0] + '_train' \
                           + str(int(os.path.splitext(csv_path.split('_train')[1])[0]) + 1) + '.csv'
                df.to_csv(csv_path)
            else:
                pass
            #train_data_dir = os.path.join(f'./training', os.path.splitext(os.path.basename(csv_path))[0])
        return df, csv_path

    def create(self, label_dir, model_type):
        """
        新しいdataframeを作成し、csvの保存とtrainingデータの場所を作成
        :param str label_dir: labelのpath
        :param str model_type:modelのtype
        :return: データフレームとtrainingデータの場所
        :rtype (pandas.DataFrame, str)
        """
        labels = sorted(list(Path(label_dir).glob('./*png')))
        df = pd.DataFrame({'filename': labels,
                           'train': ['Not Checked']*len(labels)})
        csv_path = os.path.join(label_dir, f'{model_type}_train0.csv')
        df.to_csv(csv_path)
        #train_data_dir = f'./training/{model_type}_train0'
        # TODO 学習用の場所を指定できるように
        return df, csv_path

    def update(self, slice_num):
        self.slice_num = slice_num
        self.button.setText(self.df.at[self.df.index[self.slice_num], 'train'])

    def button_func(self):
        """ ボタンがトグルされたときのスロット """
        if self.button.text() == 'Not Checked':
            self.button.setText('Checked')
            self.df.at[self.df.index[self.slice_num], 'train'] = 'Checked'
            self.df.to_csv(self.csv_path)
            #self.move_data()
        else:
            self.button.setText('Not Checked')
            self.df.at[self.df.index[self.slice_num], 'train'] = 'Not Checked'
            self.df.to_csv(self.csv_path)
            #self.delete_data()

    def move_data(self):
        shutil.copy(self.df.at[self.df.index[self.slice_num], 'filename'], self.train_data_dir)

    def delete_data(self):
        os.remove(os.path.join(self.train_data_dir,
                               os.path.basename(self.df.at[self.df.index[self.slice_num], 'filename'])))

    def check_all_data_and_mod(self):
        for i in range(len(self.df)):
            if self.df.at[self.df.index[i], 'train'] == 'Checked':
                try:
                    shutil.copy(self.df.at[self.df.index[i], 'filename'], self.train_data_dir)
                except:
                    pass
            else:
                try:
                    os.remove(os.path.join(self.train_data_dir,
                                           os.path.basename(self.df.at[self.df.index[i], 'filename'])))
                except:
                    pass
