import copy
import os
from pathlib import Path

import dask_image.imread
from qtpy.QtWidgets import QWidget, QHBoxLayout, QSlider, QLabel
from qtpy.QtCore import Qt
import numpy as np
from scipy import ndimage
from skimage import io
from skimage.filters import gaussian
import pandas as pd


def combine_blocks(block1, block2):
    temp_widget = QWidget()
    temp_layout = QHBoxLayout()
    temp_layout.addWidget(block1)
    temp_layout.addWidget(block2)
    temp_widget.setLayout(temp_layout)
    return temp_widget


def normalize_x(image):
    image = image / 127.5 - 1
    return image


def normalize_y(image):
    image = image / 255
    return image


def denormalize_y(image):
    return image * 255


def annotation_to_input(label_ermito):
    mito = (label_ermito == 1) * 255
    er = (label_ermito == 2) * 255
    mito = normalize_y(mito)
    er = normalize_y(er)
    mito_anno = np.zeros_like(mito)
    er_anno = np.zeros_like(er)
    mito = gaussian(mito, sigma=2) * 255
    er = gaussian(er, sigma=2) * 255
    mito_anno[:, :] = mito
    er_anno[:, :] = er
    anno = np.concatenate([mito_anno[:, :, np.newaxis], er_anno[:, :, np.newaxis]], 2)
    anno = normalize_x(anno[np.newaxis, :, :, :])
    return anno


def check_csv(project_path, ext):
    if not os.path.isfile(os.path.join(project_path, os.path.basename(project_path) + '.csv')):
        cols = ['project', 'type', 'ext', 'z', 'y', 'x', 'z_size', 'y_size', 'x_size', 'created_date', 'update_date',
                'path', 'notes']
        df = pd.DataFrame(index=[], columns=cols)
        filename_pattern_original = os.path.join(project_path, f'dataset/Original_size/Original/*{ext}')
        images_original = dask_image.imread.imread(filename_pattern_original)
        z, y, x = images_original.shape
        record = pd.Series(
            [os.path.basename(project_path), 'dataset', '.tif', 0, 0, 0, z, y, x, datetime.datetime.now(),
             '', os.path.join(project_path, 'dataset/Original_size/Original'), ''], index=df.columns)
        df = df.append(record, ignore_index=True)
        df.to_csv(os.path.join(project_path, os.path.basename(project_path) + '.csv'))
    else:
        pass


def check_annotations_dir(project_path):
    if not os.path.isdir(os.path.join(project_path, 'annotations/Original_size/master')):
        os.makedirs(os.path.join(project_path, 'annotations/Original_size/master'))
    else:
        pass


def check_zarr(project_path, ext):
    if not len(list((Path(project_path) / 'dataset' / 'Original_size').glob('./*.zarr'))):
        filename_pattern_original = os.path.join(project_path, f'dataset/Original_size/Original/*{ext}')
        images_original = dask_image.imread.imread(filename_pattern_original)
        images_original.to_zarr(os.path.join(project_path, f'dataset/Original_size/Original.zarr'))
    else:
        pass


def check(project_path, ext):
    check_csv(project_path, ext)
    check_zarr(project_path, ext)
    check_annotations_dir(project_path)


def load_images(directory):
    filename_pattern_original = os.path.join(directory, '*png')
    images_original = dask_image.imread.imread(filename_pattern_original)
    return images_original


def load_predicted_masks(mito_mask_dir, er_mask_dir):
    filename_pattern_mito_label = os.path.join(mito_mask_dir, '*png')
    filename_pattern_er_label = os.path.join(er_mask_dir, '*png')
    images_mito_label = dask_image.imread.imread(filename_pattern_mito_label)
    images_mito_label = images_mito_label.compute()
    images_er_label = dask_image.imread.imread(filename_pattern_er_label)
    images_er_label = images_er_label.compute()
    base_label = (images_mito_label > 127) * 1 + (images_er_label > 127) * 2
    return base_label


def load_saved_masks(mod_mask_dir):
    filename_pattern_label = os.path.join(mod_mask_dir, '*png')
    images_label = dask_image.imread.imread(filename_pattern_label)
    images_label = images_label.compute()
    base_label = images_label
    return base_label


def load_raw_masks(raw_mask_dir):
    filename_pattern_raw = os.path.join(raw_mask_dir, '*png')
    images_raw = dask_image.imread.imread(filename_pattern_raw)
    images_raw = images_raw.compute()
    base_label = np.where((126 < images_raw) & (images_raw < 171), 255, 0)
    return base_label


def combine_blocks(block1, block2):
    temp_widget = QWidget()
    temp_layout = QHBoxLayout()
    temp_layout.addWidget(block1)
    temp_layout.addWidget(block2)
    temp_widget.setLayout(temp_layout)
    return temp_widget


def save_masks(labels, out_path):
    num = labels.shape[0]
    os.makedirs(out_path, exist_ok=True)
    for i in range(num):
        label = labels[i]
        io.imsave(os.path.join(out_path, str(i).zfill(4) + '.png'), label)


def label_and_sort(base_label):
    labeled = ndimage.label(base_label, structure=np.ones((3, 3, 3)))[0]

    mks, nums = np.unique(labeled, return_counts=True)

    idx_list = list(np.argsort(nums[1:]))
    nums = np.sort(nums[1:])
    labeled_sorted = np.zeros_like(labeled)
    for i, idx in enumerate(idx_list):
        labeled_sorted = np.where(labeled == mks[1:][idx], i+1, labeled_sorted)
    return labeled_sorted, nums


def label_ct(labeled_array, nums, value):
    labeled_temp = copy.copy(labeled_array)
    idx = np.abs(nums - value).argmin()
    labeled_temp = np.where((labeled_temp < idx) & (labeled_temp != 0), 255, 0)
    return labeled_temp


def crop_img(points, layer):
    min_vals = [x - 50 for x in points]
    max_vals = [x + 50 for x in points]
    yohaku_minus = [n if n < 0 else 0 for n in min_vals]
    yohaku_plus = [x - layer.data.shape[i] if layer.data.shape[i] < x else 0 for i, x in
                   enumerate(max_vals)]
    crop_slice = tuple(slice(np.maximum(0, n), x) for n, x in zip(min_vals, max_vals))
    crop_temp = layer.data[crop_slice].persist().compute()
    cropped_img = np.zeros((100, 100, 100), np.uint8)
    cropped_img[-yohaku_minus[0]:100 - yohaku_plus[0], -yohaku_minus[1]:100 - yohaku_plus[1]
    , -yohaku_minus[2]:100 - yohaku_plus[2]] = crop_temp
    return cropped_img


def deletewidgets(layout):
    while True:
        count = layout.count()
        print(count)
        if count == 1:
            break
        item = layout.itemAt(count - 2)
        widget = item.widget()
        widget.deleteLater()


def show_so_layer(args):
    labeled_c, labeled_sorted, nums, viewer = args
    so_layer = viewer.add_image(labeled_c, colormap='cyan', name='small_object', blending='additive')

    object_slider = QSlider(Qt.Horizontal)
    object_slider.setMinimum(0)
    object_slider.setMaximum(100)
    object_slider.setSingleStep(10)
    object_slider.setValue(10)

    object_slider.valueChanged[int].connect(lambda value=object_slider: calc_object_callback(so_layer, value,
                                                                                             labeled_sorted,
                                                                                             nums))
    lbl = QLabel('object size')
    slider_widget = combine_blocks(lbl, object_slider)
    viewer.window.add_dock_widget(slider_widget, name='object_size_slider', area='left')

    def calc_object_callback(t_layer, value, labeled_array, nums):
        t_layer.data = label_ct(labeled_array, nums, value)
