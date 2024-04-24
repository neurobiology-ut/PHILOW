import copy
import os
from datetime import datetime
from pathlib import Path

import dask
import dask_image.imread
from qtpy.QtWidgets import QWidget, QHBoxLayout, QSlider, QLabel
from qtpy.QtCore import Qt
import numpy as np
from scipy import ndimage
from skimage import io, morphology
import pandas as pd


def renormalize_8bit(image):
    origin_min = image.min()
    origin_max = image.max()
    image -= origin_min
    image = image/(origin_max-origin_min)
    image *= 255
    return image.astype(np.uint8)


def check_csv(project_path, ext):
    if not os.path.isfile(os.path.join(project_path, os.path.basename(project_path) + '.csv')):
        cols = ['project', 'type', 'ext', 'z', 'y', 'x', 'z_size', 'y_size', 'x_size', 'created_date', 'update_date',
                'path', 'notes']
        df = pd.DataFrame(index=[], columns=cols)
        filename_pattern_original = os.path.join(project_path, f'dataset/Original_size/Original/*{ext}')
        images_original = dask_image.imread.imread(filename_pattern_original)
        z, y, x = images_original.shape
        record = pd.Series(
            [os.path.basename(project_path), 'dataset', '.tif', 0, 0, 0, z, y, x, datetime.now(),
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
    renormalized_images_original = dask.array.asarray([ renormalize_8bit(images_original[z]) for z in range(images_original.shape[0])])
    return renormalized_images_original


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
    return base_label, [x.name for x in sorted(list(Path(mod_mask_dir).glob('./*png')))]


def load_mask_masks(mask_dir):
    filename_pattern_label = os.path.join(mask_dir, '*png')
    images_label = dask_image.imread.imread(filename_pattern_label)
    if images_label.max() == 1:
        pass
    else:
        images_label = 1 * (images_label > 0)
    base_label = images_label
    return base_label.compute()

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


def save_masks(labels, out_path, filenames):
    num = labels.shape[0]
    for i in range(num):
        label = labels[i]
        io.imsave(os.path.join(out_path, filenames[i]), label)

def label_and_sort(base_label):
    labeled = ndimage.label(base_label, structure=np.ones((3, 3, 3)))[0]

    mks, nums = np.unique(labeled, return_counts=True)

    idx_list = list(np.argsort(nums[1:]))
    nums = np.sort(nums[1:])
    labeled_sorted = np.zeros_like(labeled)
    for i, idx in enumerate(idx_list):
        labeled_sorted = np.where(labeled == mks[1:][idx], i + 1, labeled_sorted)
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


def preprocess_cristae(ori_path, mito_path, cristae_path, names, crop_size=1000):
    ori_imgs = []
    for name in names:
        ori_img = io.imread(os.path.join(ori_path, name), as_gray=True)
        if ori_img.dtype == np.uint8:
            ori_imgs.append(ori_img)
        else:
            ori_imgs.append(renormalize_8bit(ori_img))
    mito_imgs = [io.imread(os.path.join(mito_path, name), as_gray=True) for name in names]
    cristae_imgs = [io.imread(os.path.join(cristae_path, name), as_gray=True) for name in names]

    # make gap
    preprocessed_imgs = []
    for i in range(len(cristae_imgs)):
        dilated_cristae_img = morphology.binary_dilation(cristae_imgs[i], morphology.disk(30))
        dilated_cristae_img = dilated_cristae_img - cristae_imgs[i]
        dilated_cristae_img = dilated_cristae_img * mito_imgs[i]
        merged_img = np.concatenate([
            cristae_imgs[i][..., np.newaxis],
            dilated_cristae_img[..., np.newaxis],
            np.zeros_like(cristae_imgs[i][..., np.newaxis])
        ], axis=-1)
        preprocessed_imgs.append(merged_img)
    preprocessed_imgs = np.array(preprocessed_imgs)

    # crop
    cropped_ori_imgs = []
    cropped_cristae_imgs = []
    H = ori_imgs[0].shape[0] // crop_size + 1
    W = ori_imgs[0].shape[1] // crop_size + 1
    for z in range(len(ori_imgs)):
        margin_ori_img = np.zeros((H * crop_size, W * crop_size), ori_imgs[0].dtype)
        margin_ori_img[:ori_imgs[0].shape[0], :ori_imgs[0].shape[1]] = ori_imgs[z] * mito_imgs[z]
        margin_label_img = np.zeros((H * crop_size, W * crop_size, preprocessed_imgs.shape[3]), preprocessed_imgs.dtype)
        margin_label_img[:preprocessed_imgs.shape[1], :preprocessed_imgs.shape[2]] = preprocessed_imgs[z]
        for h in range(H):
            for w in range(W):
                cropped_ori_img = margin_ori_img[h * crop_size:(h + 1) * crop_size, w * crop_size:(w + 1) * crop_size]
                if np.max(cropped_ori_img) > 0:
                    cropped_ori_imgs.append(cropped_ori_img)
                    cropped_label_img = margin_label_img[h * crop_size:(h + 1) * crop_size,
                                        w * crop_size:(w + 1) * crop_size]
                    cropped_label_img *= 255
                    cropped_cristae_imgs.append(cropped_label_img)

    return np.array(cropped_ori_imgs), np.array(cropped_cristae_imgs)


def select_train_data(dataframe, ori_imgs, label_imgs, ori_filenames):
    train_img_names = list()
    for node in dataframe.itertuples():
        if node.train == "Checked":
            train_img_names.append(node.filename)
    train_ori_imgs = list()
    train_label_imgs = list()
    for ori_img, label_img, train_filename in zip(ori_imgs, label_imgs, ori_filenames):
        if train_filename in train_img_names:
            train_ori_imgs.append(ori_img)
            train_label_imgs.append(label_img)
    print(ori_filenames)
    return np.array(train_ori_imgs), np.array(train_label_imgs)


def divide_imgs(images):
    H = -(-images.shape[1] // 412)
    W = -(-images.shape[2] // 412)

    diveded_imgs = np.zeros((images.shape[0] * H * W, 512, 512, 1), np.float32)
    print(H, W)

    for z in range(images.shape[0]):
        image = images[z]
        for h in range(H):
            for w in range(W):
                cropped_img = np.zeros((512, 512, 1), np.float32)
                cropped_img -= 1

                if images.shape[1] < 412:
                    h = -1
                if images.shape[2] < 412:
                    w = -1

                if h == -1:
                    if w == -1:
                        cropped_img[50:images.shape[1] + 50, 50:images.shape[2] + 50, 0] = image[0:images.shape[1],
                                                                                           0:images.shape[2], 0]
                    elif w == 0:
                        cropped_img[50:images.shape[1] + 50, 50:512, 0] = image[0:images.shape[1], 0:462, 0]
                    elif w == W - 1:
                        cropped_img[50:images.shape[1] + 50, 0:images.shape[2] - 412 * W - 50, 0] = image[
                                                                                                    0:images.shape[1],
                                                                                                    w * 412 - 50:
                                                                                                    images.shape[2], 0]
                    else:
                        cropped_img[50:images.shape[1] + 50, :, 0] = image[0:images.shape[1],
                                                                     w * 412 - 50:(w + 1) * 412 + 50, 0]
                elif h == 0:
                    if w == -1:
                        cropped_img[50:512, 50:images.shape[2] + 50, 0] = image[0:462, 0:images.shape[2], 0]
                    elif w == 0:
                        cropped_img[50:512, 50:512, 0] = image[0:462, 0:462, 0]
                    elif w == W - 1:
                        cropped_img[50:512, 0:images.shape[2] - 412 * W - 50, 0] = image[0:462,
                                                                                   w * 412 - 50:images.shape[2], 0]
                    else:
                        # cropped_img[50:512, :, 0] = image[0:462, w*412-50:(w+1)*412+50, 0]
                        try:
                            cropped_img[50:512, :, 0] = image[0:462, w * 412 - 50:(w + 1) * 412 + 50, 0]
                        except:
                            cropped_img[50:512, 0:images.shape[2] - 412 * (W - 1) - 50, 0] = image[0:462, w * 412 - 50:(
                                                                                                                               w + 1) * 412 + 50,
                                                                                             0]
                elif h == H - 1:
                    if w == -1:
                        cropped_img[0:images.shape[1] - 412 * H - 50, 50:images.shape[2] + 50, 0] = image[h * 412 - 50:
                                                                                                          images.shape[
                                                                                                              1],
                                                                                                    0:images.shape[2],
                                                                                                    0]
                    elif w == 0:
                        cropped_img[0:images.shape[1] - 412 * H - 50, 50:512, 0] = image[h * 412 - 50:images.shape[1],
                                                                                   0:462, 0]
                    elif w == W - 1:
                        cropped_img[0:images.shape[1] - 412 * H - 50, 0:images.shape[2] - 412 * W - 50, 0] = image[
                                                                                                             h * 412 - 50:
                                                                                                             images.shape[
                                                                                                                 1],
                                                                                                             w * 412 - 50:
                                                                                                             images.shape[
                                                                                                                 2], 0]
                    else:
                        try:
                            cropped_img[0:images.shape[1] - 412 * H - 50, :, 0] = image[h * 412 - 50:images.shape[1],
                                                                                  w * 412 - 50:(w + 1) * 412 + 50, 0]
                        except:
                            cropped_img[0:images.shape[1] - 412 * H - 50, 0:images.shape[2] - 412 * (W - 1) - 50,
                            0] = image[h * 412 - 50:images.shape[1], w * 412 - 50:(w + 1) * 412 + 50, 0]
                else:
                    if w == -1:
                        cropped_img[:, 50:images.shape[2] + 50, 0] = image[h * 412 - 50:(h + 1) * 412 + 50,
                                                                     0:images.shape[2], 0]
                    elif w == 0:
                        # cropped_img[:, 50:512, 0] = image[h*412-50:(h+1)*412+50, 0:462, 0]
                        try:
                            cropped_img[:, 50:512, 0] = image[h * 412 - 50:(h + 1) * 412 + 50, 0:462, 0]
                        except:
                            cropped_img[0:images.shape[1] - 412 * H - 50 + 412, 50:512, 0] = image[h * 412 - 50:(
                                                                                                                        h + 1) * 412 + 50,
                                                                                             0:462, 0]
                    elif w == W - 1:
                        # cropped_img[:, 0:images.shape[2]-412*W-50, 0] = image[h*412-50:(h+1)*412+50, w*412-50:images.shape[2], 0]
                        try:
                            cropped_img[:, 0:images.shape[2] - 412 * W - 50, 0] = image[h * 412 - 50:(h + 1) * 412 + 50,
                                                                                  w * 412 - 50:images.shape[2], 0]
                        except:
                            cropped_img[0:images.shape[1] - 412 * H - 50 + 412, 0:images.shape[2] - 412 * W - 50,
                            0] = image[h * 412 - 50:(h + 1) * 412 + 50, w * 412 - 50:images.shape[2], 0]
                    else:
                        # cropped_img[:, :, 0] = image[h*412-50:(h+1)*412+50, w*412-50:(w+1)*412+50, 0]
                        try:
                            cropped_img[:, :, 0] = image[h * 412 - 50:(h + 1) * 412 + 50,
                                                   w * 412 - 50:(w + 1) * 412 + 50, 0]
                        except:
                            try:
                                cropped_img[:, 0:images.shape[2] - 412 * (W - 1) - 50, 0] = image[h * 412 - 50:(
                                                                                                                       h + 1) * 412 + 50,
                                                                                            w * 412 - 50:(
                                                                                                                 w + 1) * 412 + 50,
                                                                                            0]
                            except:
                                cropped_img[0:images.shape[1] - 412 * (H - 1) - 50, :, 0] = image[h * 412 - 50:(
                                                                                                                       h + 1) * 412 + 50,
                                                                                            w * 412 - 50:(
                                                                                                                 w + 1) * 412 + 50,
                                                                                            0]
                h = max(0, h)
                w = max(0, w)
                diveded_imgs[z * H * W + w * H + h] = cropped_img
                # print(z*H*W+ w*H+h)

    return diveded_imgs


def merge_imgs(imgs, original_image_shape):
    merged_imgs = np.zeros((original_image_shape[0], original_image_shape[1], original_image_shape[2], 1), np.float32)
    H = -(-original_image_shape[1] // 412)
    W = -(-original_image_shape[2] // 412)

    for z in range(original_image_shape[0]):
        for h in range(H):
            for w in range(W):

                if original_image_shape[1] < 412:
                    h = -1
                if original_image_shape[2] < 412:
                    w = -1

                # print(z*H*W+ max(w, 0)*H+max(h, 0))
                if h == -1:
                    if w == -1:
                        merged_imgs[z, 0:original_image_shape[1], 0:original_image_shape[2], 0] = imgs[
                                                                                                      z * H * W + 0 * H + 0][
                                                                                                  50:
                                                                                                  original_image_shape[
                                                                                                      1] + 50, 50:
                                                                                                               original_image_shape[
                                                                                                                   2] + 50,
                                                                                                  0]
                    elif w == 0:
                        merged_imgs[z, 0:original_image_shape[1], 0:412, 0] = imgs[z * H * W + w * H + 0][
                                                                              50:original_image_shape[1] + 50, 50:462,
                                                                              0]
                    elif w == W - 1:
                        merged_imgs[z, 0:original_image_shape[1], w * 412:original_image_shape[2], 0] = imgs[
                                                                                                            z * H * W + w * H + 0][
                                                                                                        50:
                                                                                                        original_image_shape[
                                                                                                            1] + 50, 50:
                                                                                                                     original_image_shape[
                                                                                                                         2] - 412 * W - 50,
                                                                                                        0]
                    else:
                        merged_imgs[z, 0:original_image_shape[1], w * 412:(w + 1) * 412, 0] = imgs[
                                                                                                  z * H * W + w * H + 0][
                                                                                              50:original_image_shape[
                                                                                                     1] + 50, 50:462, 0]
                elif h == 0:
                    if w == -1:
                        merged_imgs[z, 0:412, 0:original_image_shape[2], 0] = imgs[z * H * W + 0 * H + h][50:462,
                                                                              50:original_image_shape[2] + 50, 0]
                    elif w == 0:
                        merged_imgs[z, 0:412, 0:412, 0] = imgs[z * H * W + w * H + h][50:462, 50:462, 0]
                    elif w == W - 1:
                        merged_imgs[z, 0:412, w * 412:original_image_shape[2], 0] = imgs[z * H * W + w * H + h][50:462,
                                                                                    50:original_image_shape[
                                                                                           2] - 412 * W - 50, 0]
                    else:
                        merged_imgs[z, 0:412, w * 412:(w + 1) * 412, 0] = imgs[z * H * W + w * H + h][50:462, 50:462, 0]
                elif h == H - 1:
                    if w == -1:
                        merged_imgs[z, h * 412:original_image_shape[1], 0:original_image_shape[2], 0] = imgs[
                                                                                                            z * H * W + 0 * H + h][
                                                                                                        50:
                                                                                                        original_image_shape[
                                                                                                            1] - 412 * H - 50,
                                                                                                        50:
                                                                                                        original_image_shape[
                                                                                                            2] + 50, 0]
                    elif w == 0:
                        merged_imgs[z, h * 412:original_image_shape[1], 0:412, 0] = imgs[z * H * W + w * H + h][
                                                                                    50:original_image_shape[
                                                                                           1] - 412 * H - 50, 50:462, 0]
                    elif w == W - 1:
                        merged_imgs[z, h * 412:original_image_shape[1], w * 412:original_image_shape[2], 0] = imgs[
                                                                                                                  z * H * W + w * H + h][
                                                                                                              50:
                                                                                                              original_image_shape[
                                                                                                                  1] - 412 * H - 50,
                                                                                                              50:
                                                                                                              original_image_shape[
                                                                                                                  2] - 412 * W - 50,
                                                                                                              0]
                    else:
                        merged_imgs[z, h * 412:original_image_shape[1], w * 412:(w + 1) * 412, 0] = imgs[
                                                                                                        z * H * W + w * H + h][
                                                                                                    50:
                                                                                                    original_image_shape[
                                                                                                        1] - 412 * H - 50,
                                                                                                    50:462, 0]
                else:
                    if w == -1:
                        merged_imgs[z, h * 412:(h + 1) * 412, 0:original_image_shape[2], 0] = imgs[
                                                                                                  z * H * W + 0 * H + h][
                                                                                              50:462,
                                                                                              50:original_image_shape[
                                                                                                     2] + 50, 0]
                    elif w == 0:
                        merged_imgs[z, h * 412:(h + 1) * 412, 0:412, 0] = imgs[z * H * W + w * H + h][50:462, 50:462, 0]
                    elif w == W - 1:
                        merged_imgs[z, h * 412:(h + 1) * 412, w * 412:original_image_shape[2], 0] = imgs[
                                                                                                        z * H * W + w * H + h][
                                                                                                    50:462, 50:
                                                                                                            original_image_shape[
                                                                                                                2] - 412 * W - 50,
                                                                                                    0]
                    else:
                        merged_imgs[z, h * 412:(h + 1) * 412, w * 412:(w + 1) * 412, 0] = imgs[z * H * W + w * H + h][
                                                                                          50:462, 50:462, 0]

    print(merged_imgs.shape)
    return merged_imgs
