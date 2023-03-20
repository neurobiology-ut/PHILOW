import glob
import os

import dask_image.imread
import numpy as np
from PIL import Image
from skimage import io
from tifffile import natural_sorted

from napari_philow.segmentation.predict import pred_large_image


def predict_and_save(dask_arr, net, out_dir_axis, size, device):
    """
    Args:
        dask_arr (dask.array.Array): 2D input image
        net (torch.nn.Module): model
        out_dir_axis (str): output directory for the prediction of the current axis
        size (int): patch size
        device (str): e.g. 'cpu', 'cuda:0'
    """
    for i in range(len(dask_arr)):
        pred = 255 * pred_large_image(Image.fromarray(dask_arr[i].compute()), net, device, size)
        io.imsave(os.path.join(out_dir_axis, str(i).zfill(6) + '.png'), pred.astype(np.uint8))


def predict_3ax(o_path, net, out_dir, size, device):
    os.makedirs(out_dir, exist_ok=True)
    out_dir_merge = os.path.join(out_dir, 'merged_prediction')
    os.makedirs(out_dir_merge, exist_ok=True)
    os.makedirs(f"{out_dir_merge}_raw", exist_ok=True)
    filenames = natural_sorted(glob.glob(os.path.join(o_path, '*.png')))
    xy_imgs = dask_image.imread.imread(os.path.join(o_path, '*.png'))
    yz_imgs = xy_imgs.transpose(2, 0, 1)
    zx_imgs = xy_imgs.transpose(1, 2, 0)
    predict_and_save(xy_imgs, net, os.path.join(out_dir, 'pred_xy'), size, device)
    predict_and_save(yz_imgs, net, os.path.join(out_dir, 'pred_yz'), size, device)
    predict_and_save(zx_imgs, net, os.path.join(out_dir, 'pred_zx'), size, device)
    pred_xy_imgs = dask_image.imread.imread(os.path.join(out_dir, 'pred_xy', '*png'))
    pred_yz_imgs = dask_image.imread.imread(os.path.join(out_dir, 'pred_yz', '*png'))
    pred_zx_imgs = dask_image.imread.imread(os.path.join(out_dir, 'pred_zx', '*png'))
    pred_yz_imgs_xy = pred_yz_imgs.transpose(1, 2, 0)
    pred_zx_imgs_xy = pred_zx_imgs.transpose(2, 0, 1)
    for i in range(len(pred_xy_imgs)):
        mito_img_ave = pred_xy_imgs[i] // 3 + pred_yz_imgs_xy[i] // 3 + pred_zx_imgs_xy[i] // 3
        mito_img_ave = mito_img_ave.compyte()
        img = np.where(mito_img_ave >= 127, 1, 0)
        io.imsave(f'{out_dir_merge}/{filenames[i]}', img.astype(np.uint8))
        img_ = np.where(mito_img_ave >= 127, mito_img_ave, 0)
        io.imsave(f'{out_dir_merge}_raw/{filenames[i]}', img_.astype(np.uint8))


def predict_1ax(ori_filenames, net, out_dir, size, device):
    """
    Args:
        ori_filenames (List[Path]):
        net (torch.nn.Module): model
        out_dir (str): output directory
        size (int):  patch size
        device (str): e.g. 'cpu', 'cuda:0'
    """
    os.makedirs(out_dir, exist_ok=True)
    out_dir_merge = os.path.join(out_dir, 'merged_prediction')
    os.makedirs(out_dir_merge, exist_ok=True)
    os.makedirs(f"{out_dir_merge}_raw", exist_ok=True)

    for filename in ori_filenames:
        image = Image.open(str(filename))
        mito_imgs_ave = 255 * pred_large_image(image, net, device, size)
        # threshed
        img = np.where(mito_imgs_ave >= 127, 1, 0)
        io.imsave(f'{out_dir_merge}/{filename.name}', img.astype('int32'))

        # raw
        img_ = np.where(mito_imgs_ave[:, :] >= 127, mito_imgs_ave, 0)
        io.imsave(f'{out_dir_merge}_raw/{filename.name}', img_.astype('uint8'))
