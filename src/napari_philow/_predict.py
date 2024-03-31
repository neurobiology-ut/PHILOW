import glob
import os

import dask_image.imread
import dask
import numpy as np
from PIL import Image
from skimage import io
from tifffile import natural_sorted
from tqdm import tqdm

from napari_philow.segmentation.predict import pred_large_image
from napari_philow._utils import renormalize_8bit


def predict_and_save(dask_arr, net, out_dir_axis, size, device):
    """
    Args:
        dask_arr (dask.array.Array): 3D input image
        net (torch.nn.Module): model
        out_dir_axis (str): output directory for the prediction of the current axis
        size (int): patch size
        device (str): e.g. 'cpu', 'cuda:0'
    """
    for i in tqdm(range(len(dask_arr))):
        pred = 255 * pred_large_image(Image.fromarray(dask_arr[i].compute()), net, device, size)
        io.imsave(os.path.join(out_dir_axis, str(i).zfill(6) + '.png'), pred.astype(np.uint8))


def predict_3ax(o_path, net, out_dir, size, device):
    """
    predict 3 axes (TAP) and merge the prediction
    Args:
        o_path (str): original image directory path
        net (torch.nn.Module): model
        out_dir (str): output directory
        size (int): patch size
        device (str): e.g. 'cpu', 'cuda:0'
    """
    os.makedirs(out_dir, exist_ok=True)
    out_dir_merge = os.path.join(out_dir, 'merged_prediction')
    os.makedirs(out_dir_merge, exist_ok=True)
    os.makedirs(f"{out_dir_merge}_raw", exist_ok=True)
    filenames = natural_sorted(glob.glob(os.path.join(o_path, '*.png')))
    xy_imgs = dask_image.imread.imread(os.path.join(o_path, '*.png'))
    xy_imgs = dask.array.asarray([ renormalize_8bit(xy_imgs[z]) for z in range(xy_imgs.shape[0]) ])
    print(f"xy_imgs.shape: {xy_imgs.shape}")
    yz_imgs = xy_imgs.transpose(2, 0, 1)
    zx_imgs = xy_imgs.transpose(1, 2, 0)
    os.makedirs(os.path.join(out_dir, 'pred_xy'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'pred_yz'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'pred_zx'), exist_ok=True)
    predict_and_save(xy_imgs, net, os.path.join(out_dir, 'pred_xy'), size, device)
    predict_and_save(yz_imgs, net, os.path.join(out_dir, 'pred_yz'), size, device)
    predict_and_save(zx_imgs, net, os.path.join(out_dir, 'pred_zx'), size, device)
    pred_xy_imgs = dask_image.imread.imread(os.path.join(out_dir, 'pred_xy', '*png'))
    pred_yz_imgs = dask_image.imread.imread(os.path.join(out_dir, 'pred_yz', '*png'))
    pred_zx_imgs = dask_image.imread.imread(os.path.join(out_dir, 'pred_zx', '*png'))
    print(pred_xy_imgs.shape)
    print(pred_yz_imgs.shape)
    print(pred_zx_imgs.shape)
    pred_yz_imgs_xy = pred_yz_imgs.transpose(1, 2, 0)
    pred_zx_imgs_xy = pred_zx_imgs.transpose(2, 0, 1)
    for i in tqdm(range(len(pred_xy_imgs))):
        mito_img_ave = pred_xy_imgs[i].compute() // 3 + pred_yz_imgs_xy[i].compute() // 3 + pred_zx_imgs_xy[i].compute() // 3
        img = np.where(mito_img_ave >= 127, 1, 0)
        io.imsave(f'{out_dir_merge}/{os.path.basename(filenames[i])}', img.astype(np.uint8))
        img_ = np.where(mito_img_ave >= 127, mito_img_ave, 0)
        io.imsave(f'{out_dir_merge}_raw/{os.path.basename(filenames[i])}', img_.astype(np.uint8))


def predict_1ax(ori_filenames, net, out_dir, size, device, mask_dir=None, out_channel=None):
    """
    predict 1 axis and merge the prediction
    Args:
        ori_filenames (List[Path]):
        net (torch.nn.Module): model
        out_dir (str): output directory
        size (int):  patch size
        device (str): e.g. 'cpu', 'cuda:0'
        mask_dir (str): dir path for mask which is used for mask original image before prediction
        out_channel (list): list of channel index to output
    """
    os.makedirs(out_dir, exist_ok=True)
    out_dir_merge = os.path.join(out_dir, 'merged_prediction')
    os.makedirs(out_dir_merge, exist_ok=True)
    os.makedirs(f"{out_dir_merge}_raw", exist_ok=True)

    for filename in ori_filenames:
        image = Image.open(str(filename))
        image_arr = np.array(image)
        if mask_dir is not None:
            mask = Image.open(f"{mask_dir}/{filename.name}")
            mask = np.array(mask)
            image_arr = np.where(mask == 0, 0, image_arr)
        image = renormalize_8bit(image_arr)
        image = Image.fromarray(image)
        if mask_dir is not None:
            pred_imgs_ave = 255 * pred_large_image(image, net, device, size, is_3class=True)
            mask = np.concatenate([mask[..., np.newaxis], mask[..., np.newaxis], mask[..., np.newaxis]], axis=-1)
            pred_imgs_ave = np.where(mask == 0, 0, pred_imgs_ave) # 1classのみ
        else:
            pred_imgs_ave = 255 * pred_large_image(image, net, device, size)
        # threshed
        if out_channel is not None:
            pred_imgs_ave = pred_imgs_ave[:, :, out_channel]
        img = np.where(pred_imgs_ave >= 127, 1, 0)
        io.imsave(f"{out_dir_merge}/{filename.name}", img.astype("uint8"))

        # raw
        img_ = np.where(pred_imgs_ave >= 127, pred_imgs_ave, 0)
        io.imsave(f'{out_dir_merge}_raw/{filename.name}', img_.astype('uint8'))
