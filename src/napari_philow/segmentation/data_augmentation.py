import numpy as np
import skimage
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, anno_class_img):
        for t in self.transforms:
            img, anno_class_img = t(img, anno_class_img)
        return img, anno_class_img


class RandomRotation(object):
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, img, anno_class_img):
        # 回転角度を決める
        rotate_angle = (np.random.uniform(self.angle[0], self.angle[1]))

        # 回転
        img = img.rotate(rotate_angle, Image.BILINEAR)
        anno_class_img = anno_class_img.rotate(rotate_angle, Image.NEAREST)

        return img, anno_class_img


class RandomBrightness(object):
    def __call__(self, img, anno_class_img):
        return transforms.ColorJitter(brightness=0.5)(img), anno_class_img


class RandomCrop(object):
    """randomにcropするクラス"""

    def __init__(self, size):
        self.size = size

    def __call__(self, img, anno_class_img):
        i, j, h, w = transforms.RandomCrop.get_params(img, output_size=(self.size, self.size))
        img = functional.crop(img, i, j, h, w)
        anno_class_img = functional.crop(anno_class_img, i, j, h, w)
        return img, anno_class_img


class Resize(object):
    """引数input_sizeに大きさを変形するクラス"""

    def __init__(self, input_size):
        self.input_size = input_size

    def __call__(self, img, anno_class_img):
        img = img.resize((self.input_size, self.input_size),
                         Image.BICUBIC)
        anno_class_img = anno_class_img.resize(
            (self.input_size, self.input_size), Image.NEAREST)

        return img, anno_class_img


class RandomGaussianBlur(object):
    def __call__(self, img, anno_class_img):
        if np.random.randint(2):
            img = transforms.GaussianBlur(kernel_size=5)(img)
        return img, anno_class_img


class RandomNoise(object):
    def __call__(self, img, anno_class_img):
        if np.random.randint(2):
            img = Image.fromarray((255 * skimage.util.random_noise(np.array(img), mode='s&p')).astype(np.uint8))
        return img, anno_class_img


class RondomRotateShiftScale(object):
    def __init__(self, angle, height_range, width_range, scale_range, img_size):
        """
        Args:
            angle (sequence or number):
            height_range (float):
            width_range (float):
            scale_range (list[int]):
            img_size (list[int]): image size [width, height]
        """
        self.angle = angle
        self.height_range = height_range
        self.width_range = width_range
        self.scale_range = scale_range
        self.img_size = img_size

    def __call__(self, img, mask):
        """
        Args:
            img (PIL Image or Tensor):
            mask (PIL Image or Tensor):
        Returns:
            Tuple of PIL Image or Tensor: Transformed image and mask
        """
        angle, translations, scale, shear = transforms.RandomAffine.get_params(self.angle, translate=[self.width_range,
                                                                                             self.height_range],
                                                                               scale_ranges=self.scale_range,
                                                                               shears=None, img_size=self.img_size)
        img = functional.affine(img, angle, translations, scale, shear)
        mask = functional.affine(mask, angle, translations, scale, shear)
        return img, mask


class RandomHFlip(object):
    def __call__(self, img, mask):
        if np.random.randint(2):
            functional.hflip(img)
            functional.hflip(mask)
        return img, mask


class RandomVFlip(object):
    def __call__(self, img, mask):
        if np.random.randint(2):
            functional.vflip(img)
            functional.vflip(mask)
        return img, mask
