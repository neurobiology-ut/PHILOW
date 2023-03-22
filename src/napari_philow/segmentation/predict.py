import math

import numpy as np
from torchvision.transforms import functional
from tqdm import tqdm

from .utils import add_margin


def pred_large_image(image, net, device, size):
    """
    predict large image
    Args:
        image (PIL image): input image
        net (torch.nn.Module): model
        device (str): e.g. 'cpu', 'cuda:0'
        size (int):  patch size
    Returns:
        ndarray: predicted probability map

    """
    x, y = image.size

    # if image size is larger than patch size, split image into patches than predict
    if max(x, y) > size:
        ps = size  # patch size
        ds = ps // 2  # d size
        ms = ds // 2  # margin size
        print(f"patch size: {ps}, d ssize: {ds}, margin size: {ms}")
        if y < size:
            by = int(y % ps / 2) + ms + size - y
        else:
            by = int(y % ps / 2) + ms
        if x < size:
            bx = int(x % ps / 2) + ms + size - x
        else:
            bx = int(x % ps / 2) + ms
        temp_image = add_margin(image, by, bx, by, bx, 'black')
        temp_label = np.zeros((y + by * 2, x + bx * 2), np.float32)
        row = (y + by * 2) // ds - 1
        col = (x + bx * 2) // ds - 1
        for r in range(row):
            for c in range(col):
                temp = temp_image.crop((c * ds, r * ds, c * ds + ps, r * ds + ps))
                temp = functional.to_tensor(temp).unsqueeze(0).to(device)
                pred = net(temp).cpu().detach().numpy()
                temp_label[r * ds + ms:r * ds + ps - ms, c * ds + ms:c * ds + ps - ms] = pred[0, 0, ms:ps - ms,
                                                                                         ms:ps - ms]
        return temp_label[by:y + by, bx:x + bx]
    else:
        if x < size:
            dx = size - x
        else:
            dx = 0
        if y < size:
            dy = size - y
        else:
            dy = 0
        image = add_margin(image, math.floor(dy / 2), math.floor(dx / 2), math.ceil(dy / 2), math.ceil(dx / 2), 'black')
        temp = functional.to_tensor(image).unsqueeze(0).to(device)
        pred = net(temp).cpu().detach().numpy()
        return pred[0, 0, math.floor(dy / 2):math.floor(dy / 2) + y, math.floor(dx / 2):math.floor(dx / 2) + x]
