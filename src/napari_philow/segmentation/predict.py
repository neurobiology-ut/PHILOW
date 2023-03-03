import numpy as np
from torchvision.transforms import functional
from tqdm import tqdm

from .utils import add_margin


def pred_large_image(image, net, device, size):
    """
    Args:
        image (PIL image): input image
        net (torch.nn.Module): model
        device (str): e.g. 'cpu', 'cuda:0'
        size (int):  patch size

    Returns:
        ndarray: predicted probability map

    """
    x, y = image.size
    if max(x, y) > size:
        ps = size  # patch size
        ds = ps // 2  # d size
        ms = ds // 2  # margin size
        by = int(y % ps / 2) + ms
        bx = int(x % ps / 2) + ms
        temp_image = add_margin(image, by, bx, by, bx, 'black')
        temp_label = np.zeros((y + by * 2, x + bx * 2), np.float32)
        row = (y + by * 2) // ds - 1
        col = (x + bx * 2) // ds - 1
        for r in tqdm(range(row)):
            for c in range(col):
                temp = temp_image.crop((c * ds, r * ds, c * ds + ps, r * ds + ps))
                temp = functional.to_tensor(temp).unsqueeze(0).to(device)
                pred = net(temp).cpu().detach().numpy()
                temp_label[r * ds + ms:r * ds + ps - ms, c * ds + ms:c * ds + ps - ms] = pred[0, 0, ms:ps - ms,
                                                                                         ms:ps - ms]
        return temp_label[by:y + by, bx:x + bx]
    else:
        temp = functional.to_tensor(image).unsqueeze(0).to(device)
        pred = net(temp).cpu().detach().numpy()
        return pred
