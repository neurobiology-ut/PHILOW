def dice(inputs, targets, smooth=1):
    inputs = inputs.view(-1)
    targets = targets.view(-1)

    intersection = (inputs * targets).sum()
    dice_score = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

    return dice_score