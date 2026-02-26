from monai.losses import DiceCELoss, DiceLoss

def loss_funtions(config, beta=200, gamma=400, total_epochs=600):
    if config["loss"] == "Dice":
        loss_function = DiceLoss(include_background=False,to_onehot_y=True, softmax=True)

    if config["loss"] == "DiceCE":
        loss_function = DiceCELoss(include_background=False, to_onehot_y=True, softmax=True, lambda_dice=1.0, lambda_ce=1.0)
    return loss_function, beta, gamma, total_epochs
