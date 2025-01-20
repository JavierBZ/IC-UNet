from monai.networks.nets import UNet, AutoEncoder

def modelo_neuronal(config):
    """
    Returns the neural network model based on the configuration.

    Args:
        config (dict): Configuration dictionary containing 'modelo', 'clases', and 'capas_resnet' keys.

    Returns:
        torch.nn.Module: Neural network model.
    """
    if config["modelo"] == "unet":
        return UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=config["clases"],
            channels=(16, 32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2, 2),
            dropout=config["dropout"],
            num_res_units=config["capas_resnet"],
        ).cuda()
    elif config["modelo"] == "autoencoder":
        return AutoEncoder(
            spatial_dims=3,
            in_channels=1,
            out_channels=config["clases"],
            channels=(2, 4, 8),
            strides=(2, 2, 2),
        ).cuda()
    else:
        raise ValueError("Invalid model specified in config.")
