import torch.optim as optim

def optimizers(model, config):
    """
    Returns the optimizer based on the configuration.

    Args:
        model (torch.nn.Module): The neural network model.
        config (dict): Configuration dictionary containing 'optimizer' and 'learning_rate' keys.

    Returns:
        torch.optim.Optimizer: Optimizer for training.
    """
    if config["optimizer"] == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=config["learning_rate"],
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0,
        )
    elif config["optimizer"] == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=config["learning_rate"],
            momentum=0.9,
        )
    else:
        raise ValueError("Invalid optimizer specified in config.")
    return optimizer
