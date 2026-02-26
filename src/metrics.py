from monai.metrics import DiceMetric
import torch

def evaluation_metrics(config):
    if config["metrics"] == "dice":
        metrica = DiceMetric(include_background=True)#reduction="mean"
    return metrica


