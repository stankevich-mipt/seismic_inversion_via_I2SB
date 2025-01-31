from .model import *


def get_config_by_name(model_name):

    if model_name == "unet_ch32":
        return unet_ch32()
    elif model_name == "unet_ch64":
        return unet_ch64()
    else: 
        raise NotImplementedError