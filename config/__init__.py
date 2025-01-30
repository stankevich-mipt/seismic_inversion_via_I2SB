from .model import unet_ch64


def get_config_by_name(model_name):

    if model_name == "unet_ch64":
        return unet_ch64()
    else: 
        raise NotImplementedError