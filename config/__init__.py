from .model import *

def get_config_by_name(model_name):

    if model_name == "ddpm_ch16":
        return ddpm_ch16()
    elif model_name == "ddpm_ch64":
        return ddpm_ch64()
    elif model_name == "i2sb_ch16":
        return i2sb_ch16()
    elif model_name == "i2sb_ch64":
        return i2sb_ch64()
    if model_name == "i2sb_ch16_cond":
        return i2sb_ch16_cond()
    elif model_name == "i2sb_ch64_cond":
        return i2sb_ch64_cond()
    else: 
        raise NotImplementedError