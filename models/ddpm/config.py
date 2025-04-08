import ml_collections

def ddpm_small():
    
    config = ml_collections.ConfigDict() 
    config.in_channels = 2
    config.model_channels = 16
    config.out_channels = 1
    config.num_res_blocks = 2
    config.attention_resolutions = (16,)
    config.dropout = 0.0
    config.channel_mult = (1, 2, 4)
    config.conv_resample = True
    config.dims = 2
    config.num_classes = None
    config.use_checkpoint = False
    config.use_fp16 = False
    config.num_heads = 4
    config.num_heads_upsample = -1
    config.num_head_channels = -1
    config.use_scale_shift_norm = True
    config.resblock_updown = False
    config.use_new_attention_order = False
    return config

def ddpm_small_cond():

    config = ddpm_small()
    config.in_channels = 7
    return config
    
def ddpm_large():

    config = ml_collections.ConfigDict()
    config.in_channels = 2
    config.model_channels = 64
    config.out_channels = 1
    config.num_res_blocks = 2
    config.attention_resolutions = (16, 8)
    config.dropout = 0.0
    config.channel_mult = (1, 2, 4, 8)
    config.conv_resample = True
    config.dims = 2
    config.num_classes = None
    config.use_checkpoint = False
    config.use_fp16 = False
    config.num_heads = 4
    config.num_heads_upsample = -1
    config.num_head_channels = -1
    config.use_scale_shift_norm = True
    config.resblock_updown = False
    config.use_new_attention_order = False
    return config

def ddpm_large_cond():

    config = ddpm_large()
    config.in_channels = 7
    return config

model_name_to_model_config = {
    'ddpm_small': ddpm_small,
    'ddpm_small_cond': ddpm_small_cond,
    'ddpm_large': ddpm_large,
    'ddpm_large_cond': ddpm_large_cond
}

def get_config_by_name(name: str):

    if name in model_name_to_model_config:
        return model_name_to_model_config[name]()
    else:
        raise NotImplementedError
