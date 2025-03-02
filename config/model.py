import ml_collections


def ddpm_ch16():

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


def i2sb_ch16():
    return ddpm_ch16()


def i2sb_ch16_cond():
    config = i2sb_ch16()
    config.in_channels = 7
    return config


def ddpm_ch64():

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


def i2sb_ch64():
    return ddpm_ch64()


def i2sb_ch64_cond():

    config = i2sb_ch64()
    config.in_channels = 7
    
    return config