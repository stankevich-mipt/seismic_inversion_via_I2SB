import ml_collections

def unet_ch64():

    config = ml_collections.ConfigDict()

    config.in_channels = 6
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