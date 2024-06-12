#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .spectralgpt import VisionTransformer
from .prithvi import MaskedAutoencoderViT
from .scalemae import ScaleMAE_baseline
from .croma import CROMA
from .remoteclip import RemoteCLIP
from .SSL4EO_mae import mae_vit_base_patch16_dec512d8b, mae_vit_huge_patch14_dec512d8b, mae_vit_large_patch16_dec512d8b, mae_vit_small_patch16_dec512d8b
from .SSL4EO_dino import vit_small
from .SSL4EO_moco import moco_vit_small
from .SSL4EO_data2vec import beit_small_patch16_224 

spectral_gpt_vit_base = VisionTransformer
prithvi_vit_base = MaskedAutoencoderViT
scale_mae_large = ScaleMAE_baseline
croma = CROMA
remote_clip = RemoteCLIP
ssl4eo_dino_small = vit_small
ssl4eo_moco_small = moco_vit_small
ssl4eo_data2vec_small = beit_small_patch16_224

#SSL4EO models
def choose_ssl4eo_mae(size = '384'):
    return {
        '384': mae_vit_small_patch16_dec512d8b,
        '768': mae_vit_base_patch16_dec512d8b,
        '1024': mae_vit_large_patch16_dec512d8b,
        '1280': mae_vit_huge_patch14_dec512d8b,
        
    }[size]
