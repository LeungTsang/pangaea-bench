# Adapted from: https://github.com/zhu-xlab/SSL4EO-S12/tree/main

from functools import partial
from logging import Logger
from pathlib import Path

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block, PatchEmbed

from pangaea.encoders.base import Encoder

from .pos_embed import get_2d_sincos_pos_embed


class SSL4EO_MAE_OPTICAL_Encoder(Encoder):
    """
    Paper: https://arxiv.org/abs/2211.07044
    SSL4EO_MAE_OPTICAL_Encoder is a class for MAE trained on optical data.
    Args:
        encoder_weights (str | Path): Path to the encoder weights file.
        input_size (int): Size of the input image.
        input_bands (dict[str, list[str]]): Dictionary specifying the input bands.
        output_layers (int | list[int]): Layers from which to extract the output.
        embed_dim (int, optional): Dimension of the embedding. Default is 1024.
        patch_size (int, optional): Size of the patches. Default is 16.
        in_chans (int, optional): Number of input channels. Default is 3.
        depth (int, optional): Depth of the transformer. Default is 12.
        num_heads (int, optional): Number of attention heads. Default is 16.
        mlp_ratio (float, optional): Ratio of MLP hidden dimension to embedding dimension. Default is 4.0.
        norm_layer (callable, optional): Normalization layer. Default is partial(nn.LayerNorm, eps=1e-6).
    Methods:
        initialize_weights():
            Initializes the weights of the model.
        _init_weights(m):
            Initializes the weights of a given module.
        forward(image):
            Forward pass of the encoder.
        load_encoder_weights(logger: Logger):
            Loads the encoder weights from a checkpoint and logs any missing or incompatible parameters.
    """

    def __init__(
        self,
        mlp_ratio: float = 4.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    ):
        super().__init__(**kwargs)

        self.in_chans = sum([len(v) for v in self.input_bands.values()])

        self.patch_embed = PatchEmbed(self.input_size, self.patch_size, self.in_chans, self.embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, self.embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        self.blocks = nn.ModuleList(
            [
                Block(
                    self.embed_dim,
                    self.num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(self.depth)
            ]
        )
        # self.norm = norm_layer(embed_dim)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def simple_forward(self, image):
        # embed patches
        image = self.squeeze_temporal_dimension(image)

        x = self.patch_embed(image["optical"])

        # append cls token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        # apply Transformer blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.output_layers:
                # out = self.norm(x) if i == 11 else x
                out = self.naive_reshape_to_2d(x)
                output.append(out)

        return output

    def load_encoder_weights(self, logger: Logger) -> None:
        checkpoint = torch.load(self.encoder_weights, map_location="cpu")
        pretrained_model = checkpoint["model"]

        k = pretrained_model.keys()
        pretrained_encoder = {}
        incompatible_shape = {}
        missing = {}
        for name, param in self.named_parameters():
            if name not in k:
                missing[name] = param.shape
            elif pretrained_model[name].shape != param.shape:
                incompatible_shape[name] = (param.shape, pretrained_model[name].shape)
            else:
                pretrained_encoder[name] = pretrained_model[name]

        self.load_state_dict(pretrained_encoder)
        self.parameters_warning(missing, incompatible_shape, logger)


class SSL4EO_MAE_SAR_Encoder(SSL4EO_MAE_OPTICAL_Encoder):
    """
    Paper: https://arxiv.org/abs/2211.07044
    SSL4EO_MAE_SAR_Encoder is a class for MAE trained SAR data.
    Attributes:
        model_name (str): Name of the model.
        multi_temporal (bool): Indicates if the model handles multi-temporal data.
        output_dim (int): Dimension of the output embeddings.
    Methods:
        __init__(encoder_weights: str | Path, input_size: int, input_bands: dict[str, list[str]], output_layers: int | list[int], embed_dim: int = 1024, patch_size: int = 16, in_chans: int = 3, depth: int = 12, num_heads: int = 16, mlp_ratio: float = 4.0, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
            Initializes the SSL4EO_MAE_SAR_Encoder with the given parameters.
        forward(image):
            Forward pass of the encoder.
            Args:
                image (dict): Dictionary containing SAR image data with key 'sar'.
            Returns:
                list: List of output feature maps from specified layers.
    """

    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)

    def simple_forward(self, image):
        # embed patches
        image = self.squeeze_temporal_dimension(image)

        x = self.patch_embed(image["sar"])

        # append cls token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        # apply Transformer blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.output_layers:
                # out = self.norm(x) if i == 11 else x
                out = (
                    x[:, 1:]
                    .permute(0, 2, 1)
                    .view(
                        x.shape[0],
                        -1,
                        self.input_size // self.patch_size,
                        self.input_size // self.patch_size,
                    )
                    .contiguous()
                )
                output.append(out)

        return output
