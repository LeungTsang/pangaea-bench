# Adapted from: https://github.com/antofuller/CROMA

import itertools
import math
from logging import Logger
from pathlib import Path

import torch
from einops import rearrange
from torch import einsum, nn

from pangaea.encoders.base import Encoder


class CROMA_OPTICAL_Encoder(Encoder):
    """
    Paper: https://arxiv.org/pdf/2311.00566
    CROMA_OPTICAL_Encoder is a class for extracting features from optical images using CROMA.
    Attributes:
        **kwargs : base encoder parameters.
            model_name (str): name of the model.
            encoder_weights (str | Path): path to the encoder weights.
            download_url (str): url to download the model.
            input_size (int): expected input_size of the transformer.
            patch_size (int): patch size of the transformer.
            embed_dim (int): embedding dimension of the transformer.
            depth (int): number of layers.
            num_heads (int): number of attention heads.
            has_cls_token (bool): whether the transformer has a CLS token or not.
            pyramid_features (bool): whether the encoder outputs multi-scale features.
            multi_temporal (bool): whether the model is multi-temporal or not.
            multi_temporal_fusion (bool): whether the model is multi-temporal fusion or not.
            naive_multi_forward_mode (str): for non-multi-temporal models: loop: encode images one by one; batch: in one forward
            input_bands (dict[str, list[str]]): input bands for each modality.
            output_layers (list[int]): output layer indices for multi-scale features.
            output_dim (int | Sequence[int]): output dimension(s) of the transformer.
    Methods:
        __init__(**kwargs):
            Initializes the CROMA_OPTICAL_Encoder with the given parameters.
        forward(image):
            Performs a forward pass of the encoder on the given image.
        load_encoder_weights(logger: Logger) -> None:
            Loads the pretrained weights into the encoder and logs any missing or incompatible parameters.
    """

    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.num_patches = int((self.input_size / self.patch_size) ** 2)
        self.s2_channels = len(self.input_bands['optical'].keys())  # fixed at 12 multispectral optical channels
        self.attn_bias = get_2dalibi(
            num_heads=self.num_heads, num_patches=self.num_patches
        )

        self.s2_encoder = ViT(
            dim=self.embed_dim, depth=self.depth, in_channels=self.s2_channels
        )

    def simple_forward(self, image):

        image = self.squeeze_temporal_dimension(image)

        output = self.s2_encoder(
            image["optical"],
            self.attn_bias.to(image["optical"].device),
            self.output_layers,
        )  # (bsz, num_patches, encoder_dim)

        output = [self.naive_reshape_to_2d(out) for out in output]

        return output

    def load_encoder_weights(self, logger: Logger) -> None:
        pretrained_model = torch.load(self.encoder_weights, map_location="cpu")[
            "s2_encoder"
        ]
        k = pretrained_model.keys()
        pretrained_encoder = {}
        incompatible_shape = {}
        missing = {}
        for name, param in self.s2_encoder.named_parameters():
            if name not in k:
                missing[name] = param.shape
            elif pretrained_model[name].shape != param.shape:
                incompatible_shape[name] = (param.shape, pretrained_model[name].shape)
            else:
                pretrained_encoder[name] = pretrained_model[name]

        self.s2_encoder.load_state_dict(pretrained_encoder, strict=False)
        self.parameters_warning(missing, incompatible_shape, logger)


class CROMA_SAR_Encoder(Encoder):
    """
    Paper: https://arxiv.org/pdf/2311.00566
    CROMA_SAR_Encoder is a class for extracting features from SAR images using CROMA.
    Attributes:
        **kwargs : base encoder parameters.
            model_name (str): name of the model.
            encoder_weights (str | Path): path to the encoder weights.
            download_url (str): url to download the model.
            input_size (int): expected input_size of the transformer.
            patch_size (int): patch size of the transformer.
            embed_dim (int): embedding dimension of the transformer.
            depth (int): number of layers.
            num_heads (int): number of attention heads.
            has_cls_token (bool): whether the transformer has a CLS token or not.
            pyramid_features (bool): whether the encoder outputs multi-scale features.
            multi_temporal (bool): whether the model is multi-temporal or not.
            multi_temporal_fusion (bool): whether the model is multi-temporal fusion or not.
            naive_multi_forward_mode (str): for non-multi-temporal models: loop: encode images one by one; batch: in one forward
            input_bands (dict[str, list[str]]): input bands for each modality.
            output_layers (list[int]): output layer indices for multi-scale features.
            output_dim (int | Sequence[int]): output dimension(s) of the transformer.
    Methods:
        __init__(**kwargs):
            Initializes the CROMA_SAR_Encoder with the given parameters.
        forward(image: dict) -> list[Tensor]:
            Forward pass of the encoder. Processes the input SAR image and returns the encoded output.
        load_encoder_weights(logger: Logger) -> None:
            Loads the pretrained weights into the encoder and logs any missing or incompatible parameters.
    """

    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)


        self.num_patches = int((self.input_size / self.patch_size) ** 2)
        self.s1_channels = len(self.input_bands['sar'].keys())
        self.attn_bias = get_2dalibi(
            num_heads=self.num_heads, num_patches=self.num_patches
        )

        self.s1_encoder = ViT(
            dim=self.embed_dim,
            depth=int(self.depth / 2),
            in_channels=self.s1_channels,
        )

    def simple_forward(self, image):
        # output = []
        image = self.squeeze_temporal_dimension(image)

        output = self.s1_encoder(
            image["sar"],
            self.attn_bias.to(image["sar"].device),
            self.output_layers,
        )  # (bsz, num_patches, encoder_dim)

        output = [self.naive_reshape_to_2d(out) for out in output]

        return output

    def load_encoder_weights(self, logger: Logger) -> None:
        pretrained_model = torch.load(self.encoder_weights, map_location="cpu")[
            "s1_encoder"
        ]
        k = pretrained_model.keys()
        pretrained_encoder = {}
        incompatible_shape = {}
        missing = {}
        for name, param in self.s1_encoder.named_parameters():
            if name not in k:
                missing[name] = param.shape
            elif pretrained_model[name].shape != param.shape:
                incompatible_shape[name] = (param.shape, pretrained_model[name].shape)
            else:
                pretrained_encoder[name] = pretrained_model[name]

        self.s1_encoder.load_state_dict(pretrained_encoder, strict=False)
        self.parameters_warning(missing, incompatible_shape, logger)


class CROMA_JOINT_Encoder(Encoder):
    """
    Paper: https://arxiv.org/pdf/2311.00566
    CROMA_JOINT_Encoder is a class for extracting features from optical and SAR images using CROMA.
    Attributes:
        **kwargs : base encoder parameters.
            model_name (str): name of the model.
            encoder_weights (str | Path): path to the encoder weights.
            download_url (str): url to download the model.
            input_size (int): expected input_size of the transformer.
            patch_size (int): patch size of the transformer.
            embed_dim (int): embedding dimension of the transformer.
            depth (int): number of layers.
            num_heads (int): number of attention heads.
            has_cls_token (bool): whether the transformer has a CLS token or not.
            pyramid_features (bool): whether the encoder outputs multi-scale features.
            multi_temporal (bool): whether the model is multi-temporal or not.
            multi_temporal_fusion (bool): whether the model is multi-temporal fusion or not.
            naive_multi_forward_mode (str): for non-multi-temporal models: loop: encode images one by one; batch: in one forward
            input_bands (dict[str, list[str]]): input bands for each modality.
            output_layers (list[int]): output layer indices for multi-scale features.
            output_dim (int | Sequence[int]): output dimension(s) of the transformer.
    Methods:
        __init__(**kwargs):
            Initializes the CROMA_JOINT_Encoder with the given parameters.
        forward(image: dict[str, Tensor]) -> list[Tensor]:
            Forward pass of the encoder. Takes a dictionary with SAR and optical images and returns the encoded output.
        load_encoder_weights(logger: Logger) -> None:
            Loads the pretrained weights for the encoder and logs any missing or incompatible parameters.
    """

    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.num_patches = int((self.input_size / self.patch_size) ** 2)
        self.s1_channels = len(self.input_bands['sar'].keys())
        self.s2_channels = len(self.input_bands['optical'].keys())
        self.attn_bias = get_2dalibi(
            num_heads=self.num_heads, num_patches=self.num_patches
        )

        self.s1_encoder = ViT(
            dim=self.embed_dim,
            depth=int(self.depth / 2),
            in_channels=self.s1_channels,
        )
        self.s2_encoder = ViT(
            dim=self.embed_dim, depth=self.depth, in_channels=self.s2_channels
        )
        self.cross_encoder = BaseTransformerCrossAttn(
            dim=self.embed_dim,
            depth=int(self.depth / 2),
            num_heads=self.num_heads,
        )

    def simple_forward(self, image):
        attn_bias = self.attn_bias.to(image["optical"].device)
        SAR_encodings = self.s1_encoder(
            image["sar"], attn_bias
        )  # (bsz, num_patches, encoder_dim)
        optical_encodings = self.s2_encoder(
            image["optical"], attn_bias
        )  # (bsz, num_patches, encoder_dim)
        output = self.cross_encoder(
            x=SAR_encodings,
            context=optical_encodings,
            relative_position_bias=attn_bias,
            output_layers=self.output_layers,
        )

        output = [self.naive_reshape_to_2d(out) for out in output]

        return output

    def load_encoder_weights(self, logger: Logger) -> None:
        pretrained_model = torch.load(self.encoder_weights, map_location="cpu")
        combined_state_dict = {}
        for prefix, module in pretrained_model.items():
            for k, v in module.items():
                combined_state_dict[
                    prefix.replace("joint_encoder", "cross_encoder") + "." + k
                ] = v

        pretrained_model = combined_state_dict

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

        self.load_state_dict(pretrained_encoder, strict=False)
        self.parameters_warning(missing, incompatible_shape, logger)


def get_2dalibi(num_heads, num_patches):
    # inspired by: https://github.com/ofirpress/attention_with_linear_biases
    points = list(
        itertools.product(
            range(int(math.sqrt(num_patches))), range(int(math.sqrt(num_patches)))
        )
    )

    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return (
                get_slopes_power_of_2(closest_power_of_2)
                + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
            )

    slopes = torch.Tensor(get_slopes(num_heads)).unsqueeze(1)
    idxs = []
    for p1 in points:
        for p2 in points:
            dist = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
            idxs.append(dist * slopes * -1)
    all_bias = torch.cat(idxs, dim=1)
    return all_bias.view(1, num_heads, num_patches, num_patches)


class FFN(nn.Module):
    def __init__(
        self,
        dim,
        mult=4,
        dropout=0.0,
    ):
        super().__init__()
        inner_dim = int(dim * mult)

        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),  # (BSZ, num_patches, inner_dim)
            nn.GELU(),  # (BSZ, num_patches, inner_dim)
            nn.Dropout(dropout),  # (BSZ, num_patches, inner_dim)
            nn.Linear(inner_dim, dim),  # (BSZ, num_patches, dim)
        )
        self.input_norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.input_norm(x)  # (BSZ, num_patches, dim)
        return self.net(x)  # (BSZ, num_patches, dim)


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        dropout=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        assert dim % num_heads == 0, "dim must be evenly divisible by num_heads"
        dim_head = int(dim / num_heads)
        self.scale = dim_head**-0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.input_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, relative_position_bias):
        x = self.input_norm(x)  # (BSZ, num_patches, dim)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)  # (BSZ, num_patches, dim)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads), (q, k, v)
        )  # (BSZ, num_heads, num_patches, dim_head)

        attention_scores = (
            einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        )  # (BSZ, num_heads, num_patches, num_patches)
        attention_scores = (
            attention_scores + relative_position_bias
        )  # (BSZ, num_heads, num_patches, num_patches)

        attn = attention_scores.softmax(
            dim=-1
        )  # (BSZ, num_heads, num_patches, num_patches)
        attn = self.dropout(attn)  # (BSZ, num_heads, num_patches, num_patches)

        out = einsum(
            "b h i j, b h j d -> b h i d", attn, v
        )  # (BSZ, num_heads, num_patches, dim_head)
        out = rearrange(out, "b h n d -> b n (h d)")  # (BSZ, num_patches, dim)
        return self.to_out(out)  # (BSZ, num_patches, dim)


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        dropout=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        assert dim % num_heads == 0, "dim must be evenly divisible by num_heads"
        dim_head = int(dim / num_heads)
        self.scale = dim_head**-0.5

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        self.to_out = nn.Linear(dim, dim)
        self.input_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context, relative_position_bias):
        x = self.input_norm(x)  # (BSZ, num_patches, dim)
        context = self.input_norm(context)  # (BSZ, num_patches, dim)

        q = self.to_q(x)  # (BSZ, num_patches, dim)
        k = self.to_k(context)  # (BSZ, num_patches, dim)
        v = self.to_v(context)  # (BSZ, num_patches, dim)

        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads), (q, k, v)
        )  # (BSZ, num_heads, num_patches, dim_head)

        attention_scores = (
            einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        )  # (BSZ, num_heads, num_patches, num_patches)
        attention_scores = (
            attention_scores + relative_position_bias
        )  # (BSZ, num_heads, num_patches, num_patches)

        attn = attention_scores.softmax(
            dim=-1
        )  # (BSZ, num_heads, num_patches, num_patches)
        attn = self.dropout(attn)  # (BSZ, num_heads, num_patches, num_patches)

        out = einsum(
            "b h i j, b h j d -> b h i d", attn, v
        )  # (BSZ, num_heads, num_patches, dim_head)
        out = rearrange(out, "b h n d -> b n (h d)")  # (BSZ, num_patches, dim)
        return self.to_out(out)  # (BSZ, num_patches, dim)


class BaseTransformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        num_heads=8,
        attn_dropout=0.0,
        ff_dropout=0.0,
        ff_mult=4,
        final_norm=True,
    ):
        super().__init__()
        self.final_norm = final_norm
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim=dim, num_heads=num_heads, dropout=attn_dropout),
                        FFN(dim=dim, mult=ff_mult, dropout=ff_dropout),
                    ]
                )
            )

        if self.final_norm:
            self.norm_out = nn.LayerNorm(dim)

    def forward(self, x, relative_position_bias=False, output_layers=None):
        output = []
        for i, layer in enumerate(self.layers):
            self_attn, ffn = layer
            x = self_attn(x, relative_position_bias) + x  # (BSZ, num_patches, dim)
            x = ffn(x) + x  # (BSZ, num_patches, dim)

            if output_layers is not None and i in output_layers:
                if self.final_norm and i == len(self.layers) - 1:
                    x = self.norm_out(x)
                output.append(x)

        if output_layers is None:
            if self.final_norm:
                return self.norm_out(x)
            else:
                return x
        else:
            return output



class BaseTransformerCrossAttn(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        num_heads=8,
        attn_dropout=0.0,
        ff_dropout=0.0,
        ff_mult=4,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim=dim, num_heads=num_heads, dropout=attn_dropout),
                        CrossAttention(
                            dim=dim, num_heads=num_heads, dropout=attn_dropout
                        ),
                        FFN(dim=dim, mult=ff_mult, dropout=ff_dropout),
                    ]
                )
            )

        self.norm_out = nn.LayerNorm(dim)

    def forward(self, x, context, relative_position_bias, output_layers=None):
        output = []

        for i, layer in enumerate(self.layers):
            self_attn, cross_attn, ffn = layer
            x = self_attn(x, relative_position_bias) + x  # (BSZ, num_patches, dim)
            x = (
                cross_attn(x, context, relative_position_bias) + x
            )  # (BSZ, num_patches, dim)
            x = ffn(x) + x  # (BSZ, num_patches, dim)
            if output_layers is not None and i in output_layers:
                if i == len(self.layers) - 1:
                    x = self.norm_out(x)
                output.append(x)

        if output_layers is None:
            return self.norm_out(x)
        else:
            return output


class ViT(nn.Module):
    def __init__(self, dim, depth, in_channels):
        super().__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.dim = dim
        self.num_heads = 16  # always 16, for base and large models
        self.patch_size = 8  # always 8, for base and large models

        pixels_per_patch = int(self.patch_size * self.patch_size * in_channels)
        self.linear_input = nn.Linear(pixels_per_patch, self.dim)
        self.transformer = BaseTransformer(
            dim=self.dim,
            depth=self.depth,
            num_heads=self.num_heads,
        )

    def forward(self, imgs, attn_bias, output_layers=None):
        x = rearrange(
            imgs,
            "b c (h i) (w j) -> b (h w) (c i j)",
            i=self.patch_size,
            j=self.patch_size,
        )
        # x is shape -> (bsz, num_patches, self.channels*self.patch_size*self.patch_size)

        x = self.linear_input(x)  # (bsz, num_patches, dim)
        output = self.transformer(
            x, relative_position_bias=attn_bias, output_layers=output_layers
        )

        return output
