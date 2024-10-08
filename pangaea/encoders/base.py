import os
import urllib.error
import urllib.request
from logging import Logger
from pathlib import Path

import gdown
import torch
import torch.nn as nn
import tqdm

from typing import Callable, Dict, List, Optional, Sequence, Union, Tuple

class DownloadProgressBar:
    def __init__(self, text="Downloading..."):
        self.pbar = None
        self.text = text

    def __call__(self, block_num, block_size, total_size):
        if self.pbar is None:
            self.pbar = tqdm.tqdm(
                desc=self.text,
                total=total_size,
                unit="b",
                unit_scale=True,
                unit_divisor=1024,
            )

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded - self.pbar.n)
        else:
            self.pbar.close()
            self.pbar = None


class Encoder(nn.Module):
    """Base class for encoder."""

    def __init__(
        self,
        model_name: str,
        encoder_weights: str | Path,
        download_url: str,
        input_size: int,
        patch_size: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        has_cls_token: bool,
        pyramid_features: bool,
        multi_temporal: bool,
        multi_temporal_fusion: bool,
        naive_multi_forward_mode: str,
        input_bands: dict[str, list[str]],
        output_layers: list[int],
        output_dim: int | Sequence[int],
        **kwargs
    ) -> None:
        """Initialize the Encoder.

        Args:
            model_name (str): name of the model.
            encoder_weights (str | Path): path to the encoder weights.
            download_url (str): url to download the model.
            input_size (int): expected input_size of the transformer.
            patch_size (int): patch size of the transformer.
            embed_dim (int): embedding dimension of the transformer.
            depth (int): number of layers.
            has_cls_token (bool): whether the transformer has a CLS token or not.
            pyramid_features (bool): whether the encoder outputs multi-scale features.
            multi_temporal (bool): whether the model is multi-temporal or not.
            multi_temporal_fusion (bool): whether the model is multi-temporal fusion or not.
            naive_multi_forward_mode (str): for non-multi-temporal models: loop: encode images one by one; batch: in one forward
            input_bands (dict[str, list[str]]): input bands for each modality.
            output_layers (list[int]): output layer indices for multi-scale features.
            output_dim (int | Sequence[int]): output dimension(s) of the transformer.
            **kwargs (dict): additional arguments.
        """
        super().__init__()
        self.model_name = model_name
        self.input_bands = input_bands
        self.in_chans = sum([len(v) for v in self.input_bands.values()])
        self.input_size = input_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.has_cls_token = has_cls_token
        self.output_layers = output_layers
        self.output_dim = [output_dim for _ in output_layers] if isinstance(output_dim, int) else output_dim
        self.encoder_weights = encoder_weights
        self.pyramid_features = pyramid_features
        self.multi_temporal = multi_temporal
        self.multi_temporal_fusion = multi_temporal_fusion
        self.naive_multi_forward_mode = naive_multi_forward_mode
        self.download_url = download_url
        self.__dict__.update(kwargs)

        # download_model if necessary
        self.download_model()

    def load_encoder_weights(self, logger: Logger) -> None:
        """Load the encoder weights.

        Args:
            logger (Logger): logger to log the information.

        Raises:
            NotImplementedError: raise if the method is not implemented.
        """
        raise NotImplementedError

    def parameters_warning(
        self,
        missing: dict[str, torch.Size],
        incompatible_shape: dict[str, tuple[torch.Size, torch.Size]],
        logger: Logger,
    ) -> None:
        """Print warning messages for missing or incompatible parameters

        Args:
            missing (dict[str, torch.Size]): list of missing parameters.
            incompatible_shape (dict[str, tuple[torch.Size, torch.Size]]): list of incompatible parameters.
            logger (Logger): logger to log the information.
        """
        if missing:
            logger.warning(
                "Missing parameters:\n"
                + "\n".join("%s: %s" % (k, v) for k, v in sorted(missing.items()))
            )
        if incompatible_shape:
            logger.warning(
                "Incompatible parameters:\n"
                + "\n".join(
                    "%s: expected %s but found %s" % (k, v[0], v[1])
                    for k, v in sorted(incompatible_shape.items())
                )
            )

    def freeze(self) -> None:
        """Freeze encoder's parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def naive_reshape_to_2d(self, x, feat_size=None):
        """B L C to B C (T) H W"""
        if feat_size is None:
            feat_size = self.input_size // self.patch_size
        if self.has_cls_token:
            x = x[:, 1:]
        x = x.transpose(1, 2)
        x = x.view(
            x.shape[0],
            x.shape[1],
            -1,
            feat_size,
            feat_size
        ).squeeze(2)
        return x.contiguous()

    def simple_forward(self, img: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        """Compute the forward pass of the encoder.

        Args:
            x (torch.Tensor): input image.

        Raises:
            NotImplementedError: raise if the method is not implemented.

        Returns:
            torch.Tensor: embedding generated by the encoder.
        """
        raise NotImplementedError

    def forward(self, image: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        b, c, t, h, w = image[list(image.keys())[0]].shape
        if self.multi_temporal:
            return self.simple_forward(image)
        else:
            if t == 1:
                return self.simple_forward(image)
            else:
                return self.naive_multi_temporal_forward(image)


    def enforce_single_temporal(self):
        self.multi_temporal = False
        self.multi_temporal_fusion = False

    def naive_multi_temporal_forward(self, image: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        b, c, t, h, w = image[list(image.keys())[0]].shape
        if self.naive_multi_forward_mode == 'batch':
            # B, C, T, H, W -> B*T, C, H, W
            image = {k: v.transpose(1, 2).contiguous().view(-1, c, 1, h, w) for k, v in image.items()}
            feats = self.forward(image)
            # B*T, C, H, W  -> B, C, T, H, W
            feats = [f.view(b, -1, f.shape[1], f.shape[2], f.shape[3]).transpose(1, 2) for f in feats]
        elif self.naive_multi_forward_mode == 'loop':
            feats = []
            for i in range(t):
                feats.append(self.forward({k: v[:, :, i, :, :].unsqueeze(2) for k, v in image.items()}))
            # [[img1's layer1, img1's layer2, ...], [img2's layer1, img2's layer2, ...], ...]
            # -> [[img1's layer1, img2's layer1, ...], [img1's layer2, img2's layer2, ...], ...]
            feats = [list(i) for i in zip(*feats)]
            # B, C, T, H, W
            feats = [torch.stack(feat_layers, dim=2) for feat_layers in feats]
        else:
            raise NotImplementedError("only support multi-temporal forward mode 'loop' and 'batch'")

        return feats

    @staticmethod
    def squeeze_temporal_dimension(image: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        for k, v in image.items():
            if v.dim() == 5:
                if v.shape[2] == 1:
                    image[k] = v.squeeze(2)
                else:
                    raise ValueError(f"Cannot squeeze temporal dimension {v.shape[2]} other than 1.")
            elif v.dim() != 4:
                raise ValueError(f"Unknown image shape other than (B C H W) and (B C T H W).")

        return image


    def download_model(self) -> None:
        if self.download_url and not os.path.isfile(self.encoder_weights):
            # TODO: change this path
            os.makedirs("pretrained_models", exist_ok=True)

            pbar = DownloadProgressBar(f"Downloading {self.encoder_weights}")

            if self.download_url.startswith("https://drive.google.com/"):
                gdown.download(self.download_url, self.encoder_weights)
            else:
                try:
                    urllib.request.urlretrieve(
                        self.download_url,
                        self.encoder_weights,
                        pbar,
                    )
                except urllib.error.HTTPError as e:
                    print(
                        "Error while downloading model: The server couldn't fulfill the request."
                    )
                    print("Error code: ", e.code)
                except urllib.error.URLError as e:
                    print("Error while downloading model: Failed to reach a server.")
                    print("Reason: ", e.reason)
