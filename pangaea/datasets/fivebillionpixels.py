import os
import numpy as np
from glob import glob

from PIL import Image
import tifffile as tiff

import torch
import torchvision.transforms.functional as TF

from pangaea.datasets.base import RawGeoFMDataset

class FiveBillionPixels(RawGeoFMDataset):
    def __init__(
        self,
        **kwargs
    ):
        """Initialize the FiveBillionPixels dataset.
        Link to original dataset: https://x-ytong.github.io/project/Five-Billion-Pixels.html

        Args:
            split (str): split of the dataset (train, val, test).
            dataset_name (str): dataset name.
            multi_modal (bool): if the dataset is multi-modal.
            multi_temporal (int): number of temporal frames.
            root_path (str): root path of the dataset.
            classes (list): classes of the dataset.
            num_classes (int): number of classes.
            ignore_index (int): index to ignore for metrics and loss.
            img_size (int): size of the image. 
            bands (dict[str, list[str]]): bands of the dataset.
            distribution (list[int]): class distribution.
            data_mean (dict[str, list[str]]): mean for each band for each modality. 
            Dictionary with keys as the modality and values as the list of means.
            e.g. {"s2": [b1_mean, ..., bn_mean], "s1": [b1_mean, ..., bn_mean]}
            data_std (dict[str, list[str]]): str for each band for each modality.
            Dictionary with keys as the modality and values as the list of stds.
            e.g. {"s2": [b1_std, ..., bn_std], "s1": [b1_std, ..., bn_std]}
            data_min (dict[str, list[str]]): min for each band for each modality.
            Dictionary with keys as the modality and values as the list of mins.
            e.g. {"s2": [b1_min, ..., bn_min], "s1": [b1_min, ..., bn_min]}
            data_max (dict[str, list[str]]): max for each band for each modality.
            Dictionary with keys as the modality and values as the list of maxs.
            e.g. {"s2": [b1_max, ..., bn_max], "s1": [b1_max, ..., bn_max]}
            download_url (str): url to download the dataset.
            auto_download (bool): whether to download the dataset automatically.
            use_cmyk (bool): wheter to use cmyk or RGB-NIR colours for images.
        """
        super(FiveBillionPixels, self).__init__(**kwargs)

        self._base_dir = self.root_path

        self._image_dir = sorted(glob(os.path.join(self._base_dir, self.split, 'imgs', '*.tif')))
        self._label_dir = sorted(glob(os.path.join(self._base_dir, self.split, 'labels', '*.tif')))

    def __len__(self):
        return len(self._image_dir)

    def __getitem__(self, index):

        if self.use_cmyk:
            image = Image.open(self._image_dir[index]).convert('CMYK')
            image = TF.pil_to_tensor(image)
        else:
            image = tiff.imread(self._image_dir[index])#.convert('CMYK') #check it also on the normalization
            image = image.astype(np.float32)  # Convert to float32
            image = torch.from_numpy(image).permute(2, 0, 1)
        
        target = tiff.imread(self._label_dir[index])
        target = target.astype(np.int64)  # Convert to int64 (since it's a mask)
        target = torch.from_numpy(target).long()

        output = {
            'image': {
                'optical': image,
            },
            'target': target,
            'metadata': {}
        }

        if self.preprocessor is not None:
            output = self.preprocessor(output)

        return output

    
    # @staticmethod
    # def get_splits(dataset_config):
    #     dataset_train = FiveBillionPixels(dataset_config, split="train")
    #     dataset_val = FiveBillionPixels(dataset_config, split="val")
    #     dataset_test = FiveBillionPixels(dataset_config, split="test")
    #     return dataset_train, dataset_val, dataset_test
    
