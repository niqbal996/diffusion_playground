import os
import random
import torch
from torch.utils.data import Dataset
import PIL.Image
import torch.nn as nn
from torchvision import transforms

class DepthDataset(Dataset):
    def __init__(self, root: str, drop_rate: float = 0.0, resolution: int = 512, center_crop: bool = False, random_flip: bool = False):
        self.root = root
        self.drop_rate = drop_rate

        root = os.path.expanduser(root)
        if not os.path.isdir(os.path.join(root, 'images')):
            raise FileNotFoundError(f"{os.path.join(root, 'images')} not found.")
        if not os.path.isdir(os.path.join(root, 'depth')):
            raise FileNotFoundError(f"{os.path.join(root, 'depth')} not found.")
        
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution),
            transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
            transforms.Normalize([0.5], [0.5]),
        ])

        self.image_files = sorted(os.listdir(os.path.join(root, 'images')))
        self.depth_files = sorted(os.listdir(os.path.join(root, 'depth')))
        assert len(self.image_files) == len(self.depth_files), "Number of images and depth maps must be the same"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root, 'images', self.image_files[idx])
        depth_path = os.path.join(self.root, 'depth', self.depth_files[idx])

        # Load images using PIL to ensure consistent behavior
        image = PIL.Image.open(image_path).convert('RGB')
        depth = PIL.Image.open(depth_path).convert('RGB')

        # Apply same random transforms to both images
        seed = torch.randint(0, 2**32, (1,)).item()
        
        torch.manual_seed(seed)
        image = self.transforms(image)
        
        torch.manual_seed(seed)
        depth = self.transforms(depth)

        return {
            "pixel_values": image,
            "depth_values": depth,
            "prompt": '' # Empty tensor as placeholder
        }
    