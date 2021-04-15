import torchtoolbox.transform as transforms

from augmentation import *

"""
データセットに使うtransformer
"""

train_transform = transforms.Compose([
    AdvancedHairAugmentation(hairs_folder = '/kaggle/input/melanoma-hairs'), 
    transforms.RandomResizedCrop(size = 256, scale = (0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    Microscope(p = 0.5), 
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])