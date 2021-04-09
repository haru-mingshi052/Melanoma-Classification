import numpy as np
import os
import cv2

from torch.utils.data import Dataset

class MelanomaDataset(Dataset):
    def __init__(self, df, imfolder, train, transforms, meta_features):
        self.df = df
        self.imfolder = imfolder
        self.transforms = transforms
        self.train = train
        self.meta_features = meta_features

    def __getitem__(self, index):
        im_path = os.path.join(self.imfolder, self.df.iloc[index]['image_name'] + '.jpg')
        x = cv2.imread(im_path)
        meta = np.array(self.df.iloc[index][self.meta_features].values, dtype = np.float32)

        if self.transforms:
            x = self.transforms(x)

        if self.train:
            y = self.df.iloc[index]['target']
            return (x, meta), y
        else:
            return (x, meta)

    def __len__(self):
        return len(self.df)