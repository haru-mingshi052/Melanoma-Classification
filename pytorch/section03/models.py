import torch
import torch.nn as nn

"""
モデルの定義
"""

class Net(nn.Module):
    def __init__(self, arch, n_meta_features: int):
        super(Net, self).__init__()
        self.arch = arch
        if 'ResNet' in str(arch.__class__):
            self.arch.fc = nn.Linear(in_features=512, out_features=500, bias=True)
        if 'EfficientNet' in str(arch.__class__):
            self.arch._fc = nn.Linear(in_features=1280, out_features=500, bias=True)
        self.meta = nn.Sequential(
            nn.Linear(n_meta_features, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Dropout(p = 0.2),
            nn.Linear(500, 250), 
            nn.BatchNorm1d(250),
            nn.ReLU(),
            nn.Dropout(p = 0.2),
            nn.Linear(250, 250),
            nn.BatchNorm1d(250),
            nn.ReLU(),
            nn.Dropout(p = 0.2)
        )
        self.together0 = nn.Linear(500 + 250, 100) #追加
        self.together1 = nn.BatchNorm1d(100) #追加
        self.together2 = nn.ReLU() #追加
        self.together3 = nn.Dropout(p = 0.2) #追加
        self.together4 = nn.Linear(100, 25) #追加
        self.together5 = nn.BatchNorm1d(25) #追加
        self.ouput = nn.Linear(25, 1)

    def forward(self, inputs):
        """
        No sigmoid in forward because we are going to use BCEWithLogitsLoss
        Which applies sigmoid for us when calculating a loss
        """
        x, meta = inputs
        cnn_features = self.arch(x)
        meta_features = self.meta(meta)
        features = torch.cat((cnn_features, meta_features), dim=1)
        together0 = self.together0(features) #Linear
        together1 = self.together1(together0) #BN
        together2 = self.together2(together1) #ReLU
        together3 = self.together3(together2) #Dropout
        together4 = self.together4(together3) #Linear
        together5 = self.together5(together4) #BN
        together6 = self.together2(together5) #ReLU
        together7 = self.together3(together6) #Dropout
        output = self.ouput(together7)
        return output