if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import time
import datetime
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from efficientnet_pytorch import EfficientNet

from models import Net
from transformer import train_transform, test_transform
from data_processing import MelanomaDataset, create_data

"""
モデルを学習させる関数
"""

def train(data_folder, output_folder, es_patience, epochs, TTA, model_name):
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    print("使用デバイス：", device)

    train, test = create_data(data_folder)

    arch = EfficientNet.from_pretrained('efficientnet-b1') #モデル

    meta_features = ['sex', 'age_approx'] + [col for col in train.columns if 'site_' in col]
    meta_features.remove('anatom_site_general_challenge')

    #パラメータ各種

    oof = np.zeros((len(train), 1))
    preds = torch.zeros((len(test), 1), dtype = torch.float32, device = device)

    skf = KFold(n_splits = 5, shuffle = True, random_state = 47)

    #===========================
    # ループスタート
    #===========================
    for fold, (idxT, idxV) in enumerate(skf.split(np.arange(15)), 1):
        print('=' * 20, 'Fold', fold, '=' * 20)

        #kfold
        train_idx = train.loc[train['fold'].isin(idxT)].index
        val_idx = train.loc[train['fold'].isin(idxV)].index

        #学習パラメータ
        model_path = f'/model_{fold}.pth'
        best_val = 0
        patience = es_patience

        model = Net(arch = arch, n_meta_features = len(meta_features))

        model = model.to(device)
        
        #optimizer
        optim = torch.optim.Adam(model.parameters(), lr = 0.001)
        #scheduler
        scheduler = ReduceLROnPlateau(
            optimizer = optim, 
            mode = 'max', 
            patience = 1, 
            verbose = True, 
            factor = 0.2
        )

        criterion = nn.BCEWithLogitsLoss()

        trainDataset = MelanomaDataset(
            df = train.iloc[train_idx].reset_index(drop = True), 
            imfolder = data_folder + '/train', 
            train = True, 
            transforms = train_transform, 
            meta_features = meta_features
        )

        valDataset = MelanomaDataset(
            df = train.iloc[val_idx].reset_index(drop = True),
            imfolder = data_folder + '/train',
            train = True,
            transforms = test_transform,
            meta_features = meta_features
        )

        testDataset = MelanomaDataset(
            df = test,
            imfolder = data_folder + '/test',
            train = False,
            transforms = test_transform,
            meta_features = meta_features
        )

        train_loader = DataLoader(dataset = trainDataset, batch_size = 64, shuffle = True, num_workers = 2)
        val_loader = DataLoader(dataset = valDataset, batch_size = 16, shuffle = False, num_workers = 2)
        test_loader = DataLoader(dataset = testDataset, batch_size = 16, shuffle = False, num_workers = 2)

        #=====================
        # epochs
        #=====================
        for epoch in range(epochs):
            start_time = time.time()
            correct = 0
            epoch_loss = 0
            
            #train_loop
            model.train()
            for x, y in train_loader:
                x[0] = torch.tensor(x[0], device = device, dtype = torch.float32)
                x[1] = torch.tensor(x[1], device = device, dtype = torch.float32)
                y = torch.tensor(y, device = device, dtype = torch.float32)
                optim.zero_grad()
                z = model(x)
                loss = criterion(z, y.unsqueeze(1))
                loss.backward()
                optim.step()
                pred = torch.round(torch.sigmoid(z))
                correct += (pred.cpu() == y.cpu().unsqueeze(1)).sum().item()
                epoch_loss += loss.item()
            train_acc = correct / len(train_idx)

            model.eval()
            val_preds = torch.zeros((len(val_idx), 1), dtype = torch.float32, device = device)

            with torch.no_grad():
                #validation_loop
                for j, (x_val, y_val) in enumerate(val_loader):
                    x_val[0] = torch.tensor(x_val[0], device = device, dtype = torch.float32)
                    x_val[1] = torch.tensor(x_val[1], device = device, dtype = torch.float32)
                    y_val = torch.tensor(y_val, device = device, dtype = torch.float32)
                    z_val = model(x_val)
                    val_pred = torch.sigmoid(z_val)
                    val_preds[j * val_loader.batch_size : j * val_loader.batch_size + x_val[0].shape[0]] = val_pred

                val_acc = accuracy_score(train.iloc[val_idx]['target'].values, torch.round(val_preds.cpu()))
                val_roc = roc_auc_score(train.iloc[val_idx]['target'].values, val_preds.cpu())

                print('Epoch{:03}: | Loss：{:.3f} | Train acc：{:.3f} | Val acc：{:.3f} | Val roc_auc：{:.3f} | Training time：{}'.format(
                    epoch + 1,
                    epoch_loss,
                    train_acc,
                    val_acc,
                    val_roc,
                    str(datetime.timedelta(seconds = time.time() - start_time))[:7])
                )

                scheduler.step(val_roc)

                if val_roc >= best_val:
                    best_val = val_roc
                    patience = es_patience

                    torch.save(model, output_folder + model_path)

                else:
                    patience -= 1
                    if patience == 0:
                        print('Early stopping. Best Val roc_auc：{:.3f}'.format(best_val))
                        break

        model = torch.load(output_folder + model_path)
        model.eval()
        val_preds = torch.zeros((len(val_idx), 1), dtype = torch.float32, device = device)

        #evaluation loop
        with torch.no_grad():
            for j, (x_val, y_val) in enumerate(val_loader):
                x_val[0] = torch.tensor(x_val[0], device = device, dtype = torch.float32)
                x_val[1] = torch.tensor(x_val[1], device = device, dtype = torch.float32)
                y_val = torch.tensor(y_val, device = device, dtype = torch.float32)
                z_val = model(x_val)
                val_pred = torch.sigmoid(z_val)
                val_preds[j * val_loader.batch_size : j * val_loader.batch_size + x_val[0].shape[0]] = val_pred
            oof[val_idx] = val_preds.cpu().numpy()

            for _ in range(TTA):
                for i, x_test in enumerate(test_loader):
                    x_test[0] = torch.tensor(x_test[0], device = device, dtype = torch.float32)
                    x_test[1] = torch.tensor(x_test[1], device = device, dtype = torch.float32)
                    z_test = model(x_test)
                    z_test = torch.sigmoid(z_test)
                    preds[i * test_loader.batch_size : i * test_loader.batch_size + x_test[0].shape[0]] += z_test

                preds /= TTA

    preds /= skf.n_splits

    return preds, oof