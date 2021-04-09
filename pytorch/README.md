# 必要なライブラリ
・python 3系  
・numpy  
・pandas  
・scikit-learn  
・OpenCV  
・pythorch
・efficientnet_pytorch  
・torchtoolbox  
</br>

# 使いかた
１．kaggleにデータセットとしてファイルをアップロード  
    
２．ノートに[データセット](https://www.kaggle.com/cdeotte/jpeg-melanoma-256x256)とアップロードしたデータセットを追加  

３．必要なライブラリのインストール  
```py
!pip install efficientnet_pytorch
!pip install torchtoolbox
```
  
４．作業ディレクトリ(pytorch/section0x)に移動  
```py
import os  
path = "..input/データセット名/Melanoma-Classification/pytorch/section0x"
os.chdir(path)  
```
  
５．ファイルの実行 
```py 
!python submission.py 
```
</br>