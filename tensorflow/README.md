# 必要なライブラリ  
・python 3系  
・numpy  
・pandas  
・scikit-learn  
・OpenCV  
・tensorflow  
・efficientnet  
</br>

# 使いかた  
１．kaggleにデータセットとしてファイルをアップロード  
    
２．ノートにデータファイルとアップロードしたデータセットを追加  
  
３．作業ディレクトリに移動  
import os  
path = "..input/データセット名/Melanoma-Classification/tensorflow  
os.chdir(path)  
  
４．ファイルの実行  
!python submission.py --model 'モデル名'  
モデル名 = 'VGG16', 'ResNet', 'SENet', 'EfficientNet'
</br>

# 参考文献  
https://www.kaggle.com/ibtesama/siim-baseline-keras-vgg16
