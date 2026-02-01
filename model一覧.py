#結合
class CombinedModel(nn.Module):
    def __init__(self, model_a, model_b):
        super().__init__()
        self.feature_extractor = model_a
        self.classifier = model_b

    def forward(self, x):
        # model_aに通してから
        x = self.feature_extractor(x)
        # 必要ならここで形を変えたり処理を挟める
        # model_bに通す
        x = self.classifier(x)
        return x

# インスタンス化
model = CombinedModel(model_a, model_b)


#CNN
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        # ここで使う部品（レイヤー）を定義する
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)#入力チャネル3、出力チャネル32
        self.pool = nn.MaxPool2d(2, 2)#カーネルサイズ2、ストライド2
        self.dropout = nn.Dropout(0.5)#ドロップアウト率50%
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(256 * 4 * 4, num_classes)#全結合層

    def forward(self, x):
        # ここでデータの「通り道」を作る（順伝播）
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(x)に関してに関して
        
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)#dropoutは不活性にする
        
        x = self.pool(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))#relu
        ##ここから平坦化してクラス分類へ
        x = self.flatten(x)
        x = self.fc(x)
        
        return x

#cnnのクラスしてい自動版　　「[バッチサイズ, 3, 縦, 横] という形の、float32 型のテンソル」
class MyCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # ... (conv層の定義) ...
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)#入力チャネル3、出力チャネル32
        self.pool = nn.MaxPool2d(2, 2)#カーネルサイズ2、ストライド2
        self.dropout = nn.Dropout(0.5)#ドロップアウト率50%
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # 全体平均をとる部品（AdaptiveMaxPool2dやAvgPool2d）
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes) # 入力はチャンネル数の256だけでOK！

    def forward(self, x):
       x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(x)
        
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)#dropoutは不活性にする
        
        x = self.pool(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))#relu
        
        x = self.gap(x)   # (Batch, 256, H, W) -> (Batch, 256, 1, 1) になる
        x = torch.flatten(x, 1) # (Batch, 256) に平坦化
        x = self.fc(x)    # 最終的なクラス数へ
        return x
    
"""
criterion = nn.CrossEntropyLoss()      criterion(y_hat, labels)に関して

引数,　　中身の例 (テンソル),　　意味
y_hat,　　"[[2.0, 0.5, -1.0], [0.1, 3.0, 0.2]]",　　1枚目は「猫(0)」のスコアが一番高い。2枚目は「犬(1)」のスコアが一番高い。
labels,　　"[0, 2]",　　1枚目の正解は「猫(0)」。2枚目の正解は「鳥(2)」。
"""