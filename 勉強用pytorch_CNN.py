import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from 勉強用dataset import train_loader, test_loader
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MyCNN(nn.Module):
    def __init__(self, num_classes=8):
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
    
model=MyCNN(num_classes=8).to(device)
from datasets import load_dataset
DATASET_NAME = "prithivMLmods/Face-Age-10K" 
dataset = load_dataset(DATASET_NAME)

criterion = nn.CrossEntropyLoss()      #criterion(y_hat, labels)に関して
optimizer = optim.Adam(model.parameters(), lr=0.01)
num_epochs = 5

for epoch in range(num_epochs):
    model.train()   # 学習モード
    total_loss = 0

    for batch in train_loader:
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        # 勾配初期化
        optimizer.zero_grad()

        # forward
        outputs = model(images)

        # loss
        loss = criterion(outputs, labels)

        # backward
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
os.makedirs("/workspace/checkpoints", exist_ok=True)
torch.save(model.state_dict(), "/workspace/checkpoints/final.pth")

