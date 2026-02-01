######
##データセットの整形
######
from datasets import load_dataset
DATASET_NAME = "prithivMLmods/Face-Age-10K" 
dataset = load_dataset(DATASET_NAME)

print(dataset)
print(type(dataset))
#print(dataset['train'][0])
#print(dataset['train'][0]['image'])
import pandas as pd
df = pd.DataFrame(dataset['train'])
print(df.head())
print(df.head()['image'])
print(df['label'].value_counts())

"""
df['image'] は まだ PIL Image オブジェクト です

PIL Image には .shape 属性はありません。

.shape は NumPy 配列や Tensor に変換してから使います

(200, 200, 3) とラベルの集合であることを確認
"""
import numpy as np
df['image'] = df['image'].apply(lambda img: np.array(img))
print(df.head()['image'][0].shape)

#split into train and val
from sklearn.model_selection import train_test_split

# df はすでに Pandas DataFrame で 'image' と 'label' がある前提
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
#乱数の種（seed）random_state=42
#stratify=df['label'] ラベル分布を train/test でも同じ比率にすること
print(train_df['label'].value_counts())

#detaloaderの作成
import torch
from torch.utils.data import DataLoader, Dataset
class MyDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image = self.dataframe.iloc[idx]['image']
        label = self.dataframe.iloc[idx]['label']
        
        if self.transform:
            image = self.transform(image)
            age = torch.tensor(label, dtype=torch.long)

        return image, age

from torchvision import transforms
transform = transforms.Compose([
    transforms.ToTensor(),  # NumPy配列をTensorに変換し、[0, 1]に正規化
])
train_dataset = MyDataset(train_df, transform=transform)
test_dataset = MyDataset(test_df, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
#shuffle=Trueデータを読み込む順番を毎エポックごとにランダムに入れ替える
print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of test samples: {len(test_dataset)}")