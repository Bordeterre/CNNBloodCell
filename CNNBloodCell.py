##### IMPORT #####
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split

from torchvision.io import read_image
from torchvision import transforms

import os
import pandas as pd


##### FONCTIONS #####
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,10, kernel_size=16, stride = 2),
            nn.ReLU(inplace=True),

            #nn.BatchNorm1d(16),
            #nn.Dropout(0.2),
            
            #nn.Conv2d(16,10 , kernel_size=8, stride = 2),
            #nn.ReLU(inplace=True),
            
            #nn.BatchNorm1d(8),
            #nn.Dropout(0.2),
            
            #nn.Conv2d(8, 10, kernel_size=4, stride = 2),
            #nn.ReLU(inplace=True),
            
            #nn.BatchNorm1d(4),
            #nn.Dropout(0.2),
            
            nn.Flatten()
            
        )
        
        self.classifier = nn.Sequential(
            #nn.Linear(4, 32),
            #nn.Linear(8, 16),
            #nn.Linear(16, 8), derniere sortie nombre de classes
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class CustomImageDataset():
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        df = pd.read_csv(annotations_file)
        df = df[df['Category'].notnull()]
        self.img_labels = df
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        path = str(self.img_labels.iloc[idx, 1])
        path = "BloodImage_"+"0"*(5-len(path))+path+".jpg"
        img_path = os.path.join(self.img_dir, path)
        
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 2]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

def split_dataset(dataset, test_prop=0.05, valid_prop=0.02) :
    # Attention, valid_prop est proportion du train+valid test, pas du dataset total

    test_size = int(test_prop*len(dataset))
    train_set, test_set = random_split(dataset, [len(dataset)-test_size, test_size])

    valid_size = int(valid_prop*len(train_set))
    train_set, valid_set = random_split(train_set, [len(train_set)-valid_size, valid_size])

    return train_set, valid_set, test_set

def build_loader(set, batch_size, shuffle=False) :
    return T.utils.data.DataLoader(train_set, batch_size=batch_size,
                                         shuffle=shuffle, num_workers=2)



# train

##### MAIN #####
device = 'cuda' if T.cuda.is_available() else 'cpu'

transform = transforms.Compose([transforms.Resize((224,320))])


dataset = CustomImageDataset("data/dataset-master/labels.csv","data/dataset-master/JPEGImages/", transform)
train_set, valid_set, test_set = split_dataset(dataset) 


batch_size=1
train_loader = build_loader(train_set, batch_size, shuffle=True)
test_loader = build_loader(test_set, batch_size)
valid_loader = build_loader(valid_set, batch_size)





net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=1e-3)

for epoch in range(2): # loop over the dataset multiple time
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        print(i)
        print(data)
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()