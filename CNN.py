##### IMPORTS ######
import os
import shutil


import pandas as pd
from sklearn.model_selection import train_test_split

import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

from torchvision import transforms
from torchvision.datasets import ImageFolder

##### ARCHITECTURE #####
class BloodCell(nn.Module):
    def __init__(self):
        super().__init__()
        
        # NETWORK
        self.network = nn.Sequential(
            
            # 1
            nn.Conv2d(3, 16, kernel_size = 3 , stride = (2,2)),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(p=0.2),
            nn.MaxPool2d(kernel_size =2),
            
            # 2
            nn.Conv2d(16, 8, kernel_size = 3 , stride = (2,2)),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(p=0.2),
            nn.MaxPool2d(kernel_size =2),
            
            # 3
            nn.Conv2d(8, 4, kernel_size = 3, stride = (2,2)),
            nn.ReLU(),
            nn.BatchNorm2d(4),
            nn.Dropout(p=0.2),
            nn.MaxPool2d(kernel_size =2),
        )

        # AVGPOOL
        self.avgpool = nn.AdaptiveAvgPool2d((15,15))

        # CLASSIFIER
        self.classifier = nn.Sequential(
            
            nn.Flatten(),
            nn.Linear(4*15*15,200),
            nn.ReLU(),
            nn.Linear(200, 50),
            nn.ReLU(),
            nn.Linear(50,4),
            nn.ReLU(),
        )
    
    def forward(self, xb):
        xb = self.network(xb)
        xb = self.avgpool(xb)
        xb = self.classifier(xb)
        return xb

##### FUNCTIONS ######
def load_csv(path) :
    # 1 : read the csv
    df = pd.read_csv(path+"dataset-master/labels_full.csv",sep =",")
    
    # 2 : remove images with missing labels
    df = df[df['Category'].notnull()]
    
    # 3 : translate the "Image" column to the path of the actual image
    df['Image'] = df['Image'].apply(lambda x : 
                                    "BloodImage_" 
                                    + (5-len(str(x)))*"0"
                                    + str(x)
                                    + ".jpg")
    # 4 drop unnecessary columns
    df = df[['Image', 'Category']]
    
    # 5 drop categories without enough data to train the model on
    category_value_count = df.Category.value_counts()
    rare_categories = category_value_count[category_value_count < 10].index
    df = df.loc[~df["Category"].isin(rare_categories)]

    return df

def build_folder(path, folder ,df) :
    # reset and make folder
    if(os.path.isdir(path+folder)) :
        shutil.rmtree(path+folder) 
    os.mkdir(path+folder)
    
    # make subfolders
    for categories in df.Category.unique():
        if(not os.path.isdir(path+folder+"/"+categories)) :
            os.mkdir(path+folder+"/"+categories)
    
    # fill subfolders
    for index, row in df.iterrows() :
        origin = path + "dataset-master/JPEGImages/" + row["Image"]
        target = path + folder + "/" + row["Category"] + "/" + row["Image"]
        if(os.path.isfile(origin)) :
            shutil.copyfile(origin, target)
        else :
            print(origin + " is in the csv, but does not exist")
          
def build_training_and_validation(path, df, valid_size) :
    # 1 : separate training and validation set
    train_df, valid_df = train_test_split(
        df, 
        test_size = valid_size,
        stratify = df['Category'])
    
    # 2 : build training and validation folders
    build_folder(path, "training", train_df)
    build_folder(path, "validation", valid_df)

##### MAIN ######
# path where you want to store your data. Must contain the dataset-master file
path = "./data/"

# how you with to transform the data
transform = transforms.Compose([
    transforms.Resize((120,120)),
    transforms.ToTensor()])



# I : read the labels
df = load_csv(path)

# II : separate training and validation set
build_training_and_validation(path, df, 0.05)

# III : build
net = BloodCell()
batch_size = 128
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=1e-3)

training_set = ImageFolder(path+"training",transform = transform)
validation_set = ImageFolder(path+"validation",transform = transform)

training = DataLoader(training_set, batch_size, shuffle = True)
validation = DataLoader(validation_set, batch_size)


"""
for epoch in range (100):
    net.train()
    running_loss = 0.0
    for i, data in enumerate(training, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    else:
        print("Epoch{} - Training loss : {}".format(epoch,running_loss/len(training)))

"""
for epoch in range (100):
    net.train()
    training_loss = 0.0
    testing_loss = 0.0

    true_training, testing = train_test_split(
        training, 
        test_size = valid_size,
        stratify = df['Category'])



