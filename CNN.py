##### PARAMETERS #####
# path where you want to store your data. Must contain the dataset-master file
path = "./data/"
# optimizer selected. Only SGD and RMSprop are supported
optimizer_name = "SGD"
# how many epoch you want to use to train the CNN
epochs = 100

##### IMPORTS ######
import os
import shutil

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

from torchvision import transforms
from torchvision.datasets import ImageFolder

import matplotlib.pyplot as plt

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
    df = pd.read_csv(path+"dataset-master/labels.csv",sep =",")
    
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
          
def build_training_and_testing(path, df, valid_size) :
    # 1 : separate training and testing set
    train_df, valid_df = train_test_split(
        df, 
        test_size = valid_size,
        stratify = df['Category'])
    
    # 2 : build training and testing folders
    build_folder(path, "training", train_df)
    build_folder(path, "testing", valid_df)

##### MAIN ######


# I : read the labels
df = load_csv(path)

# II : separate training and testing set
build_training_and_testing(path, df, 0.25)

# III : training parameter
net = BloodCell()
batch_size = 128
criterion = nn.CrossEntropyLoss()

if(optimizer_name == "SGD") :
    optimizer = optim.SGD(net.parameters(), lr=1e-3)
elif(optimizer_name == "RMSprop") :
    optimizer = optim.RMSprop(net.parameters(), lr=1e-3)
else : 
    raise Exception("Please use the SGD or RMSprop optimizer !")

transform = transforms.Compose([transforms.Resize((120,120)),transforms.ToTensor()])
training_set = ImageFolder(path+"training",transform = transform)
testing_set = ImageFolder(path+"testing",transform = transform)

training_loader = DataLoader(training_set, batch_size, shuffle = True)
testing_loader = DataLoader(testing_set, batch_size)

# IV : Training
min_valid_loss = np.inf
training_losses = []
valid_losses = []
training_accuracies = []
valid_accuracies = []

for e in range(epochs):
    train_loss = 0.0
    train_accuracy = 0.0
    net.train()
    for data, labels in training_loader :
        optimizer.zero_grad()
        output = net(data)
        loss = criterion(output,labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
        _, predicted = T.max(output, 1)
        train_accuracy += (predicted == labels).sum().item()/len(labels)
        
    with T.no_grad(): 
        valid_loss = 0.0
        valid_accuracy = 0.0
        net.eval()
        for data, labels in testing_loader:
            output = net(data)
            loss = criterion(output,labels)
            valid_loss += loss.item()
            
            _, predicted = T.max(output, 1)
            valid_accuracy += (predicted == labels).sum().item()/len(labels)
                
            
    print(f'Epoch {e+1}')
    print(f'\t Training Loss: {train_loss / len(training_loader)}')
    print(f'\t Validation Loss: {valid_loss / len(testing_loader)}')
    print(f'\t Training Accuracy: {train_accuracy / len(training_loader)}')
    print(f'\t Validation Accuracy: {valid_accuracy / len(testing_loader)}')
    training_losses.append(train_loss / len(training_loader))
    valid_losses.append(valid_loss / len(testing_loader))
    training_accuracies.append(train_accuracy / len(training_loader))
    valid_accuracies.append(valid_accuracy / len(testing_loader))

    if(valid_loss / len(testing_loader) < min_valid_loss) :
        min_valid_loss = valid_loss
        print("\t Amélioration de la loss, sauvegarde du modèle")
        T.save(net.state_dict(), "./CNNBLoodCell")


#V : Plots
plt.plot(training_losses, label = "training")
plt.plot(valid_losses, label = "validation")
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Cross entropy loss')
plt.title('Loss plot, using the ' + optimizer_name + ' optimizer')
plt.savefig("loss_plot_"+ optimizer_name +".png")
plt.show()

plt.plot(training_accuracies, label = "training")
plt.plot(valid_accuracies, label = "validation")
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy plot, using the ' + optimizer_name + ' optimizer')
plt.savefig("accuracy_plot_"+ optimizer_name +".png")
plt.show()
