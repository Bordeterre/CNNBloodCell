
##### IMPORTS #####
import pandas as pd

from sklearn.model_selection import train_test_split

from torchvision.io import read_image
import torch.nn as nn


##### FUNCTION #####
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


# Import data
def image_to_tensor(path) :
    try : 
        return read_image(path)
    
    except :
        return None

def import_and_clean(path) :
    # Import csv
    df = pd.read_csv(path+"labels.csv",sep =",").drop("Unnamed: 0", axis=1)
    df["len"] = 5-df["Image"].astype(str).str.len()
    df["zeroes"] =  "0"
    df["Image"] =  "BloodImage_" + df["zeroes"] * df["len"] +df["Image"].astype(str) + ".jpg"
    
    
    #cleanup csv
    df = df.drop(["len", "zeroes"], axis=1)
    df = df[df['Category'].notnull()]

    #Import images
    df["Tensor"] = df["Image"].apply(lambda x : image_to_tensor(path+"JPEGImages/"+x))
    
    df = df[df["Tensor"].notnull()]
    
    return list(df["Category"]), list(df["Tensor"])


##### MAIN ######
path = "data/dataset-master/"
labels,features = import_and_clean(path)

train_features, validation_features, train_labels, validation_labels = train_test_split(features, labels, test_size = 0.05)
train_features, test_features, train_labels, test_labels = train_test_split(train_features, train_labels, test_size = 0.2)



batch_size=1





net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=1e-3)

for t in range(2):
    optimizer.zero_grad()
    predictions = mod(train_features)
    loss = criterion(predictions,train_labels)
    loss.backward()
    optimizer.step()