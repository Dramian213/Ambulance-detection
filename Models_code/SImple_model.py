import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset,random_split
from torchvision import transforms, models
from PIL import Image
import os

#Data loading and checking
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)
data_dir = os.path.join(base_dir, 'carsdataset')

if os.path.exists(data_dir):
    print(f"{data_dir} exists")
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path):
            print(f"{item_path} is a directory")
else:
    print(f"{data_dir} does not exist")

#Dataset
class AmbulanceDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        ambulance_dir = os.path.join(data_dir, "ambulance")
        if os.path.exists(ambulance_dir):
            for img_name in os.listdir(ambulance_dir):
                self.image_paths.append(os.path.join(ambulance_dir, img_name))
                self.labels.append(1)\

        no_ambulance_dir = os.path.join(data_dir, "noambulance")
        if os.path.exists(no_ambulance_dir):
            for img_name in os.listdir(no_ambulance_dir):
                self.image_paths.append(os.path.join(no_ambulance_dir, img_name))
                self.labels.append(0)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (100, 100), color="black")

        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


#Simple transformation - resize and tensor only
simple_transform = transforms.Compose([
    transforms.Resize((100,100)),
    transforms.ToTensor(),
])

#Testing
i = 1
if i == 1:
    try:
        dataset = AmbulanceDataset(data_dir, simple_transform)
        print(f"Work, amount of pictures: {len(dataset)}")

        img,label = dataset[0]
        print(img.shape)
        print(label)

        ambulance_count = sum(1 for label in dataset.labels if label == 1)
        nonambulance_count = sum(1 for label in dataset.labels if label == 0)

        print(ambulance_count)
        print(nonambulance_count)

    except Exception as ex:
        print(ex)

#Simple CNN model
class TinyCNN(nn.Module):
    def __init__(self):
        super(TinyCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(in_features=16*25*25, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=2)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

# model run test
model = TinyCNN()
print(model)

'''
# test on random tensor
test_input = torch.randn(15,3,100,100)
output = model(test_input)
print("f\nInput:\n{}".format(test_input))
print("f\nOutput:\n{}".format(output))

'''

# Dataset division for train and validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_Dataset,val_Dataset = random_split(dataset,[train_size,val_size])

print(f"Abount of training set: {len(train_Dataset)}")
print(f"Abount of validation set: {len(val_Dataset)}")

#Data Loader
batch_size = 32

train_loader = DataLoader(
    train_Dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_Dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)

print(f"Batch size: {batch_size}")
print(f"Number of training samples: {len(train_loader)} \n"
      f"Number of validation samples: {len(val_loader)}")

'''for imagres, labels in train_loader:
    print(f"Images shape:{imagres.shape}"
          f" Labels shape:{labels.shape}"
          f" Przykladowe etykiety {labels[:5]}")
    break
'''
#Simple train function
def simple_train_function(model, train_loader, val_loader,num_epochs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Starting training on: {device}")

    model = model.to(device)
    #loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.  Adam(model.parameters(), lr=0.001)

    #History of training
    history = {"train_loss": [], "val_loss": [],"train_acc": [], "val_acc": [], "recall": [], "precision": []}

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            images, labels = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            #Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            #Backward pass
            loss.backward()
            optimizer.step()

            #Stats
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == labels).sum().item()

            #Progress in range of 10 batch
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}, loss: {running_loss:.4f}")

            train_loss = running_loss/len(train_loader)
            train_acc = correct / total * 100

            #Walidation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        tp, fn = 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                tp += ((predicted == 1) & (labels == 1)).sum().item()
                fn += ((predicted == 0) & (labels == 1)).sum().item()

        val_loss = val_loss/len(val_loader)
        val_acc = correct / total * 100
        recall = tp/(tp + fn+ 1e-8)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["recall"].append(recall)

        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"\nTrain loss: {train_loss:.4f}"
                f"\nValidation loss: {val_loss:.4f}"
                f"\nTrain acc: {train_acc:.4f}"
                f"\nValidation acc: {val_acc:.4f}"
                f"\nRecall: {recall:.4f}")

    return history, model

history, trained_model = simple_train_function(model, train_loader, val_loader, num_epochs=5)








