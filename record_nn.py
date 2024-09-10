'''
-- Loads the image data in 
-- Normalizes images
-- Creates training and validation sets 
-- Sets up the model and the training loop
-- Applies the model to unlabeled data
-- Calculates the most probable class for each document
NOTE: the code has been modified/shortened for clarity
'''
import sys, copy, json, copy, shutilm, os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from PIL import Image
import pandas as pd

class CustomImageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Load image and apply transformations
        img_path = self.dataframe.iloc[idx]['image_path']
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image
    
def create_unlabeled_data(path):
    '''
    path (str) -- path to the unlabeled data
    '''
    transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images if needed
    transforms.ToTensor(),          # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])

    # Load the CSV data
    df = pd.read_csv(path)
    # Create the custom dataset
    unlabeled_dataset = CustomImageDataset(dataframe=df, transform=transform)
    # Create a DataLoader for the dataset
    unlabeled_loader = torch.utils.data.DataLoader(unlabeled_dataset, batch_size=32, shuffle=False)
    '''
    Returns the loader with unlabeled image data
    '''
    return unlabeled_loader

def transformer(data_dir):
    '''
    Transforms the image into a training-ready format
    data_dir (str) -- path to a directory with Chinese character data
    '''
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                        data_transforms[x])
            for x in ['train', 'val']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=32,
                            shuffle=True, num_workers=4)
            for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    '''
    Returns: 
    class names (list) -- a list of strings containing labels
    dataloaders (dict) -- a dictionary containing train and val sets
    dataset_sizes (dict) -- A dictionary containing the sizes (number of samples) of the training and validation datasets.
    image_datasets (dict) --  A dictionary containing ImageFolder datasets for the training and validation datasets.
    '''
    return class_names, dataset_sizes, dataloaders, image_datasets


# Defining the CNN model
class Net(nn.Module):
    def __init__(self, class_names):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, len(class_names))
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Training the model
def train_model(model, criterion, optimizer, device, dataloaders, dataset_sizes, class_names,
                num_epochs=25):
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        print()

    return model

def predict_labels(model, unlabeled_data, device='cpu'):
    '''
    Apply a trained model to an unlabeled dataset and return predicted labels.
    model (torch.nn.Module): The trained PyTorch model.
    unlabeled_data (torch.utils.data.Dataset): The dataset containing unlabeled data.
    device (str): The device to run the model on ('cpu' or 'cuda').
    '''
    model.eval()  # Set the model to evaluation mode
    predictions = []

    # Move the model to the specified device
    model.to(device)

    # Create a DataLoader for the unlabeled data
    unlabeled_loader = torch.utils.data.DataLoader(unlabeled_data, batch_size=32, shuffle=False)

    with torch.no_grad():  # Disable gradient calculation for inference
        for batch in unlabeled_loader:
            inputs = batch['input'].to(device)  # Adjust according to your dataset structure
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)  # Get the predicted class
            predictions.extend(predicted.cpu().numpy())
    
    '''
    Returns:
        List[int]: Predicted labels for the unlabeled dataset.
    '''

    return predictions



def create_final_df(path1, path2, label_list):
    '''
    Calculates the most likely author for each prison record
    path1 (str) -- path to the df with all prison records
    path2 (str) -- path to the df that maps each character to a prison record
    label_list -- list of labels produced by the model
    '''
    document_df = pd.read_csv(path1)
    character_df = pd.read_csv(path2)
    character_df['new_labels'] = label_list
    character_df = character_df.groupby('group')['record_id'].agg(lambda x: x.mode().iloc[0])
    document_df = document_df.merge(character_df, on = "record_id", how = "left")
    return document_df
    
    
    
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    class_names, dataset_sizes, dataloaders, image_datasets = transformer("/chn_chars_ver2")
    model = Net(class_names).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model = train_model(model, criterion, optimizer, device, dataloaders, dataset_sizes, class_names,
                        num_epochs=25)
    un_loader = create_unlabeled_data("/unlabeled.csv")
    predicted_labels = predict_labels(model, un_loader, device='cuda')
    final_df = create_final_df("/all_data.csv",
                               "/unlabeled.csv", predicted_labels)
    final_df.to_csv("/joined_records.csv")
    


if __name__ == "__main__":
    main()
    