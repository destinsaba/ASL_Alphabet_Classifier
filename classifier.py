import torch
import glob
import matplotlib.pylab as plt
from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision.models import resnet50
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import wandb
import os
import shutil

# Define hyperparameters
HYPERPARAMETERS = {
    "project_name": "ASL Classifier",
    "entity": "destinsaba-fun",
    "learning_rate": 0.001,
    "epochs": 15,
    "batch_size": 32,
    "num_workers": 2,
    "train_ratio": 0.8,
    "val_ratio": 0.2,
    "dev_path": "/home/destin.saba/transfer-learning/ASL_data/asl_alphabet_train/asl_alphabet_train",
    "test_path": "/home/destin.saba/transfer-learning/ASL_data/asl_alphabet_test/asl_alphabet_test",
    "input_shape": (3, 224, 224),
    "num_classes": 29,
    "model_path": './ASL_net.pth'
}


def main():
    # Initialize wandb
    wandb.init(project=HYPERPARAMETERS["project_name"], entity=HYPERPARAMETERS["entity"])

    # Check if GPU is available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)

    import torch.optim as optim
    from torch.optim.lr_scheduler import ExponentialLR

    DEV_PATH = HYPERPARAMETERS["dev_path"]
    TEST_PATH = HYPERPARAMETERS["test_path"]

    # Transforms 
    torchvision_transform = transforms.Compose([transforms.Resize((224,224)),\
        transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
        transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406] ,std=[0.229, 0.224, 0.225] )])


    torchvision_transform_test = transforms.Compose([transforms.Resize((224,224)),\
        transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406] ,std=[0.229, 0.224, 0.225])])


    # Load datasets
    development_dataset = ImageFolder(root=DEV_PATH, transform=torchvision_transform)
    test_dataset = ImageFolder(root=TEST_PATH, transform=torchvision_transform_test)

    # Define the split ratio for training and validation datasets
    train_ratio = HYPERPARAMETERS["train_ratio"]
    val_ratio = HYPERPARAMETERS["val_ratio"]

    # Calculate the sizes of each dataset
    train_size = int(train_ratio * len(development_dataset))
    val_size = len(development_dataset) - train_size

    # Split the development dataset into training and validation datasets
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(development_dataset, [train_size, val_size], generator=generator)

    # Define batch size and number of workers (adjust as needed)
    batch_size = HYPERPARAMETERS["batch_size"]
    num_workers = HYPERPARAMETERS["num_workers"]

    # Create data loaders
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    class_names = train_dataset.dataset.classes
    print(class_names)
    print("Train set:", len(trainloader)*batch_size)
    print("Val set:", len(valloader)*batch_size)
    print("Test set:", len(testloader)*batch_size)

    train_iterator = iter(trainloader)
    train_batch = next(train_iterator)

    class ASLModel(nn.Module):
        def __init__(self,  num_classes, input_shape, transfer=False, unfreeze_layers=2):
            super().__init__()

            self.transfer = transfer
            self.num_classes = num_classes
            self.input_shape = input_shape

            # transfer learning if weights=True
            self.feature_extractor = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

            if self.transfer:
                # layers are frozen by using eval()
                self.feature_extractor.eval()
                # freeze params
                for param in self.feature_extractor.parameters():
                    param.requires_grad = False

                # Unfreeze the last few layers
                for param in list(self.feature_extractor.parameters())[-unfreeze_layers:]:
                    param.requires_grad = True

            n_features = self._get_conv_output(self.input_shape)
            self.classifier = nn.Linear(n_features, num_classes)

        def _get_conv_output(self, shape):
            batch_size = 1
            tmp_input = torch.autograd.Variable(torch.rand(batch_size, *shape))

            output_feat = self.feature_extractor(tmp_input) 
            n_size = output_feat.data.view(batch_size, -1).size(1)
            return n_size

        # will be used during inference
        def forward(self, x):
            x = self.feature_extractor(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)

            return x

    net = ASLModel(HYPERPARAMETERS["num_classes"], HYPERPARAMETERS["input_shape"], True, unfreeze_layers=2)
    net.to(device)

    criterion = nn.CrossEntropyLoss() # Loss function
    optimizer = torch.optim.AdamW(net.parameters(), lr=HYPERPARAMETERS["learning_rate"])
    scheduler = ExponentialLR(optimizer, gamma=0.9)

    # Log hyperparameters
    wandb.config.update(HYPERPARAMETERS)

    nepochs = HYPERPARAMETERS["epochs"]
    PATH = HYPERPARAMETERS["model_path"] # Path to save the best model

    best_loss = 1e+20
    for epoch in range(nepochs):  # loop over the dataset multiple times
        # Training Loop
        train_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        print(f'{epoch + 1},  train loss: {train_loss / i:.3f},', end = ' ')
        scheduler.step()

        val_loss = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for i, data in enumerate(valloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = net(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
            print(f'val loss: {val_loss / i:.3f}')

            # Log metrics
            wandb.log({"train_loss": train_loss / i, "val_loss": val_loss / i})

            # Save best model
            if val_loss < best_loss:
                print("Saving model")
                torch.save(net.state_dict(), PATH)
                best_loss = val_loss

    print('Finished Training')

    # Load the best model to be used in the test set
    net = ASLModel(HYPERPARAMETERS["num_classes"], HYPERPARAMETERS["input_shape"], False)
    net.load_state_dict(torch.load(PATH))
    net.to(device)
    net.eval()

    print('Starting Testing')

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)  # Move data to GPU
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the test images: {accuracy} %')

    # Log final accuracy
    wandb.log({"test_accuracy": accuracy})

    print(total)

if __name__ == "__main__":
    main()