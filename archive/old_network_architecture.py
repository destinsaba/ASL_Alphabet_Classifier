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
    "epochs": 25,
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
        
def main():
    # Display the model architecture
    model = ASLModel(HYPERPARAMETERS['num_classes'], HYPERPARAMETERS['input_shape'])
    print(model)

    # Print the number of parameters in each layer of the last Bottleneck block and the fully connected layer
    for name, param in model.feature_extractor.layer4.named_parameters():
        print(f"{name} has {param.numel()} parameters")

    for name, param in model.feature_extractor.fc.named_parameters():
        print(f"{name} has {param.numel()} parameters")

    # Print the number of parameters in the last two layers of the feature extractor
    for param in list(model.feature_extractor.parameters())[-2:]:
        print(f"{param.shape} has {param.numel()} parameters")

if __name__ == "__main__":
    main()