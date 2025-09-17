import torch.nn as nn
import torch.nn.functional as F

class SerbianBanknoteCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.dropout = nn.Dropout(0.5)
        
        self.fc1 = nn.Linear(128*30*30, 256) 
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = self.feature_extraction(x)        
        x = x.view(x.size(0), -1)              
        x = self.dropout(F.relu(self.fc1(x)))   
        x = self.fc2(x)                         
        return x
