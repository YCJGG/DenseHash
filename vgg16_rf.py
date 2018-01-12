import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math

class VGG(nn.Module):
    def __init__(self, num_classes = 1000):
        super(VGG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,64,3, padding=1),
            nn.ReLU(inplace = True),
            nn.Conv2d(64,64,3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(64,128,3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(128,256,3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(256,512,3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=1, padding = 1),

	    # conv5
            nn.Conv2d(512,512,3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3, padding=1),#,dilation=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=1,padding = 1),
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(512,4096,7,padding = 3),
            nn.ReLU (True),
            nn.Dropout(True),
            nn.Conv2d(4096,4096,1),
            nn.ReLU(True),
            nn.Dropout(True),
            nn.AvgPool2d(28),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x 
