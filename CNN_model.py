import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import torch
import vgg16_rf


class cnn_model(nn.Module):
    def __init__(self, original_model, model_name, bit):
        super(cnn_model, self).__init__()
        if model_name =='vgg16':
	    # generate self-defined vgg16_rf network, and copy VGG-16 weights into it
            net = vgg16_rf.VGG()
            net_dict = net.state_dict()
            pretrain_dict = original_model.state_dict()
            pretrain_dict = {k : v for k, v in pretrain_dict.items() if  k in net_dict}
            net_dict.update(pretrain_dict) 
            net.load_state_dict(net_dict)           
    
            self.features = net.features
            self.classifier = net.classifier
            # newly initialized output layer
            self.classifier2 = nn.Sequential(
                nn.Conv2d(4096,bit,1)   
            )

            self.model_name = 'vgg16'           
        # for p in self.features.parameters():
        #     p.requires_grad = False

    def forward(self, x):
        f = self.features(x)
        ff = self.classifier2(f)
        y = self.classifier(ff)
        y = y.view(y.size(0),-1)
        return y,ff

if __name__=="__main__":
    pass
