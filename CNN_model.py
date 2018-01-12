import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import torch
import vgg11
import vgg16
import vgg16_rf


class cnn_model(nn.Module):
    def __init__(self, original_model, model_name, bit):
        super(cnn_model, self).__init__()
        if model_name == 'vgg11':
            net = vgg11.vgg11()
            net_dict = net.state_dict()
            pretrain_dict = original_model.state_dict()
            
            pretrain_dict = {k : v for k, v in pretrain_dict.items() if k in net_dict}
            net_dict.update(pretrain_dict) 
            net.load_state_dict(net_dict)           

            self.features = net.features
            cl1 = nn.Linear(25088, 4096)
            cl1.weight = net.classifier[0].weight
            cl1.bias = net.classifier[0].bias

            cl2 = nn.Linear(4096, 4096)
            cl2.weight = net.classifier[3].weight
            cl2.bias = net.classifier[3].bias

            self.classifier = nn.Sequential(
                cl1,
                nn.ReLU(inplace=True),
                nn.Dropout(),
                cl2,
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, bit),
            )
            self.model_name = 'vgg11'
        if model_name == 'alexnet':
            self.features = original_model.features
            cl1 = nn.Linear(256 * 6 * 6, 4096)
            cl1.weight = original_model.classifier[1].weight
            cl1.bias = original_model.classifier[1].bias

            cl2 = nn.Linear(4096, 4096)
            cl2.weight = original_model.classifie[4].weight
            cl2.bias = original_model.classifier[4].bias

            self.classifier = nn.Sequential(
                nn.Dropout(),
                cl1,
                nn.ReLU(inplace=True),
                nn.Dropout(),
                cl2,
                nn.ReLU(inplace=True),
                nn.Linear(4096, bit),
            )
            self.model_name = 'alexnet'
        if model_name =='vgg16':
            net = vgg16_rf.VGG()
       
            net_dict = net.state_dict()
        
            pretrain_dict = original_model.state_dict()
          
            #pretrain_dict = {('features.'+ k) : v for k, v in pretrain_dict.items() if ('features.'+ k) in net_dict}
            pretrain_dict = {k : v for k, v in pretrain_dict.items() if  k in net_dict}
            net_dict.update(pretrain_dict) 
            net.load_state_dict(net_dict)           
    
            self.features = net.features
            self.classifier = net.classifier
            
            self.classifier2 = nn.Sequential(
                nn.Linear(4096,bit)   
               
            )

            self.model_name = 'vgg16'           
        # for p in self.features.parameters():
        #     p.requires_grad = False

    def forward(self, x):
        f = self.features(x)
        #print(f.size())
        if self.model_name == 'vgg11':
            f = f.view(f.size(0), -1)
        if self.model_name == 'alexnet':
            f = f.view(f.size(0), 256 * 6 * 6)
        if self.model_name == 'vgg16':
            f = self.classifier(f)
            #print('test',f.size())
            f = f.view(f.size(0),-1)
        y = self.classifier2(f)
        #print(y.size())
        return y

if __name__=="__main__":
    pass
    # alexnet = models.alexnet(pretrained=True)
    #print(alexnet)
    # vgg11_classifier = cnn_model(vgg11, 'vgg11', 1000)
    #
    # vgg11 = vgg11.cuda()
    # vgg11_classifier = vgg11_classifier.cuda()
    #
    # # evaluation phase
    # vgg11.eval()
    # vgg11_classifier.eval()
    #
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    #
    # train_loader = torch.utils.data.DataLoader(
    #     datasets.ImageFolder('data/img/', transforms.Compose([
    #         transforms.Scale(256),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         normalize,
    #     ])),
    #     batch_size=1,
    #     shuffle=False,
    #     num_workers=1,
    # )
    #
    # criterion = nn.CrossEntropyLoss().cuda()
    # for i, (input, target) in enumerate(train_loader):
    #     input_var = Variable(input.cuda())
    #     output1 = vgg11(input_var)
    #     output2 = vgg11_classifier(input_var)
    #
    #     print(output1)
    #     print(output2)
    #
