import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
        
def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)): 
        m.weight.data.normal_(0.0, 0.02)
        # m.bias.data.fill_(0)
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0.0, 0.02)
        # m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class LinearBlock(nn.Module):
    def __init__(self, in_f, out_f, bn=True, act='relu'):
        super(LinearBlock, self).__init__()
        self.bn = bn
        self.act = act
        
        if act == 'relu':
            self.act_f = F.relu
        elif act == 'lrelu':
            self.act_f = F.leaky_relu
        else:
            self.act_f = lambda x: x
            
        self.fc = nn.Linear(in_f, out_f, bias=False)
        self.bn = nn.BatchNorm1d(out_f)
    
    def forward(self, x):
        out = self.fc(x)
        if self.bn: out = self.bn(out)
        out = self.act_f(out)
            
        return out

class ConvBlock(nn.Module):
    def __init__(self, in_f, out_f, bn=True, act='relu', mode='up'):
        super(ConvBlock, self).__init__()
        self.bn = bn
        self.act = act
        
        if act == 'relu':
            self.act_f = F.relu
        elif act == 'lrelu':
            self.act_f = F.leaky_relu
        else:
            self.act_f = lambda x: x
            
        self.conv1 = nn.Conv2d(in_f, in_f, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(in_f)
        if mode == 'up':
            self.conv2 = nn.ConvTranspose2d(in_f, out_f, 4, 2, 1, bias=False)
        elif mode == 'down':
            self.conv2 = nn.Conv2d(in_f, out_f, 4, 2, 1, bias=False)
            
        self.bn2 = nn.BatchNorm2d(out_f)
    
    def forward(self, x):
        out = self.conv1(x)
        if self.bn: out = self.bn1(out)
        out = self.act_f(out)
        out = self.conv2(out)
        if self.bn: out = self.bn2(out)
        out = self.act_f(out)
            
        return out
    

class G(nn.Module):
    def __init__(self, features):
        super(G, self).__init__()
        self.features = features
        self.linear = LinearBlock(features, 4*4*features)
        self.upconv1 = ConvBlock(features, features)
        self.upconv2 = ConvBlock(features, features)
        self.upconv3 = ConvBlock(features, features)
        self.upconv4 = ConvBlock(features, features)
        self.upconv5 = ConvBlock(features, features, bn=False, act=None)
        self.final = nn.Conv2d(features, 3, 3, 1, 1, bias=False)

    def forward(self, x):
        out = self.linear(x)
        out = out.view([-1,self.features,4,4])
        out = self.upconv1(out)
        out = self.upconv2(out)
        out = self.upconv3(out)
        out = self.upconv4(out)
        out = self.upconv5(out)
        out = self.final(out)
        out = F.tanh(out)
        
        return out

    
class D(nn.Module):
    def __init__(self, features):
        super(D, self).__init__()
        self.conv1 = ConvBlock(3, features, act='lrelu', mode='down')
        self.conv2 = ConvBlock(features, features, act='lrelu', mode='down')
        self.conv3 = ConvBlock(features, features, act='lrelu', mode='down')
        self.conv4 = ConvBlock(features, features, act='lrelu', mode='down')
        self.conv5 = ConvBlock(features, features, act='lrelu', mode='down')
        self.conv6 = ConvBlock(features, features, act='lrelu', mode='down')
        self.fc1 = LinearBlock(128*2*2, 512, act='lrelu')
        self.fc2 = nn.Linear(512, 1, bias=False)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = out.view([-1, 128*2*2])
        out = self.fc1(out)
        out = self.fc2(out)
        out = out.mean()
        
        return out