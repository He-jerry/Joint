import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
class residual(nn.Module):
  def __init__(self):
    super(residual,self).__init__()
    self.conv1=nn.Conv2d(256,256,4,2,padding=1)
    self.bn1=nn.BatchNorm2d(256)
    self.relu1=nn.ReLU(inplace=True)

    self.conv2=nn.Conv2d(256,256,4,2,padding=1)
    self.bn2=nn.BatchNorm2d(256)
    self.relu2=nn.ReLU(inplace=True)
  def forward(self,x):
    x1=self.relu1(self.bn1(self.conv1(x)))
    x2=self.bn2(self.conv2(x1))
    x2=F.interpolate(x2,size=[x.shape[3],x.shape[2]])
    x3=x2+x
    x3=self.relu2(x3)
    return x3

class basicblock(nn.Module):
  def __init__(self,in_channels,out_channels,kz,st,pd):
    super(basicblock,self).__init__()
    self.conv=nn.Conv2d(in_channels,out_channels,kz,st,padding=pd)
    self.bn=nn.InstanceNorm2d(out_channels)
    self.relu=nn.LeakyReLU(0.2,True)
  def forward(self,x):
    return self.relu(self.bn(self.conv(x)))
    
class basictpblock(nn.Module):
  def __init__(self,in_channels,out_channels,kz,st,pd):
    super(basictpblock,self).__init__()
    self.conv=nn.ConvTranspose2d(in_channels,out_channels,kz,st,padding=pd)
    self.bn=nn.InstanceNorm2d(out_channels)
    self.relu=nn.LeakyReLU(0.2,True)
  def forward(self,x):
    return self.relu(self.bn(self.conv(x)))
    

class sepnet(nn.Module):
  def __init__(self):
    super(sepnet,self).__init__()
    self.b1=basicblock(3,64,3,1,1)
    self.b2=basicblock(64,128,3,1,1)
    self.b3=basicblock(128,256,3,1,1)

    self.r1=residual()
    self.r2=residual()
    self.r3=residual()
    self.r4=residual()
    self.r5=residual()
    self.r6=residual()
    self.r7=residual()
    self.r8=residual()
    self.r9=residual()

    #reflection
    self.t11=basictpblock(256,128,3,1,1)
    self.t12=basictpblock(128,64,3,1,1)
    self.t13=nn.ConvTranspose2d(64,3,3,1,1)
    self.tanh1=nn.Tanh()

    #background
    self.t21=basictpblock(256,128,3,1,1)
    self.t22=basictpblock(128,64,3,1,1)
    self.t23=nn.ConvTranspose2d(64,3,3,1,1)
    self.tanh2=nn.Tanh()

    #areamap
    self.t31=basictpblock(256,128,3,1,1)
    self.t32=basictpblock(128,64,3,1,1)
    self.t33=nn.ConvTranspose2d(64,1,3,1,1)
    #self.tanh3=nn.Sigmoid()
  def forward(self,x):
    stg1=self.b3(self.b2(self.b1(x)))
    #stg1=F.interpolate(stg1,scale_factor=8)
    stg2=self.r9(self.r8(self.r7(self.r6(self.r5(self.r4(self.r3(self.r2(self.r1(stg1)))))))))

    #reflection
    rf=self.tanh1(self.t13(self.t12(self.t11(stg2))))

    #background
    bg=self.tanh2(self.t23(self.t22(self.t21(stg2))))

    #areamap
    areamap=self.t33(self.t32(self.t31(stg2)))

    #rf=F.interpolate(rf,size=[256,256])
    #bg=F.interpolate(bg,size=[256,256])
    #areamap=F.interpolate(areamap,size=[256,256])

    return rf,bg,areamap
    

class gennet(nn.Module):
  def __init__(self):
    super(gennet,self).__init__()
    self.b11=basicblock(3,64,4,2,1)
    self.b12=basicblock(64,128,4,2,1)
    self.b13=basicblock(128,256,4,2,1)

    self.b21=basicblock(3,64,4,2,1)
    self.b22=basicblock(64,128,4,2,1)
    self.b23=basicblock(128,256,4,2,1)

    self.b31=basicblock(1,64,4,2,1)
    self.b32=basicblock(64,128,4,2,1)
    self.b33=basicblock(128,256,4,2,1)

    self.r1=residual()
    self.r2=residual()
    self.r3=residual()
    self.r4=residual()
    self.r5=residual()
    self.r6=residual()
    self.r7=residual()
    self.r8=residual()
    self.r9=residual()

    self.t11=basictpblock(256,128,4,2,1)
    self.t12=basictpblock(128,64,4,2,1)
    self.t13=nn.ConvTranspose2d(64,3,4,2,1)
    self.tanh1=nn.Tanh()
  def forward(self,bg,rf,areamap):
    b=self.b13(self.b12(self.b11(bg)))
    r=self.b23(self.b22(self.b21(rf)))
    am=self.b33(self.b32(self.b31(areamap)))

    stg1=b+r*F.sigmoid(am)

    stg2=self.r9(self.r8(self.r7(self.r6(self.r5(self.r4(self.r3(self.r2(self.r1(stg1)))))))))

    mix=self.tanh1(self.t13(self.t12(self.t11(stg2))))

    #mix=F.interpolate(mix,size=[256,256])


    return mix
    
class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A):
        # Concatenate image and condition image by channels to produce input
        return self.model(img_A)
        
class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()        
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]        

    def forward(self, x, y):              
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())        
        return loss
        
from torchvision import models
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)        
        h_relu3 = self.slice3(h_relu2)        
        h_relu4 = self.slice4(h_relu3)        
        h_relu5 = self.slice5(h_relu4)                
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out