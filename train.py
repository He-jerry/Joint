#! /usr/bin/env python


import torch
import torchvision
import torch.nn as nn
from dataset import ImageDataset
import torchvision.transforms as transforms
from torchvision.utils import save_image
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from torchvision import datasets
from tqdm import tqdm
from torch.autograd import Variable
import torch.nn.functional as F

from network import sepnet,gennet,Discriminator,VGGLoss

def tensor2im(input_image, imtype=np.uint8):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225) 
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        for i in range(len(mean)):
            image_numpy[i] = image_numpy[i] * std[i] + mean[i]
        image_numpy = image_numpy * 255
        image_numpy = np.transpose(image_numpy, (1, 2, 0))  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)

def save_img(im, path,h,w):
    im_grid = im
    im_numpy = tensor2im(im_grid) 
    im_array = Image.fromarray(im_numpy)
    im_array=im_array.resize((h,w))
    im_array.save(path)

trainloader = DataLoader(
    ImageDataset(transforms_=None),
    batch_size=4,
    shuffle=False,drop_last=True
)
print("data length:",len(trainloader))
def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if nets is not None:
      for param in nets.parameters():
        param.requires_grad = requires_grad
snet=sepnet()
gnet=gennet()
discriminator=Discriminator()

snet=nn.DataParallel(snet)
gnet=nn.DataParallel(gnet)
discriminator=nn.DataParallel(discriminator)

snet.train()
gnet.train()
discriminator.train()

snet.cuda()
gnet.cuda()
discriminator.cuda()

criterionMSE=nn.MSELoss().cuda()
criterionL1 = nn.L1Loss().cuda()
criterionVGG=VGGLoss()
criterion_bce=torch.nn.BCEWithLogitsLoss().cuda()

criterionVGG.cuda()

lp=10
le=0.5
llr=0.5
lc=2
k=0
optimizerD = torch.optim.Adam(discriminator.parameters(),lr=0.0002, betas=(0.5, 0.999))
optimizerG1 = torch.optim.Adam(snet.parameters(),lr=0.0002,betas=(0.5, 0.999))
optimizerG2 = torch.optim.Adam(gnet.parameters(),lr=0.0002,betas=(0.5, 0.999))

eopchnum=180
k=0
print("start training")
for epoch in range(0, eopchnum):
   print("epoch:",epoch)
   iteration=0
   train_iterator = tqdm(trainloader, total=len(trainloader))
   for total in train_iterator:
     iteration=iteration+1
     #total={"mix": mixs, "mask": masks,"ref": refs, "bg": bgs}
     real_mix=total["mix"]
     real_mask=total["mask"]
     real_ref=total["ref"]
     real_bg=total["bg"]
     
     real_mix=real_mix.cuda()
     real_mask=real_mask.cuda()
     real_ref=real_ref.cuda()
     real_bg=real_bg.cuda()
     
     outrf,outbg,outmap=snet(real_mix)
     outmix=gnet(real_bg,real_ref,real_mask)
     
     if iteration==15:
        fk_B=tensor2im(outrf.cpu()[0,:,:,:])
        save_img(fk_B,"/public/zebanghe2/joint/sample/"+"sample"+'_ref'+str(epoch)+'.jpg',384,384)
        fk_B=tensor2im(outbg.cpu()[0,:,:,:])
        save_img(fk_B,"/public/zebanghe2/joint/sample/"+"sample"+'_trans'+str(epoch)+'.jpg',384,384)
        fk_B=tensor2im(F.sigmoid(outmap).cpu()[0,:,:,:])
        save_img(fk_B,"/public/zebanghe2/joint/sample/"+"sample"+'_map'+str(epoch)+'.png',384,384)
        fk_B=tensor2im(outmix.cpu()[0,:,:,:])
        save_img(fk_B,"/public/zebanghe2/joint/sample/"+"sample"+'_mix'+str(epoch)+'.jpg',384,384)
     
     optimizerD.zero_grad()
     
     set_requires_grad(discriminator, True)
     optimizerD.zero_grad()
     recon_real=discriminator(real_mix.detach())
     recon_fake=discriminator(outmix.detach())
     valid=torch.ones(recon_real[0][0].shape).cuda()
     fake=torch.ones(recon_fake[0][0].shape).cuda()
     errD_real=criterionMSE(recon_real[0][0],valid)
     errD_fake=criterionMSE(recon_fake[0][0],fake)
     
     errD=errD_real+errD_fake
     #errDs=Variable(errD,requires_grad=True)
     errD.backward()
     optimizerD.step()
     set_requires_grad(discriminator, False)
     #del errD,recon_real,recon_fake
     
     optimizerG1.zero_grad()
     optimizerG2.zero_grad()
     
     lossprecm=criterionL1(outmix,real_mix)
     lossvggm=criterionMSE(outmix,real_mix)
     recon_fake=discriminator(outmix.detach())
     lossq1=criterionMSE(recon_fake[0][0],valid)
     
     errG2=lossprecm+lossvggm+lossq1
     #errG2=Variable(errG2,requires_grad=True)
     errG2.backward()
     optimizerG2.step()
     #del errG2,outmix
     
     lossprecr=criterionL1(outrf,real_ref)
     lossprecb=criterionL1(outbg,real_bg)
     
     
     lossvggr=criterionMSE(outrf,real_ref)
     lossvggb=criterionMSE(outbg,real_bg)
     
     #lossvgg=criterionVGG(real_mix,outmix)
     
     lossbce=criterion_bce(outmap,real_mask)
     
     genmix=gnet(outrf.detach(),outbg.detach(),outmap.detach())
     
     lossstg2=criterionMSE(genmix,real_mix)
     
     recon_fake=discriminator(outmix.detach())
     
     lossq1=criterionMSE(recon_fake[0][0],valid)
     
     if epoch>5:
       errG=(lossprecr+lossvggr)+(lossprecb+lossvggb)+lossbce+lossstg2+lossq1
     else:
       errG=(lossprecr+lossvggr)+(lossprecb+lossvggb)
     #errGs=Variable(errG,requires_grad=True)
     errG.backward()
     optimizerG1.step()
     #del outrf,outbg
     
     train_iterator.set_description("batch:%3d,iteration:%3d,loss_g:%3f,loss_d:%3f,loss_bce:%3f,loss_MSE:%3f,loss_l1:%3f"%(epoch+1,iteration,errG.item(),errD.item(),lossbce.item(),lossvggm.item()+lossvggr.item()+lossvggb.item(),lossprecm.item()+lossprecr.item()+lossprecb.item()))
   if epoch%10==9:
     torch.save(snet,"snet_p1_%3d.pth"%epoch)
     torch.save(gnet,"gnet_p1_%3d.pth"%epoch)
     torch.save(discriminator,'discriminator1_p1_%3d.pth'%epoch)
     
     
     
     