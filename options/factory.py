from timm import scheduler
import torch.nn as nn
import numpy as np
import torch
from torch import optim as optim
from torch.optim import lr_scheduler
from skimage import io
from PIL import Image
import warnings
from skimage import color, io, transform

def create_optimizer(model, opt):

    parameters_Unet = model.parameters()
    optimizer_Unet = optim.Adam(parameters_Unet, lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=0)
    # optimizer_Unet = optim.SGD(parameters_Unet, lr=args['lr'], momentum=args['momentum'], nesterov=True, weight_decay=weight_decay)

    return optimizer_Unet


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def create_scheduler(optimizer, opt):
    
    def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    return scheduler


def save_gray_imgs(fake_B, path):
    if not isinstance(fake_B, np.ndarray):
        if isinstance(fake_B, torch.Tensor):  # get the data from a variable
            fake_B_tensor = fake_B.data
         
        fake_B_numpy = fake_B_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if fake_B_numpy.shape[0] == 1:  # grayscale to RGB
            fake_B_numpy = np.tile(fake_B_numpy, (3, 1, 1))
        fake_B_numpy = (np.transpose(fake_B_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:
        fake_B_numpy = fake_B
        

    io.imsave(path, fake_B_numpy.astype(np.uint8))
    
    # image_pil = Image.fromarray(fake_B.astype(np.uint8))
    # image_pil.save(path)
    
def print_current_losses(filename, epoch, total, loss, t_comp):
    """print current losses on console; also save the losses to the disk
    Parameters:
        epoch (int) -- current epoch
        iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
        losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
        t_comp (float) -- computational time per data point (normalized by batch_size)
    """
    message = '(epoch: %d, total_Loss_Unet is: %.3f, time: %.3f, current_loss_Unet: %.3f.) ' % (epoch, total, t_comp, loss)
    with open(filename, "a") as log_file:
        log_file.write('%s\n' % message)  # save the message
        
def print_current_psnr(filename, epoch, psnr):
    """print current losses on console; also save the losses to the disk
    Parameters:
        epoch (int) -- current epoch
        iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
        losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
        t_comp (float) -- computational time per data point (normalized by batch_size)
    """
    message = '(epoch: %s, psnr: %.3f.) ' % (str(epoch), psnr)
    with open(filename, "a") as log_file:
        log_file.write('%s\n' % message)  # save the message
        
        
def lab_to_rgb(img):
    assert img.dtype == np.float32

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        return (255 * np.clip(color.lab2rgb(img), 0, 1)).astype(np.uint8)