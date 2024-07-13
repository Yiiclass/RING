import torch
from torch import nn
from skimage.feature import canny
import torchvision.models as models
from torchvision import models
import kornia

class L1_norm(nn.Module):
    def __init__(self):
        super(L1_norm, self).__init__()

    def forward(self,x):
        return torch.abs(x).sum().cuda()
    

class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h =  (x.size()[2]-1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]), 2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size



class PerceptualLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(PerceptualLoss, self).__init__()
        self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()
        self.weights = weights

    def forward(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        content_loss = 0.0
        content_loss += self.weights[0] * self.criterion(x_vgg['relu1_1'], y_vgg['relu1_1'])
        content_loss += self.weights[1] * self.criterion(x_vgg['relu2_1'], y_vgg['relu2_1'])
        content_loss += self.weights[2] * self.criterion(x_vgg['relu3_1'], y_vgg['relu3_1'])
        content_loss += self.weights[3] * self.criterion(x_vgg['relu4_1'], y_vgg['relu4_1'])
        content_loss += self.weights[4] * self.criterion(x_vgg['relu5_1'], y_vgg['relu5_1'])

        return content_loss


class VGG19(torch.nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        features = models.vgg19(pretrained=True).features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()
        self.relu3_4 = torch.nn.Sequential()

        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()
        self.relu4_4 = torch.nn.Sequential()

        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()
        self.relu5_4 = torch.nn.Sequential()

        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])

        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(14, 16):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(16, 18):
            self.relu3_4.add_module(str(x), features[x])

        for x in range(18, 21):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(21, 23):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(23, 25):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(25, 27):
            self.relu4_4.add_module(str(x), features[x])

        for x in range(27, 30):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(30, 32):
            self.relu5_2.add_module(str(x), features[x])

        for x in range(32, 34):
            self.relu5_3.add_module(str(x), features[x])

        for x in range(34, 36):
            self.relu5_4.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        relu3_4 = self.relu3_4(relu3_3)

        relu4_1 = self.relu4_1(relu3_4)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)
        relu4_4 = self.relu4_4(relu4_3)

        relu5_1 = self.relu5_1(relu4_4)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)
        relu5_4 = self.relu5_4(relu5_3)

        out = {
            'relu1_1': relu1_1,
            'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            'relu3_2': relu3_2,
            'relu3_3': relu3_3,
            'relu3_4': relu3_4,

            'relu4_1': relu4_1,
            'relu4_2': relu4_2,
            'relu4_3': relu4_3,
            'relu4_4': relu4_4,

            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            'relu5_3': relu5_3,
            'relu5_4': relu5_4,
        }
        return out


def tv_norm_loss_2d(image):
    """
    Computes the TV norm loss for a 2D image.
    
    Args:
    image (torch.Tensor): A 2D image, such as an image, with shape (N, C, H, W).
    weight (float): A weight parameter to adjust the relative importance of the loss.
    
    Returns:
    torch.Tensor: The TV norm loss for the image.
    """
    # Compute the gradient of the image in the horizontal and vertical directions
    dx = torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])
    dy = torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :])
    
    # Compute the TV norm loss as the L2 norm of the gradient vector
    tv_norm = torch.sqrt(torch.sum(dx**2) + torch.sum(dy**2))
    
    # Scale the TV norm loss by the weight parameter and return it
    return tv_norm


def TV_loss(fake, real, B):
    
    fake_x_diff = torch.abs(fake[:, :, :, :-1] - fake[:, :, :, 1:])
    fake_y_diff = torch.abs(fake[:, :, :-1, :] - fake[:, :, 1:, :])
    fake_loss = torch.sum(fake_x_diff) + torch.sum(fake_y_diff)
    
    real_x_diff = torch.abs(real[:, :, :, :-1] - real[:, :, :, 1:])
    real_y_diff = torch.abs(real[:, :, :-1, :] - real[:, :, 1:, :])
    real_loss = torch.sum(real_x_diff) + torch.sum(real_y_diff)
    
    return torch.abs(fake_loss - real_loss)



def calculate_canny_batch(input):
    # Apply the Canny edge detector to the input images
    
    # canny
    edges = kornia.filters.canny(input, low_threshold=0.1, high_threshold=0.20, kernel_size=(5, 5), sigma=(1, 1), hysteresis=True, eps=1e-06)
    
    # laplacian
    # edges = kornia.filters.laplacian(input, kernel_size=(5, 5), border_type='reflect', normalized=True)
    
    # sobel
    # edges = kornia.filters.sobel(input, normalized=True, eps=1e-06)
    
    # sharp
    # edges =  kornia.filters.unsharp_mask(input, kernel_size=(5, 5), sigma=(1, 1), border_type='reflect')
    
    
    return edges


def canny_loss(fake, real, Batch):
    
    # Convert the PyTorch tensors to numpy arrays
    fake_value = calculate_canny_batch(fake)
    real_value = calculate_canny_batch(real)
    

    fake_value_batch = []
    real_value_batch = []
    for i in range(Batch):
    # Apply calculate_canny to the i-th image in the batch
        fake_tem = fake_value[i]
        fake_value_batch.append(fake_tem)
        real_tem = real_value[i]
        real_value_batch.append(real_tem)
        
    fake_value_batch = torch.stack(fake_value_batch)
    real_value_batch = torch.stack(real_value_batch)
    
    # Compute the pixel-wise absolute difference between the edge maps
    diff = torch.abs(fake_value_batch - real_value_batch)
    # Compute the mean of the absolute difference
    loss = torch.mean(diff)

    return loss





def grad_loss(predicted, target, B):
    sobel_kernel = torch.tensor([
    [1, 0, -1],
    [2, 0, -2],
    [1, 0, -1]
], dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()
    # Calculate gradients of predicted and target images
    grad_pred = torch.abs(torch.nn.functional.conv2d(predicted, sobel_kernel))
    grad_target = torch.abs(torch.nn.functional.conv2d(target, sobel_kernel))

    # Calculate the difference in gradients and square the result
    diff = grad_pred - grad_target
    diff_sq = torch.square(diff)

    # Apply weighting to the loss
    loss = torch.sum(diff_sq)

    return loss