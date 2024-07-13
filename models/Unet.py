import os
import torch
import torch.nn as nn
from . import networks
from util import LOSS
import cv2


class Unet(nn.Module):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.
    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: ||G(A)-B||_1
        By default, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_128', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0)
            
        return parser
            
    def __init__(self, opt):
        """Initialize the Unet class.
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__()
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.train_type = opt.train_type
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # save all the checkpoints to save_dir
        if opt.preprocess != 'scale_width':  # with [scale_width], input images might have different sizes, which hurts the performance of cudnn.benchmark.
            torch.backends.cudnn.benchmark = True
        self.image_paths = []
        self.metric = 0  # used for learning rate policy 'plateau'
        # define networks (both generator and discriminator)
        # self.N2G = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'FFC', opt.norm,
        #                             not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        if(self.train_type != "pretrain"):
            self.N2G = networks.define_G(3, 1, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.G2R = networks.define_G(1, 2, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

    
            
        if self.isTrain:
            
            self.lambda_loss = opt.lambda_loss
            # define loss functions
            self.criterionMSE =torch.nn.MSELoss()
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionSmoothL1 = torch.nn.SmoothL1Loss()
            self.criterionTV = LOSS.TV_loss
            self.criterioncanny = LOSS.canny_loss
            self.criteriongrad = LOSS.grad_loss
            self.percep = LOSS.PerceptualLoss().to(self.device)
            
        if(opt.isTrain):
            if(self.train_type == "finetune"):
                self.load_networks("pretrain")
            else:
                pass
        else :
            self.load_networks(opt.test_epoch)

    def forward(self, data):
        self.real_NIR = data['real_NIR'].to(self.device)
        self.real_L = data['real_L'].to(self.device)
        # self.real_NIR_hf = data['A_hf'].to(self.device)
        self.real_ab = data['real_ab'].to(self.device)
        self.image_paths = data['paths']
        if(self.train_type == "pretrain"):
            self.fake_ab = self.G2R(self.real_L.float())
            
            self.fake_lab = torch.cat((self.real_L, self.fake_ab), dim=1)
            self.real_lab = torch.cat((self.real_L, self.real_ab), dim=1)
            
        else :
            self.fake_L = self.N2G(self.real_NIR)
            self.fake_ab = self.G2R(self.fake_L)
            
            self.fake_lab = torch.cat((self.fake_L, self.fake_ab), dim=1)
            self.real_lab = torch.cat((self.real_L, self.real_ab), dim=1)

        
        # return self.fake_lab, self.real_lab
            
    def get_loss_G(self, epoch):
        self.loss_G = self.criterionL1(self.fake_lab.float(), self.real_lab.float())
        return self.loss_G

    def L1_norm(self,x):
        return(torch.norm(x,p=1).cuda)

    def save_networks(self, epoch):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        if(self.train_type != "pretrain"):
            save_filename = '%s_N2G.pth' % (epoch)
            save_path = os.path.join(self.save_dir, save_filename)
            print(self.save_dir)
            net = getattr(self, 'N2G')
            if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                torch.save(net.cpu().state_dict(), save_path)
                net.cuda(self.gpu_ids[0])
            else:
                torch.save(net.cpu().state_dict(), save_path)
        save_filename = '%s_G2R.pth' % (epoch)
        save_path = os.path.join(self.save_dir, save_filename)
        print(self.save_dir)
        net = getattr(self, 'G2R')
        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            torch.save(net.cpu().state_dict(), save_path)
            net.cuda(self.gpu_ids[0])
        else:
            torch.save(net.cpu().state_dict(), save_path)
            
            
    def load_networks(self, epoch):
        if(self.train_type != "pretrain"):
            load_filename = '%s_N2G.pth' % (epoch)
            load_path = os.path.join(self.save_dir, load_filename)
            checkpoint = torch.load(load_path, map_location=self.device)
            self.N2G.load_state_dict(checkpoint)
            print("N2G parameters have load")
        # torch.set_default_dtype(torch.float32)
        load_filename = '%s_G2R.pth' % (epoch)
        load_path = os.path.join(self.save_dir, load_filename)
        checkpoint = torch.load(load_path, map_location=self.device)
        self.G2R.load_state_dict(checkpoint)
        print("G2R parameters have load")
        

        
# import os
# import torch
# import torch.nn as nn
# from . import networks
# from util import LOSS
# import cv2


# class Unet(nn.Module):
#     """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.
#     The model training requires '--dataset_mode aligned' dataset.
#     By default, it uses a '--netG unet256' U-Net generator,
#     pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
#     """
#     @staticmethod
#     def modify_commandline_options(parser, is_train=True):
#         """Add new dataset-specific options, and rewrite default values for existing options.

#         Parameters:
#             parser          -- original option parser
#             is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

#         Returns:
#             the modified parser.

#         For pix2pix, we do not use image buffer
#         The training objective is: ||G(A)-B||_1
#         By default, UNet with batchnorm, and aligned datasets.
#         """
#         # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
#         parser.set_defaults(norm='batch', netG='unet_128', dataset_mode='aligned')
#         if is_train:
#             parser.set_defaults(pool_size=0)
            
#         return parser
            
#     def __init__(self, opt):
#         """Initialize the Unet class.
#         Parameters:
#             opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
#         """
#         super().__init__()
#         self.opt = opt
#         self.gpu_ids = opt.gpu_ids
#         self.isTrain = opt.isTrain
#         self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
#         self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # save all the checkpoints to save_dir
#         if opt.preprocess != 'scale_width':  # with [scale_width], input images might have different sizes, which hurts the performance of cudnn.benchmark.
#             torch.backends.cudnn.benchmark = True
#         self.image_paths = []
#         self.metric = 0  # used for learning rate policy 'plateau'
#         # define networks (both generator and discriminator)
#         # self.N2G = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'FFC', opt.norm,
#         #                             not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
#         self.N2G = networks.define_G(3, 1, opt.ngf, opt.netG, opt.norm,
#                                         not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
#         self.G2R = networks.define_G(1, 3, opt.ngf, opt.netG, opt.norm,
#                                         not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

    
            
#         if self.isTrain:
            
#             self.lambda_loss = opt.lambda_loss
#             # define loss functions
#             self.criterionMSE =torch.nn.MSELoss()
#             self.criterionL1 = torch.nn.L1Loss()
#             self.criterionSmoothL1 = torch.nn.SmoothL1Loss()
#             self.criterionTV = LOSS.TV_loss
#             self.criterioncanny = LOSS.canny_loss
#             self.criteriongrad = LOSS.grad_loss
            
#         if(opt.isTrain):
#             pass
#         else :
#             self.load_networks(opt.test_epoch)

#     def forward(self, data):
#         self.real_NIR = data['real_NIR'].to(self.device)
#         self.real_L = data['real_L'].to(self.device)
#         # self.real_NIR_hf = data['A_hf'].to(self.device)
#         self.real_RGB = data['real_RGB'].to(self.device)
#         self.image_paths = data['paths']
        
#         self.fake_L = self.N2G(self.real_NIR)
#         self.fake_RGB = self.G2R(self.fake_L)
        
        
#         return self.fake_RGB, self.real_RGB
            
#     def get_loss_G(self, epoch):
#         # threshold = 150
#         # if epoch <= threshold:
#         #     self.loss_G = self.criterionMSE(self.fake_L, self.real_L)
#         # else :
#         #     Lambda = epoch / (self.opt.epochs - threshold) * self.lambda_loss
#         #     self.loss_G = self.criterionMSE(self.fake_L, self.real_L)  +  Lambda * self.criterioncanny(self.fake_L, self.real_L, self.fake_L.shape[0])

#         self.loss_G = self.criterionL1(self.fake_RGB, self.real_RGB) #+ 0.5 * self.criterionL1(self.fake_L, self.real_L)
#         return self.loss_G

#     def L1_norm(self,x):
#         return(torch.norm(x,p=1).cuda)

#     def save_networks(self, epoch):
#         """Save all the networks to the disk.

#         Parameters:
#             epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
#         """
#         save_filename = '%s_N2G.pth' % (epoch)
#         save_path = os.path.join(self.save_dir, save_filename)
#         print(self.save_dir)
#         net = getattr(self, 'N2G')
#         if len(self.gpu_ids) > 0 and torch.cuda.is_available():
#             torch.save(net.cpu().state_dict(), save_path)
#             net.cuda(self.gpu_ids[0])
#         else:
#             torch.save(net.cpu().state_dict(), save_path)
#         save_filename = '%s_G2R.pth' % (epoch)
#         save_path = os.path.join(self.save_dir, save_filename)
#         print(self.save_dir)
#         net = getattr(self, 'G2R')
#         if len(self.gpu_ids) > 0 and torch.cuda.is_available():
#             torch.save(net.cpu().state_dict(), save_path)
#             net.cuda(self.gpu_ids[0])
#         else:
#             torch.save(net.cpu().state_dict(), save_path)
            
            
#     def load_networks(self, epoch):
#         load_filename = '%s_N2G.pth' % (epoch)
#         load_path = os.path.join(self.save_dir, load_filename)
#         checkpoint = torch.load(load_path, map_location=self.device)
#         self.N2G.load_state_dict(checkpoint)
#         # torch.set_default_dtype(torch.float32)
#         load_filename = '%s_G2R.pth' % (epoch)
#         load_path = os.path.join(self.save_dir, load_filename)
#         checkpoint = torch.load(load_path, map_location=self.device)
#         self.G2R.load_state_dict(checkpoint)