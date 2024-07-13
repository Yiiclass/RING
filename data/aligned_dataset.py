import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
import torch
from PIL import Image
import torchvision.transforms as transforms
import cv2
from skimage import color
import numpy as np

class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.
    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        #self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        self.AB_paths =[]
        self.train_type = opt.train_type
        for self.root, self.dirs, self.files in os.walk(self.dir_AB):
            for self.file in self.files:
                self.full_name = os.path.join(self.root, self.file)
                self.AB_paths.append(self.full_name)

        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc


    def numpy_to_torch(self, img):
        tensor = torch.from_numpy(np.moveaxis(img, -1, 0))      # [c, h, w]
        return tensor.type(torch.float32)


    def __getitem__(self, index):
        #输出的是一个14的tensor
        path = self.AB_paths[index]
        AB = Image.open(path).convert('RGB')
        
        if(self.train_type == "pretrain"):
            NIR = AB.resize((256, 256))
            RGB = AB.resize((256, 256))
        # split AB image into NIR and RGB
        else:
            w, h = AB.size
            w2 = int(w / 2)
            NIR = AB.crop((0, 0, w2, h))
            RGB = AB.crop((w2, 0, w, h))

        NIR = NIR.resize((256, 256))
        RGB = RGB.resize((256, 256))

        
        LAB = color.rgb2lab(np.array(RGB) / 255.0)
        LAB = LAB.transpose((2, 0, 1))
        real_L = torch.from_numpy((LAB[0, :, :] - 50) / 50).unsqueeze(0)
        real_a = torch.from_numpy(LAB[1, :, :] / 128.0).unsqueeze(0)
        real_b = torch.from_numpy(LAB[2, :, :] / 128.0).unsqueeze(0)

        real_ab = torch.cat([real_a, real_b], dim=0)
        
        real_NIR =  np.array(NIR) / 127.5 - 1
        real_NIR = torch.from_numpy(real_NIR.transpose((2, 0, 1))).float()
        # real_NIR =  all_channle_transform(NIR)

        return {'real_NIR': real_NIR, 'real_L': real_L, 'real_ab': real_ab, 'paths': self.AB_paths[index]}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)