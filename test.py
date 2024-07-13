


from options.test_options import TestOptions
from data import create_dataset
from options.factory import save_gray_imgs, lab_to_rgb
import torch
import ntpath
import os
import numpy as np
from models import Unet


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    opt = TestOptions().parse()   # get training options
    opt.serial_batches = True
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)
    model = Unet(opt)
    
    model.eval()
        
    image_dir = os.path.join(opt.results_dir, opt.name)
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
        
    image_dir = os.path.join(opt.results_dir, opt.name, opt.test_epoch + "_epoch_" + opt.name)
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
        
    for i, data in enumerate(dataset):
        image_path = data['paths']
        
        with torch.no_grad():
            fake_L, real_L = model(data)
            
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]
        image_name = '%s_%s.png' % ("fake", name)
        save_path = os.path.join(image_dir, image_name)
            
        print('processing (%04d)-th image... %s' % (i + 1, image_path))
            
        if fake_L is not None:
            save_gray_imgs(fake_L, save_path)