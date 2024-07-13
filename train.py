from options.train_options import TrainOptions
from data import create_dataset
from options.factory import create_optimizer, create_scheduler, print_current_losses, print_current_psnr, save_gray_imgs, lab_to_rgb
import torch
import time
import os
import numpy as np
from tensorboardX import SummaryWriter
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from models import Unet
from PIL import Image

# wandb.init(project='RING', sync_tensorboard=True)
writer = SummaryWriter()

if __name__ == '__main__':
    
    opt = TrainOptions().parse()   # get training options
    train_dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    opt.phase = 'test'
    val_dataset = create_dataset(opt, True)
    dataset_size = len(train_dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)
    model = Unet(opt)
    
    optimizer_Unet = create_optimizer(model, opt)
    scheduler_Unet = create_scheduler(optimizer_Unet, opt)
    
    total_iters = 0                # the total number of training iterations
    best_psnr = 0
    loss_path = os.path.join(opt.checkpoints_dir, opt.name, 'loss.txt')
    psnr_path = os.path.join(opt.checkpoints_dir, opt.name, 'psnr.txt')
    start_epoch = opt.start_epoch
    num_epochs = opt.epochs
    for epoch in range(start_epoch, num_epochs):
        print('Training...', epoch)
        epoch_start_time = time.time()
        total_loss_Unet = 0
        for i, data in enumerate(train_dataset):
            
            total_iters += opt.batch_size
            model(data)
            
            optimizer_Unet.zero_grad()
            loss_Unet = model.get_loss_G(epoch)     
            loss_Unet.backward()
            optimizer_Unet.step()
            
            total_loss_Unet += loss_Unet.item()



        print("epoch {} total_Loss_Unet is {}, current loss is {}, stopping training".format(epoch, total_loss_Unet / dataset_size, loss_Unet.item()))
        
        writer.add_scalar('loss_Unet', total_loss_Unet, epoch) 
        # if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
        #     print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
        #     model.save_networks('latest')
        #     model.save_networks(epoch)
        if epoch == num_epochs - 1:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
        
        scheduler_Unet.step()
        epoch_end_time = time.time()
        print_current_losses(loss_path, epoch, total_loss_Unet, loss_Unet.item(), epoch_end_time - epoch_start_time)
        print('the %d epoch consume %fs...'% (epoch, epoch_end_time - epoch_start_time))
        
        # 验证保存最好结果
        total_psnr = 0
        model.eval()
        
        for i, data in enumerate(val_dataset):
            with torch.no_grad():
                fake_RGB, real_RGB = model(data)
            fake_RGB_numpy = fake_RGB.data[0].cpu().float().numpy()  # convert it into a numpy array
            real_RGB_numpy = real_RGB .data[0].cpu().float().numpy()  # convert it into a numpy array
            
            fake_RGB_numpy = (np.transpose(fake_RGB_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
            
            real_RGB_numpy = (np.transpose(real_RGB_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
            
            image_psnr = compare_psnr(real_RGB_numpy.astype(np.uint8), fake_RGB_numpy.astype(np.uint8))
            total_psnr += image_psnr
        model.train()
        
        writer.add_scalar('test_psnr', total_psnr, epoch) 
        print_current_psnr(psnr_path, epoch, total_psnr / len(val_dataset))
        if total_psnr > best_psnr:
            best_psnr = total_psnr
            print('saving the best model at the end of epoch %d' % (epoch))
            model.save_networks('best')
            print_current_psnr(psnr_path, 'the best', total_psnr / len(val_dataset))
        
writer.close()