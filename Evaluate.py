import os
import cv2
import numpy as np
import math
from skimage import color
import colour
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

def compare_rmse(img1,img2):
    mse = np.mean((img1 - img2) ** 2)
    return math.sqrt(mse)

def compare_deltae(img1,img2):
    
    image_1_lab = cv2.cvtColor(img1.astype(np.float32)/255,cv2.COLOR_RGB2Lab)
    image_2_lab = cv2.cvtColor(img2.astype(np.float32)/255,cv2.COLOR_RGB2Lab)

    delta = colour.difference.delta_E(image_1_lab,image_2_lab)
    a = np.mean(delta)
    return a

real_path = "/data5/gaoyunyi/RING/results/test_latest/images/"
fake_path = "/data5/gaoyunyi/RING/results/test_latest/images/"
result_path = "/data5/gaoyunyi/RING/results/test_latest/all_benchmark.txt"
index = 0

# rename image 
# for i in range(10,205,5):
#     filename = str(i) + '_fake_B.png'
#     index += 1
#     new_name = "ICVL_" + str(index) + '.png'
#     old_path = os.path.join(fake_path, filename)
#     new_path = os.path.join(fake_path, new_name)
#     # Rename the file
#     os.rename(old_path, new_path)
    
# calculate psnr and ssim
img_list = os.listdir(real_path)
lenth = 0
PSNR = 0
SSIM = 0
RMSE = 0
DeltaE = 0
for img in img_list:
    if "real_B" in img:
        img1_path = real_path + str(img)
        img2_path = fake_path + str(img).replace("real", "fake")
        print(img1_path)
        print(img2_path)
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        print(img1.shape)
        print(img2.shape)
        PSNR_tem = compare_psnr(img1, img2)
        SSIM_tem = compare_ssim(img1, img2, win_size=11, data_range=255, multichannel=True)
        RMSE_tem = compare_rmse(img1, img2)
        DeltaE_tem  = compare_deltae(img1, img2)
        
        PSNR += PSNR_tem
        SSIM += SSIM_tem
        RMSE += RMSE_tem
        DeltaE += DeltaE_tem
        
        with open(result_path, "a", encoding='utf-8')as f:
            f.write(img + " and " + img.replace("real", "fake") + "'s PSNR is " + str(PSNR_tem) + ", SSIM is " + str(SSIM_tem) + ", RMSE is " + str(RMSE_tem) + ", DeltaE is " + str(DeltaE_tem) + ".\n")
        lenth += 1
        
PSNR_mean = PSNR/lenth
SSIM_mean = SSIM/lenth
RMSE_mean = RMSE/lenth
DeltaE_mean = DeltaE/lenth

print("The mean of PSNR is " + str(PSNR_mean) + ".")
print("The mean of SSIM is " + str(SSIM_mean) + ".")
print("The mean of RMSE is " + str(RMSE_mean) + ".")
print("The mean of DeltaE is " + str(DeltaE_mean) + ".")

with open(result_path, "a", encoding='utf-8')as f:
    f.write("The mean of PSNR is " + str(PSNR_mean) + ".\n")
    f.write("The mean of SSIM is " + str(SSIM_mean) + ".\n")
    f.write("The mean of RMSE is " + str(RMSE_mean) + ".\n")
    f.write("The mean of DeltaE is " + str(DeltaE_mean) + ".\n")
    

