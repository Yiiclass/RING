CUDA_VISIBLE_DEVICES=4  python train.py \
--dataroot Datasets/CSS_15S5C/ICVL/739nm \
--name try_first \
--batch_size 2 \
--lr 0.001 \
--epochs 501 \
--ngf 50 \
--n_epochs 200 \
--n_epochs_decay 300 \