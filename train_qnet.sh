# model=unet-explr-l23-cosmic500-e60/CP60-450.pth
# model=uresnet-l23-cosmic500-t1/CP20-450.pth

# model=model/unet-l23-cosmic500-e50/CP1-450.pth

# model=unet-adam-l23-cosmic500-e50/CP50-450.pth


time python train_qnet.py -s 0.5 \
    --start-epoch 0 --nepoch 30 \
    --start-train 0 --ntrain 25  \
    --start-val 25 --nval 5 \