import sys
import os
import math
import itertools
from optparse import OptionParser
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim

# Import UNet from the unet package (which now includes quantum components)
from unet import UNet
# from uresnet import UResNet # Not used in this version
# from nestedunet import NestedUNet # Not used in this version

from eval_util import eval_dice, eval_loss, eval_eff_pur
from utils import get_ids, split_ids, split_train_val, get_imgs_and_masks, batch, chw_to_hwc
from utils import h5_utils as h5u


# --- IMPORTANT: PennyLane Installation ---
# To run this code, you need to install PennyLane and its PyTorch interface.
# You can do this using pip:
# pip install pennylane pennylane-lightning[torch]
# -----------------------------------------

def print_lr(optimizer):
    """Prints the learning rate(s) of the optimizer."""
    for param_group in optimizer.param_groups:
        print(param_group['lr'])


def lr_exp_decay(optimizer, lr0, gamma, epoch):
    """
    Applies an exponential decay to the learning rate.
    Note: PyTorch's built-in schedulers are generally preferred for this.
    """
    lr = lr0 * math.exp(-gamma * epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def train_net(net,
              im_tags=['frame_loose_lf0', 'frame_mp2_roi0', 'frame_mp3_roi0'],
              ma_tags=['frame_ductor0'],
              truth_th=100,
              #file_img=[f"data/g4-rec-r{i}.h5" for i in range(10)],
              file_img  = [f"data/g4-rec-{i}_zero.h5" for i in [0,1,3,4,5]],
              #file_mask=[f"data/g4-tru-r{i}.h5" for i in range(10)],
			  file_mask = [f"data/g4-tru-{i}.h5" for i in [0,1,3,4,5,]],
              sepoch=0,
              nepoch=1,
              strain=0,
              ntrain=10, 
              sval=450,
              nval=50,
              batch_size=10,
              lr=0.1,
              val_percent=0.10,  # Not directly used in the current split logic, but kept for context
              save_cp=True,
              gpu=False,
              img_scale=0.5):  # Not directly used in data loading, but kept for context

    dir_checkpoint = 'checkpoints_pennylane/'
    if not os.path.exists(dir_checkpoint):
        os.makedirs(dir_checkpoint)

    # Dataset ID generation
    iddataset = {}
    event_per_file = 10
    event_zero_id_offset = 0

    def id_gen(index):
        return (index // event_per_file, index % event_per_file + event_zero_id_offset)

    iddataset['train'] = [id_gen(i) for i in list(strain + np.arange(ntrain))]
    iddataset['val'] = [id_gen(i) for i in list(sval + np.arange(nval))]

    # Open log file for training progress
    outfile_log = open(dir_checkpoint + '/log', 'a+')

    print(f"Training IDs: {iddataset['train']}", file=outfile_log, flush=True)
    print(f"Validation IDs: {iddataset['val']}", file=outfile_log, flush=True)

    print(f'''
    Starting training:
        Epochs: {nepoch}
        Batch size: {batch_size}
        Learning rate: {lr}
        Training size: {len(iddataset['train'])}
        Validation size: {len(iddataset['val'])}
        Checkpoints: {str(save_cp)}
        CUDA: {str(gpu)}
    ''', file=outfile_log, flush=True)

    N_train = len(iddataset['train'])

    # Optimizer setup
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    # optimizer = optim.Adam(net.parameters(), lr=lr) # Adam is an alternative optimizer

    # Loss function (Binary Cross-Entropy Loss for segmentation)
    criterion = nn.BCELoss()

    print(f'''
    im_tags: {im_tags}
    ma_tags: {ma_tags}
    truth_th: {truth_th}
    ''', file=outfile_log, flush=True)

    # Open CSV files for logging metrics
    outfile_loss_batch = open(dir_checkpoint + '/loss-batch.csv', 'a+')
    outfile_loss = open(dir_checkpoint + '/loss.csv', 'a+')
    outfile_eval_dice = open(dir_checkpoint + '/eval-dice.csv', 'a+')
    outfile_eval_loss = open(dir_checkpoint + '/eval-loss.csv', 'a+')

    # Evaluation image setup
    eval_labels = [
        '75-75',
        '87-85',
    ]
    eval_imgs = []
    eval_masks = []
    for label in eval_labels:
        eval_imgs.append('eval/eval-' + label + '/g4-rec-0.h5')
        eval_masks.append('eval/eval-' + label + '/g4-tru-0.h5')
    outfile_ep = []
    for label in eval_labels:
        outfile_ep.append(open(dir_checkpoint + '/ep-' + label + '.csv', 'a+'))

    # Load model checkpoint if sepoch > 0
    if sepoch > 0:
        net.load_state_dict(torch.load('{}/CP{}.pth'.format(dir_checkpoint, sepoch - 1)))

    # Main training loop
    for epoch in range(sepoch, sepoch + nepoch):
        # Learning rate scheduler (currently using fixed LR or simple exponential decay)
        # scheduler = lr_exp_decay(optimizer, lr, 0.04, epoch)
        scheduler = optimizer  # Using the optimizer directly if no complex scheduler is needed

        print(f'epoch: {epoch} start')
        print(optimizer, file=outfile_log, flush=True)

        # Data loading parameters
        rebin = [1, 10]
        x_range = [476, 952]  # PDVD, V
        y_range = [0, 600]
        z_scale = 4000

        print(f'''
        file_img: {file_img}
        file_mask: {file_mask}
        ''', file=outfile_log, flush=True)

        print(f'Starting epoch {epoch}/{nepoch}.')
        net.train()  # Set network to training mode

        # Prepare data iterators for training, validation, and evaluation
        train_data_iterator = zip(
            h5u.get_chw_imgs(file_img, iddataset['train'], im_tags, rebin, x_range, y_range, z_scale),
            h5u.get_masks(file_mask, iddataset['train'], ma_tags, rebin, x_range, y_range, truth_th)
        )
        val_data_iterator = zip(
            h5u.get_chw_imgs(file_img, iddataset['val'], im_tags, rebin, x_range, y_range, z_scale),
            h5u.get_masks(file_mask, iddataset['val'], ma_tags, rebin, x_range, y_range, truth_th)
        )
        eval_data = []
        for i in range(len(eval_imgs)):
            id_eval = [0]
            eval_data.append(
                zip(
                    h5u.get_chw_imgs(eval_imgs[i], id_eval, im_tags, rebin, x_range, y_range, z_scale),
                    h5u.get_masks(eval_masks[i], id_eval, ma_tags, rebin, x_range, y_range, truth_th)
                )
            )

        epoch_loss = 0

        # Iterate over batches for training
        for i, b in enumerate(batch(train_data_iterator, batch_size)):
            imgs = np.array([item[0] for item in b]).astype(np.float32)
            true_masks = np.array([item[1] for item in b])

            # Optional: plot images/masks for debugging
            # if False:
            #     h5u.plot_mask(b[0][1])
            #     h5u.plot_img(chw_to_hwc(b[0][0]))

            # Convert numpy arrays to PyTorch tensors
            imgs = torch.from_numpy(imgs)
            true_masks = torch.from_numpy(true_masks)

            # Move tensors to GPU if enabled
            if gpu:
                imgs = imgs.cuda()
                true_masks = true_masks.cuda()

            # Forward pass
            masks_pred = net(imgs)

            # Flatten predictions and true masks for BCELoss
            masks_probs_flat = masks_pred.view(-1)
            true_masks_flat = true_masks.view(-1).float()  # Ensure true_masks are float for BCELoss

            # Calculate loss
            loss = criterion(masks_probs_flat, true_masks_flat)
            epoch_loss += loss.item()

            # Print batch loss and log to file
            print(f'{epoch} : {i * batch_size / N_train:.4f} --- loss: {loss.item():.6f}')
            print(f'{i * batch_size / N_train:.4f}, {loss.item():.6f}', file=outfile_loss_batch, flush=True)

            # Backward pass and optimizer step
            optimizer.zero_grad()  # Clear previous gradients
            loss.backward()  # Compute gradients
            scheduler.step()  # Update model parameters

        # Calculate and print average epoch loss
        epoch_loss = epoch_loss / (i + 1)
        print(f'Epoch finished ! Loss: {epoch_loss:.6f}')
        print(f'{epoch:.4f}, {epoch_loss:.6f}', file=outfile_loss, flush=True)

        # Save model checkpoint
        if save_cp:
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP{epoch}.pth')
            print(f'Checkpoint e{epoch} saved !')

        # Perform validation and evaluation
        if True:  # Always perform validation
            # Use itertools.tee to create two independent iterators from val_data_iterator
            # as eval_loss consumes the iterator.
            val1, val2 = itertools.tee(val_data_iterator, 2)

            # val_dice = eval_dice(net, val1, gpu) # Dice evaluation (commented out in original)
            # print(f'Validation Dice Coeff: {epoch:.4f}, {val_dice:.6f}')
            # print(f'{epoch:.4f}, {val_dice:.6f}', file=outfile_eval_dice, flush=True)

            val_loss = eval_loss(net, criterion, val2, gpu)  # Loss evaluation
            print(f'Validation Loss: {epoch:.4f}, {val_loss:.6f}')
            print(f'{epoch:.4f}, {val_loss:.6f}', file=outfile_eval_loss, flush=True)

            # Efficiency and Purity evaluation (commented out in original)
            # for data, out in zip(eval_data, outfile_ep):
            #     ep = eval_eff_pur(net, data, 0.5, gpu)
            #     print(f'{epoch}, {ep[0]:.4f}, {ep[1]:.4f}, {ep[2]:.4f}, {ep[3]:.4f}', file=out, flush=True)


def get_args():
    """Parses command-line arguments for training."""
    parser = OptionParser()
    parser.add_option('--start-epoch', dest='sepoch', default=0, type='int',
                      help='start epoch number')
    parser.add_option('-e', '--nepoch', dest='nepoch', default=1, type='int',
                      help='number of epochs')

    parser.add_option('--start-train', dest='strain', default=0, type='int',
                      help='start sample for training')
    parser.add_option('--ntrain', dest='ntrain', default=10, type='int',
                      help='number of sample for training')
    parser.add_option('--start-val', dest='sval', default=450, type='int',
                      help='start sample for val')
    parser.add_option('--nval', dest='nval', default=50, type='int',
                      help='number of sample for nval')

    parser.add_option('-b', '--batch-size', dest='batchsize', default=1,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.1,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=False, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    parser.add_option('-s', '--scale', dest='scale', type='float',
                      default=0.5, help='downscaling factor of the images')

    (options, args) = parser.parse_args()
    return options


if __name__ == '__main__':
    args = get_args()

    # Set number of threads for Torch (important for CPU performance)
    torch.set_num_threads(15)

    # Image and mask tags for data loading
    # im_tags = ['frame_tight_lf0', 'frame_loose_lf0'] #lt
    im_tags = ['frame_loose_lf0', 'frame_mp2_roi0', 'frame_mp3_roi0']  # l23
    # im_tags = ['frame_loose_lf0', 'frame_tight_lf0', 'frame_mp2_roi0', 'frame_mp3_roi0']    # lt23
    ma_tags = ['frame_deposplat0']
    truth_th = 10

    # Initialize the U-Net model. This will now include the HybridDoubleConv.
    net = UNet(len(im_tags), len(ma_tags))
    # net = UResNet(len(im_tags), len(ma_tags)) # Alternative models
    # net = NestedUNet(len(im_tags),len(ma_tags)) # Alternative models

    # Load pre-trained model if specified
    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))

    # Move model to GPU if enabled
    if args.gpu:
        net.cuda()
        # cudnn.benchmark = True # faster convolutions, but more memory (can be enabled for performance)

    # Start training
    try:
        train_net(net=net,
                  im_tags=im_tags,
                  ma_tags=ma_tags,
                  truth_th=truth_th,
                  sepoch=args.sepoch,
                  nepoch=args.nepoch,
                  strain=args.strain,
                  ntrain=args.ntrain,
                  sval=args.sval,
                  nval=args.nval,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  gpu=args.gpu,
                  img_scale=args.scale)
    except KeyboardInterrupt:
        # Save model state if training is interrupted
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

