import os
import time
import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data_loader import HDRDataset
from model import FHDR
from options import Options
from util import (
    load_checkpoint,
    make_required_directories,
    mu_tonemap,
    save_checkpoint,
    save_hdr_image,
    save_ldr_image,
    update_lr,
)
from vgg import VGGLoss

def PSNR(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    max_pixel = 1.0  # Assuming the image range is [0, 1]
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.0)

# initialise training options
opt = Options().parse()

# ======================================
# loading data
# ======================================

dataset = HDRDataset(mode="train", opt=opt)
data_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

print("Training samples: ", len(dataset))

# ========================================
# model init
# ========================================

model = FHDR(iteration_count=opt.iter)

# ========================================
# gpu configuration
# ========================================

str_ids = opt.gpu_ids.split(",")
opt.gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >= 0:
        opt.gpu_ids.append(id)

# set gpu device
if len(opt.gpu_ids) > 0:
    assert torch.cuda.is_available()
    assert torch.cuda.device_count() >= len(opt.gpu_ids)

    torch.cuda.set_device(opt.gpu_ids[0])

    if len(opt.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

    model.cuda()

# ========================================
#  initialising losses and optimizer
# ========================================

l1 = torch.nn.L1Loss()
perceptual_loss = VGGLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999))

make_required_directories(mode="train")

# ==================================================
#  loading checkpoints if continuing training
# ==================================================

if opt.continue_train:
    try:
        start_epoch, model = load_checkpoint(model, opt.ckpt_path)

    except Exception as e:
        print(e)
        print("Checkpoint not found! Training from scratch.")
        start_epoch = 1
        model.apply(weights_init)
else:
    start_epoch = 1
    model.apply(weights_init)

if opt.print_model:
    print(model)

# ========================================
#  training
# ========================================

for epoch in range(start_epoch, opt.epochs + 1):

    epoch_start = time.time()
    running_loss = 0
    running_psnr = 0  # Initialize PSNR accumulator for the epoch

    # check whether LR needs to be updated
    if epoch > opt.lr_decay_after:
        update_lr(optimizer, epoch, opt)

    print("Epoch: ", epoch)

    for batch, data in enumerate(tqdm(data_loader, desc="Batch %")):

        optimizer.zero_grad()

        input = data["ldr_image"].data.cuda()
        ground_truth = data["hdr_image"].data.cuda()

        # forward pass ->
        output = model(input)

        l1_loss = 0
        vgg_loss = 0
        psnr_values = torch.zeros(len(output)).cuda()  # Initialize a tensor to store PSNR values

        # tonemapping ground truth ->
        mu_tonemap_gt = mu_tonemap(ground_truth)

        # computing loss for n generated outputs (from n-iterations) ->
        for i, image in enumerate(output):
            l1_loss += l1(mu_tonemap(image), mu_tonemap_gt)
            vgg_loss += perceptual_loss(mu_tonemap(image), mu_tonemap_gt)
            psnr_values[i] = PSNR(mu_tonemap(image), mu_tonemap_gt)  # Store PSNR value in tensor

        # averaged over n iterations
        l1_loss /= len(output)
        vgg_loss /= len(output)
        psnr_value = torch.mean(psnr_values)  # Calculate mean PSNR value

        # averaged over batches
        l1_loss = torch.mean(l1_loss)
        vgg_loss = torch.mean(vgg_loss)

        # FHDR loss function
        loss = l1_loss + (vgg_loss * 10)

        # output is the final reconstructed image i.e. last in the array of outputs of n iterations
        output = output[-1]

        # backpropagate and step
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_psnr += psnr_value.item()  # Track PSNR for the epoch

        if (batch + 1) % opt.log_after == 0:  # logging batch count and loss value
            print(
                "Epoch: {} ; Batch: {} ; Training loss: {}; PSNR: {}".format(
                    epoch, batch + 1, running_loss / opt.log_after, running_psnr / opt.log_after
                )
            )
            running_loss = 0
            running_psnr = 0  # Reset PSNR for the next logging interval

    epoch_finish = time.time()
    time_taken = (epoch_finish - epoch_start) // 60

    avg_loss = running_loss / len(data_loader)
    avg_psnr = running_psnr / len(data_loader)

    print("End of epoch {}. Time taken: {} minutes. Average loss: {}. Average PSNR: {}.".format(epoch, int(time_taken), avg_loss, avg_psnr))

    if epoch % opt.save_ckpt_after == 0:
        save_checkpoint(epoch, model)

print("Training complete!")
