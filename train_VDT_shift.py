import argparse
import os.path

import torch
from torch import nn,optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset.dataset_loader import VideoDataset
from utils import extend_batch, extend_motion, fw_diffusion_vid, fw_diffusion_mot, MixedSequenceMaskGenerator
from TimeSformer.VDT_vm import VDT
import wandb
import random


if __name__ == "__main__":
    # initialization 
    nhead = 6
    depth = 12
    patch_size= 2
    hidden_size = 768
    epoch = 100
    in_channels = 4
    input_size = 8
    checkpoint_path = os.path.join('checkpoints_mmVDT_vae_seg')

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(DEVICE)
    # The parser setting
    parser = argparse.ArgumentParser()
    parser = VideoDataset.add_model_specific_args(parser)
    parser.add_argument('--data_path', type=str, default='dataset/data_20240807_processed')

    parser.add_argument('--sequence_length', type=int, default=16)
    parser.add_argument('--resolution', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--LR_RATE', type=int, default=1e-4)
    parser.add_argument('--NUM_EPOCHS', type=int, default=100)
    parser.add_argument('--iteration_num', type=int, default=1000)
    args = parser.parse_args(args=[])


    # Load the Dataset
    Dataset = VideoDataset
    dataset = Dataset(args.data_path, args.sequence_length,rgb=False, train=True, resolution=args.resolution)

    # dataloader for the trainning
    dataloader_train = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=True
    )

    # dataloader for the evaluation
    dataset = Dataset(args.data_path, args.sequence_length, train=False, resolution=args.resolution)
    dataloader_val = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False
    )

    input_mot_channel = args.sequence_length * 2
    mask_generator = MixedSequenceMaskGenerator()

    # The parameters for forward diffusion process
    beta = torch.linspace(1e-4, 0.02, args.iteration_num) 
    alpha = 1.0-beta
    alpha_cum = torch.tensor([alpha[0]] + [torch.prod(alpha[:i+1]) for i in range(1, len(beta))])


   # Load the VDT model and VAE
    eps_est_model = VDT(
    input_size=input_size,
    patch_size=patch_size,
    in_channels=in_channels,
    hidden_size=hidden_size,
    depth=depth,
    num_frames=16,
    num_heads=nhead,
    mlp_ratio=4.0,
    learn_sigma=False
    ).to(DEVICE)

    from diffusers.models import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-mse").to(DEVICE)
    vae.encoder.conv_in = nn.Conv2d(9, vae.encoder.conv_in.out_channels, kernel_size=3, stride=1, padding=1)
    vae.decoder.conv_out = nn.Conv2d(vae.decoder.conv_out.in_channels, 9, kernel_size=3, stride=1, padding=1)

    modelData = 'checkpoints_vae_seg/vae_40.pth.tar'
    checkpoint = torch.load(modelData)
    vae.load_state_dict(checkpoint['model_state_dict'])
    vae.to(DEVICE)

    # freeze vae
    for param in vae.parameters():
        param.requires_grad = False
    

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="seg_vae_VDT",

        # track hyperparameters and run metadata
        config={
        "learning_rate": args.LR_RATE,
        "epochs": args.NUM_EPOCHS,
        }
    )
    wandb.define_metric("Epoch")
    wandb.define_metric("mean_video_loss", step_metric="Epoch")
    wandb.define_metric("mean_motion_loss", step_metric="Epoch")

    #################################### Train loop #########################################
    from utils import mask_vid_mot
    optimizer = optim.Adam(eps_est_model.parameters(), lr=args.LR_RATE)
    loss_l1 = nn.L1Loss()
    loss_l2 = nn.MSELoss()
    loss_CE = nn.CrossEntropyLoss()
    #start Training
    for epoch in range(args.NUM_EPOCHS):
        mean_loss_vid = 0
        mean_loss_mot = 0
        eps_est_model.train()
        loop = tqdm(dataloader_train)

        for x in loop:
            loop.set_description(f"Epoch {epoch}")
            optimizer.zero_grad()
            
            # Video_clips: [16, 16, 64, 64, 9]) motion: [16, 2, 16]
            x,_,_ = x
            batch, motion = dict.values(x)  
            batch = batch.to(DEVICE)
            motion = motion.to(DEVICE)
            
            # Video_clips: [16, 16, 64, 64, 9]) motion: [16, 16, 2]  
            vid = extend_batch(batch, dataloader_train, args.batch_size)
            mot = extend_motion(motion, dataloader_train, args.batch_size)
            mot = mot.permute(0,2,1)
            
            # Map input images to latent space + normalize latents:
            vid = vid.view(-1,64,64,9).permute(0,3,1,2)
            enc_vid = vae.encode(vid).latent_dist.sample()
            enc_vid = enc_vid.reshape(args.batch_size, args.sequence_length,4,8,8).permute(0,1,3,4,2)
            # Mask the inputs
            mask_vid, mask_motion = mask_generator(enc_vid, mot)

            # Noised input data, Video noise eps_vid and Motion noise eps_mot
            t = torch.randint(0, args.iteration_num, (args.batch_size, 1)).squeeze()
            # print(t)
            diff_vid, eps_vid = fw_diffusion_vid(enc_vid, alpha_cum, t)
            diff_mot, eps_mot = fw_diffusion_mot(mot, alpha_cum, t)

            # Masked input Noise
            diff_vid, diff_mot = mask_vid_mot(mask_vid,mask_motion, enc_vid, mot, diff_vid, diff_mot)
            
            # The reconstructed outputs: pred_vid, pred_mot
            diff_vid = diff_vid.permute(0,4,1,2,3)

            t = t.to(DEVICE)
            pred_vid, pred_mot = eps_est_model(diff_vid, diff_mot, t)
            pred_vid = pred_vid.permute(0,2,3,4,1)
            pred_vid = mask_vid.to(torch.uint8) * pred_vid
            eps_vid = mask_vid.to(torch.uint8) * eps_vid
            pred_mot = mask_motion.to(torch.uint8) * pred_mot
            eps_mot = mask_motion.to(torch.uint8) * eps_mot

            loss_vid = loss_l2(pred_vid, eps_vid)
            loss_mot = loss_l2(pred_mot, eps_mot)

            loss = loss_vid + loss_mot
            mean_loss_vid += loss_vid.item()
            mean_loss_mot += loss_mot.item()
            # log metrics to wandb
            log_dict = {
                "Epoch":epoch,
                "mean_video_loss": mean_loss_vid/(loop.n+1), 
                "mean_motion_loss": mean_loss_mot/(loop.n+1),
            }
            wandb.log(log_dict)

            loss.backward()
            optimizer.step()
            
            loop.set_postfix(loss=loss.item(), noise_est_loss_vid=mean_loss_vid/(loop.n+1),noise_est_loss_mot=mean_loss_mot/(loop.n+1))

        if ((epoch+1) % 10 == 0):
            torch.save({'model_state_dict': eps_est_model.state_dict()}, checkpoint_path + '/VDT_20240807_d'+ str(depth) +'_h'+ str(hidden_size) +'_nh'+ str(nhead) +'_np'+ str(patch_size) +'_it1000_'  + str(epoch+1) + '.pth.tar')
