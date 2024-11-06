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
from diffusers.models import AutoencoderKL
from utils import mask_vid_mot


if __name__ == "__main__":
    # initialization 
    checkpoint_path = os.path.join('checkpoints_vae_seg')

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(DEVICE)
    # The parser setting
    parser = argparse.ArgumentParser()
    parser = VideoDataset.add_model_specific_args(parser)
    parser.add_argument('--data_path', type=str, default='dataset/2024-10-31_xacti_5fps')

    parser.add_argument('--sequence_length', type=int, default=40)
    parser.add_argument('--resolution', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--LR_RATE', type=int, default=1e-4)
    parser.add_argument('--NUM_EPOCHS', type=int, default=60)
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
    dataset = Dataset(args.data_path, args.sequence_length,rgb=False, train=False, resolution=args.resolution)
    dataloader_val = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False
    )

    input_dim = 9
    output_dim = 9
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-mse")
    # Modify the encoder and decoder in the VAE
    vae.encoder.conv_in = nn.Conv2d(input_dim, vae.encoder.conv_in.out_channels, kernel_size=3, stride=1, padding=1)
    vae.decoder.conv_out = nn.Conv2d(vae.decoder.conv_out.in_channels, output_dim, kernel_size=3, stride=1, padding=1)
    vae.to(DEVICE)

    optimizer = optim.Adam(vae.parameters(), lr=args.LR_RATE)
    loss_CE = nn.CrossEntropyLoss()
    #start Training
    for epoch in range(args.NUM_EPOCHS):
        mean_loss_vid = 0
        mean_loss_mot = 0
        vae.train()
        loop = tqdm(dataloader_train)

        for x in loop:
            loop.set_description(f"Epoch {epoch}")
            optimizer.zero_grad()
            
            # Video_clips: [16, 16, 64, 64, 9])
            batch, _ = dict.values(x)  
            batch = batch.to(DEVICE)
            vid = extend_batch(batch, dataloader_train, args.batch_size)

            # Map input images to latent space + normalize latents:
            vid = vid.view(-1,64,64,9).permute(0,3,1,2)
            z = vae.encode(vid).latent_dist.sample()
            recon_vid = vae.decode(z).sample
            gt_vid = torch.argmax(vid,dim=1)
            # print(gt_vid.shape, recon_vid.shape)
            loss_vid = loss_CE(recon_vid, gt_vid)

            mean_loss_vid += loss_vid.item()

            loss_vid.backward()
            optimizer.step()
            
            loop.set_postfix(loss_vid=mean_loss_vid/(loop.n+1))

        if ((epoch+1) % 20 == 0):
            torch.save({'model_state_dict': vae.state_dict()}, checkpoint_path + '/vae_20241031'+ str(epoch+1) + '.pth.tar')


