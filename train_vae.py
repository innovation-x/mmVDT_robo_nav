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

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import warnings

# 忽略所有 UserWarning
warnings.filterwarnings("ignore", category=UserWarning)

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ['MASTER_ADDR'] = 'localhost'  # 主节点地址，单机训练可以用localhost
os.environ['MASTER_PORT'] = '12345'      # 通讯端口，可以是未被占用的任意端口
world_size = 4

def train(rank):
    # initialization 
    checkpoint_path = os.path.join('checkpoints_vae_seg')

    ## initialize the DDP #####
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    print('GPU:', rank, 'is activated.')
    # The parser setting
    parser = argparse.ArgumentParser()
    parser = VideoDataset.add_model_specific_args(parser)
    parser.add_argument('--data_path', type=str, default='dataset/2024-10-31_xacti_5fps')

    parser.add_argument('--sequence_length', type=int, default=16)
    parser.add_argument('--resolution', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--LR_RATE', type=int, default=1e-4)
    parser.add_argument('--NUM_EPOCHS', type=int, default=60)
    args = parser.parse_args(args=[])

    # Load the Dataset
    Dataset = VideoDataset
    dataset = Dataset(args.data_path, args.sequence_length,rgb=False, train=True, resolution=args.resolution)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)

    # dataloader for the trainning
    dataloader_train = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False
    )

    # dataloader for the evaluation
    dataset = Dataset(args.data_path, args.sequence_length, train=False, resolution=args.resolution)
    dataloader_val = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
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
    vae.to(rank)
    if torch.cuda.device_count() > 1:
        vae = DDP(vae, device_ids=[rank])
    vae.to(rank)

    optimizer = optim.Adam(vae.parameters(), lr=args.LR_RATE)
    loss_CE = nn.CrossEntropyLoss()
    #start Training
    for epoch in range(args.NUM_EPOCHS):
        mean_loss_vid = 0
        vae.train()
        loop = tqdm(dataloader_train)

        for x in loop:
            loop.set_description(f"Epoch {epoch}")
            optimizer.zero_grad()
            
            # Video_clips: [16, 16, 64, 64, 9])
            x,_,_ = x
            batch, _ = dict.values(x)  
            vid = batch.to(rank)
            # vid = extend_batch(vid, dataloader_train, args.batch_size)

            # Map input images to latent space + normalize latents:
            vid = vid.view(-1,64,64,9).permute(0,3,1,2)
            z = vae.module.encode(vid).latent_dist.sample()
            recon_vid = vae.module.decode(z).sample
            gt_vid = torch.argmax(vid,dim=1)
            # print(gt_vid.shape, recon_vid.shape)
            loss_vid = loss_CE(recon_vid, gt_vid)

            mean_loss_vid += loss_vid.item()

            loss_vid.backward()
            optimizer.step()
            
            loop.set_postfix(loss_vid=mean_loss_vid/(loop.n+1))

        if ((epoch+1) % 20 == 0):
            torch.save({'model_state_dict': vae.state_dict()}, checkpoint_path + '/vae_20241031'+ str(epoch+1) + '.pth.tar')


if __name__ == "__main__":
    mp.spawn(train, nprocs=world_size, join=True)