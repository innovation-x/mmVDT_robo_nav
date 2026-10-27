import torch
import matplotlib.pyplot as plt
import numpy as np
import random
import copy
import numpy as np
import cv2

squence_length = 16
half_seq_len = int(squence_length/2)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
class MixedSequenceMaskGenerator():
    """
    The fixed mask that masks actions and observations after timestep 8, which is consistent with the paper. 
    """
    def __init__(self, t1=half_seq_len, t2=squence_length, x1=0, y1=0, x2=64, y2=64):
        self.t1 = t1
        self.t2 = t2
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
    
    def temporal_mask_generator(self, batch, motion,idx=None,t=half_seq_len):
        mask_vid = torch.zeros_like(batch)
        mask_mot = torch.zeros_like(motion)
        # Predict
        if idx in [0,1,4]:
            mask_vid[:, t:self.t2] = 1
            mask_mot[:, t:self.t2] = 1

        # # Backward
        # if idx == 1:
        #     mask_vid[:, 0:self.t1] = 1
        #     mask_mot[:, 0:self.t1] = 1
            
        # Interpolation(motion known)
        elif idx == 2:
            mask_vid[:, t:self.t2] = 1
        # Interpolation(video known)
        elif idx == 3:
            mask_mot[:, t:self.t2] = 1
        # Only one frame
        # elif idx == 3:
        #     frame_idx = random.randint(0,self.t2 -1)
        #     mask_vid[:, 0:frame_idx] = 1
        #     mask_vid[:, frame_idx+1:] = 1

        #     mask_mot[:, 0:frame_idx] = 1
        #     mask_mot[:, frame_idx+1:] = 1

        return mask_vid, mask_mot

    def __call__(self, batch, motion, idx=None, t=half_seq_len):
        if idx is not None:
            mask_vid, mask_mot = self.temporal_mask_generator(batch, motion,idx, t)
        else:
            idx = random.randint(0,4)
            mask_vid, mask_mot = self.temporal_mask_generator(batch, motion,idx, t)
        return mask_vid, mask_mot, idx

def mask_vid_mot(mask_vid, mask_mot, vid, mot, diff_vid, diff_mot): 
    # Masked noised input data
    vid[mask_vid.byte()] = 0
    diff_vid[1-mask_vid.byte()] = 0
    diff_vid += vid

    mot[mask_mot.byte()] = 0
    diff_mot[1-mask_mot.byte()] = 0
    diff_mot += mot 
    return diff_vid, diff_mot

def extend_batch(batch, dataloader, batch_size):
    """
    If the batch size is less than batch_size, extends it with
    data from the dataloader until it reaches the required size.
    Here batch is a tensor.
    Returns the extended batch.
    """
    while batch.shape[0] != batch_size:
        dataloader_iterator = iter(dataloader)
        nw_batch,_,_ = next(dataloader_iterator)
        nw_batch,_ = dict.values(nw_batch)
        nw_batch = nw_batch.to(DEVICE)
        if nw_batch.shape[0] + batch.shape[0] > batch_size:
            nw_batch = nw_batch[:batch_size - batch.shape[0]]
        batch = torch.cat([batch, nw_batch], 0)
    return batch

def extend_motion(motion, dataloader, batch_size):
    """
    If the batch size is less than batch_size, extends it with
    data from the dataloader until it reaches the required size.
    Here batch is a tensor.
    Returns the extended batch.
    """
    while motion.shape[0] != batch_size:
        dataloader_iterator = iter(dataloader)
        nw_batch,_,_ = next(dataloader_iterator)
        _, nw_motion = dict.values(nw_batch)
        nw_motion = nw_motion.to(DEVICE)

        if nw_motion.shape[0] + motion.shape[0] > batch_size:
            nw_motion = nw_motion[:batch_size - motion.shape[0]]
        motion = torch.cat([motion, nw_motion], 0)
    return motion


def fw_diffusion_vid(x, alpha_cum, t):
    ext_alpha = torch.sqrt(alpha_cum[t]).view(-1, 1, 1, 1, 1).to(DEVICE)
    ext_n_alpha = torch.sqrt(1.0 - alpha_cum[t]).view(-1, 1, 1, 1, 1).to(DEVICE)
    eps = torch.randn(*x.shape).to(DEVICE)
    diff_x = ext_alpha * x + ext_n_alpha*eps
    return diff_x, eps


def fw_diffusion_mot(x, alpha_cum, t):
    ext_alpha = torch.sqrt(alpha_cum[t]).view(-1, 1, 1).to(DEVICE)
    ext_n_alpha = torch.sqrt(1.0 - alpha_cum[t]).view(-1, 1, 1).to(DEVICE)
    eps = torch.randn(*x.shape).to(DEVICE)
    diff_x = ext_alpha * x + ext_n_alpha*eps
    return diff_x, eps


def visual_img(batch, motion, mot_viz = True):
    '''
    This function can visualize the video clips and present the 
    motions on the frames. 
    The shape of video clips: (1,16,64,64,9)
    The shape of motion: (1,16,2)
    '''

    linear_gt = []
    angle_gt = []

    fig = plt.figure(figsize=(15, 10))
    batch_rgb = torch.argmax(batch, 4)
    squence_length = batch_rgb.shape[1]
    C, T, H, W = batch_rgb.shape
    batch_rgb = batch_rgb.squeeze().cpu().detach().numpy()
    motion = motion.squeeze().cpu().detach().numpy()
    # The ground-truth unobserved actions (gt robot actions)
    for t in range(squence_length):
        if (t >= half_seq_len):            
            x = float(motion[t,0].mean())
            z = float(motion[t,1].mean())
            linear_gt.append(x)
            angle_gt.append(z)
    #         print('li:%.3f'%x,'ang:%.3f'%z)
    # print('==========================================================')

    # Visualization of full information video inputs and gt robot actions
    for t in range(squence_length):
        im = torch.tensor(batch_rgb[t:t+1,:,:])
        im = im.view(H, W, C)
        ax = fig.add_subplot(6, int(squence_length/2), int(squence_length/2)+t+1, xticks=[], yticks=[])
        if t==int(squence_length/2):
            map_x = 60 - 10*torch.cumsum(torch.tensor(linear_gt[::int(squence_length/16)]), dim=0)*np.cos(np.pi/4)
            map_z = 32 - torch.cumsum(torch.cumsum(torch.tensor(angle_gt[::int(squence_length/16)]), dim=0),dim=0)*np.cos(np.pi/4)
        ax.imshow(im[:,:],'gray')  
    if mot_viz:
        fig1 = plt.figure(figsize=(20, 5))
        plt.plot(map_z.tolist(), map_x.tolist(), color='b', marker='o', linestyle='-',linewidth=2,markersize=5)
        im = torch.tensor(batch_rgb[half_seq_len-1,:,:]).view(H,W,C)
        plt.imshow(im, 'gray') 

def visual_img_each_channel(batch, mot_viz = True):
    '''
    This function can visualize the video clips and present the 
    motions on the frames. 
    The shape of video clips: (1,16,64,64,9)
    The shape of motion: (1,16,2)
    '''

    linear_gt = []
    angle_gt = []

    fig = plt.figure(figsize=(15, 10))
    batch_ch = batch[:,:,:,:].squeeze().cpu().detach().numpy()
    T, H, W = batch_ch.shape
    squence_length = T
    for t in range(squence_length):
        im = torch.tensor(batch_ch[t:t+1,:,:])
        im = im.view(H, W, 1)
        if squence_length == 16:
            ax = fig.add_subplot(6, int(squence_length/2), int(squence_length/2)+t+1, xticks=[], yticks=[])
        else:
            ax = fig.add_subplot(6, int(squence_length/4), int(squence_length/2)+t+1, xticks=[], yticks=[])

        ax.imshow(im,'gray')

def visual_vid_mmot(batch, motion, rgb=False, mot_viz = True, is_road=False, grid=True, frame_t = half_seq_len):
    '''
    This function can visualize the video clips and present the 
    motions on the frames. 
    The shape of video clips: (1,16,64,64,9)
    The shape of motion: (1,16,2)
    '''

    fig = plt.figure(figsize=(15, 10))
    if not rgb:
        batch_rgb = torch.argmax(batch, 4)
        C, T, H, W = batch_rgb.shape
    else:
        batch_rgb = batch
        T, H, W, C =batch_rgb.shape
    squence_length = T
   
    batch_rgb = batch_rgb.squeeze().cpu().detach().numpy()
    # Visualization of full information video inputs and gt robot actions
    for t in range(squence_length):
        im = torch.tensor(batch_rgb[t:t+1,:,:])
        im = im.view(H, W, C)
        if squence_length == 16:
            ax = fig.add_subplot(6, int(squence_length/2), int(squence_length/2)+t+1, xticks=[], yticks=[])
        else:
            ax = fig.add_subplot(6, int(squence_length/4), int(squence_length/2)+t+1, xticks=[], yticks=[])

        ax.imshow(im[:,:],'gray')  
        if grid:
            # 设置网格
            ax.set_xticks(range(0, W, 10))  # 设置 x 轴的网格间隔
            ax.set_yticks(range(0, H, 10))  # 设置 y 轴的网格间隔
            ax.grid(True, color='blue', linestyle='--', linewidth=0.3)  # 设置网格样式

            # 隐藏坐标轴
            ax.set_xticklabels([])  # 隐藏 x 轴标签
            ax.set_yticklabels([])  # 隐藏 y 轴标签

    # The ground-truth unobserved actions (gt robot actions)
    fig1 = plt.figure(figsize=(20, 5))
    i = 0
    for motion in motion:
        i += 1
        motion = motion.squeeze().cpu().detach().numpy()
        linear_gt = []
        angle_gt = []
        for t in range(squence_length):
            if (t >= frame_t):            
                x = float(motion[t,0].mean())
                z = float(motion[t,1].mean())
                linear_gt.append(x)
                angle_gt.append(z)
                if mot_viz:
                    print('li:%.3f'%x,'ang:%.3f'%z)
        print('==========================================================')

                        
        map_x = 60 - 6 * torch.cumsum(torch.tensor(linear_gt[::int(squence_length/16)]), dim=0)*np.cos(np.pi/4)
        map_z = 32 -0.6 * torch.cumsum(torch.cumsum(torch.tensor(angle_gt[::int(squence_length/16)]), dim=0),dim=0)*np.cos(np.pi/4)
        if mot_viz:
            if (i == 0):
                plt.plot(map_z.tolist(), map_x.tolist(), color='r', marker='o', linestyle='-',linewidth=1.5,markersize=3.5)
            else: 
                plt.plot(map_z.tolist(), map_x.tolist(), color='orange', marker='o', linestyle='-',linewidth=2,markersize=5)
            im = torch.tensor(batch_rgb[frame_t-1,:,:]).view(H,W,C)
            plt.imshow(im, 'gray') 
        if is_road:
            return(map_z[-1], map_x[-1])

def visual_vid_mmot_each_channel(batch, motion, channel = 1, mot_viz = True, grid=True, frame_t = half_seq_len, visual_on = True):
    '''
    This function can visualize the video clips and present the 
    motions on the frames. 
    The shape of video clips: (1,16,64,64,9)
    The shape of motion: (1,16,2)
    '''
    
    batch_ch = batch[:,:,:,:,channel].squeeze().cpu().detach().numpy()
    T, H, W = batch_ch.shape
    squence_length = T
    if visual_on:
        fig = plt.figure(figsize=(20, 10))
        for t in range(squence_length):
            im = torch.tensor(batch_ch[t:t+1,:,:])
            im = im.view(H, W, 1)
            if squence_length == 16:
                ax = fig.add_subplot(6, int(squence_length/2), int(squence_length/2)+t+1, xticks=[], yticks=[])
            else:
                ax = fig.add_subplot(6, int(squence_length/4), int(squence_length/2)+t+1, xticks=[], yticks=[])

            ax.imshow(im,'gray')
            if grid:
                # 设置网格
                ax.set_xticks(range(0, W, 10))  # 设置 x 轴的网格间隔
                ax.set_yticks(range(0, H, 10))  # 设置 y 轴的网格间隔
                ax.grid(True, color='blue', linestyle='--', linewidth=0.3)  # 设置网格样式

                # 隐藏坐标轴
                ax.set_xticklabels([])  # 隐藏 x 轴标签
                ax.set_yticklabels([])  # 隐藏 y 轴标签

    # The ground-truth unobserved actions (gt robot actions)
    i = 0
    if mot_viz:
        fig1 = plt.figure(figsize=(20, 5))
        for motion in motion:
            i += 1
            motion = motion.squeeze().cpu().detach().numpy()
            linear_gt = []
            angle_gt = []
            for t in range(squence_length):
                if (t >= frame_t):            
                    x = float(motion[t,0].mean())
                    z = float(motion[t,1].mean())
                    linear_gt.append(x)
                    angle_gt.append(z)
                    print('li:%.3f'%x,'ang:%.3f'%z)
            print('==========================================================')
    return torch.tensor(batch_ch[frame_t:])
                        
def gray2color(video):
    ''' The input video size: [16, 9, 64, 64] '''
    video = torch.argmax(video, 1).unsqueeze(1).cpu().numpy() *0.1*255
    video = np.round(video*10)/10
    # print(video[14,0,30])
    video_r = copy.deepcopy(video)
    video_g = copy.deepcopy(video)
    video_b = copy.deepcopy(video)

    ################ Wall: Light cream######################
    video_r[video_r== 76.5] = 238 
    video_g[video_g== 76.5] = 213 
    video_b[video_b== 76.5] = 183 

    ################ Person: Deep orange-red ######################
    video_r[video_r==153] = 212 
    video_g[video_g==153] = 76 
    video_b[video_b==153] = 60 

    ################ Floor: gray ######################
    video_r[video_r==178.5] = 128
    video_g[video_g==178.5] = 128 
    video_b[video_b==178.5] = 128 

    ################ Window: Blue-Green ######################
    video_r[video_r==102] = 68
    video_g[video_g==102] = 117 
    video_b[video_b==102] = 122 

    ################ ceiling: Off-White ######################
    video_r[video_r==25.5] = 220
    video_g[video_g==25.5] = 220 
    video_b[video_b==25.5] = 220 

    ################ Door: Purple ##########################
    video_r[video_r==127.5] = 69 
    video_g[video_g==127.5] = 42 
    video_b[video_b==127.5] = 61 

    video_rgb = torch.cat([torch.tensor(video_r),torch.tensor(video_g), torch.tensor(video_b)], dim=1)/255
    return video_rgb

def gray3c_2color(video):
    ''' The input video size: [16, 9, 64, 64] '''
    video = torch.argmax(video, 1).unsqueeze(1).cpu().numpy() *0.1*255
    video = np.round(video*10)/10
    # print(video[14,0,30])
    video_r = copy.deepcopy(video)
    video_g = copy.deepcopy(video)
    video_b = copy.deepcopy(video)

    ################ Person: Light cream######################
    video_r[video_r== 25.5] = 212
    video_g[video_g== 25.5] = 76
    video_b[video_b== 25.5] = 60

    ################ Floor: Deep orange-red ######################
    video_r[video_r==51] = 128
    video_g[video_g==51] = 128
    video_b[video_b==51] = 128

    ################ background: gray ######################
    video_r[video_r==0] = 0
    video_g[video_g==0] = 0
    video_b[video_b==0] = 0

    video_rgb = torch.cat([torch.tensor(video_r),torch.tensor(video_g), torch.tensor(video_b)], dim=1)/255
    return video_rgb



def channel2color(binary_video):
    ''' The input video size: [t, 64, 64, 9] '''
    video = binary_video.squeeze().permute(0,3,1,2).cpu().numpy() # [t, 9, 64, 64]
    # print(video[14,0,30])
    # 创建一个空的 RGB 图像
    video_r = np.zeros((video.shape[0], video.shape[2], video.shape[3]), dtype=np.uint8)
    video_g = np.zeros((video.shape[0], video.shape[2], video.shape[3]), dtype=np.uint8)
    video_b = np.zeros((video.shape[0], video.shape[2], video.shape[3]), dtype=np.uint8)


    # 定义每个 channel 的颜色
    colors = [
        (0, 0, 0),       # Undetected: Black
        (220, 220, 220),  # Ceiling: Off-White
        (0, 0, 0),  # Undetected: Black
        (238, 213, 183),  # Wall: Light cream
        (68, 117, 122),   # Window: Blue-Green
        (69, 42, 61),     # Door: Purple
        (212, 76, 60),    # Person: Deep orange-red
        (128, 128, 128),  # Floor: gray
        (0, 0, 0),       # Undetected: Black
    ]

    # 遍历每个 channel，将有值的部分替换为指定的颜色
    for i in range(video.shape[1]):
        mask = video[:, i, :, :] == 1
        video_r[mask] = colors[i][0]
        video_g[mask] = colors[i][1]
        video_b[mask] = colors[i][2]

    # 合成一个彩色的图
    video_rgb = np.stack([video_r, video_g, video_b], axis=-1)  # [t, 64, 64, 3]
    video_rgb = torch.tensor(video_rgb) / 255.0  # 归一化

    return video_rgb


if __name__ == '__main__':
    a = torch.randn([1, 40, 64, 64, 9])
    b = torch.randn([1,40,2]) 
    mask_generator = MixedSequenceMaskGenerator(t2=40)
    mask_vid, mask_motion, _ = mask_generator(a,b, idx=3, t=10)
    print(mask_vid)

