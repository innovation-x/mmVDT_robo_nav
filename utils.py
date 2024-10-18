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
    
    def temporal_mask_generator(self, batch, motion,idx=0):
        mask_vid = torch.zeros_like(batch)
        mask_mot = torch.zeros_like(motion)
        # Predict
        if idx == 0:
            mask_vid[:, self.t1:self.t2] = 1
            mask_mot[:, self.t1:self.t2] = 1
        # Backward
        elif idx == 1:
            mask_vid[:, 0:self.t1] = 1
            mask_mot[:, 0:self.t1] = 1
        # Interpreation
        elif idx == 2:
            mask_vid[:, self.t1:self.t2 -2] = 1
            mask_mot[:, self.t1:self.t2 -2] = 1

        return mask_vid, mask_mot

    def __call__(self, batch, motion, idx=None):
        if idx is not None:
            mask_vid, mask_mot = self.temporal_mask_generator(batch, motion,idx)
        else:
            idx = random.randint(0,2)
            mask_vid, mask_mot = self.temporal_mask_generator(batch, motion,idx)
        return mask_vid, mask_mot

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
        nw_batch,_ = dict.values(next(dataloader_iterator))
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
        _, nw_motion = dict.values(next(dataloader_iterator))
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


def visual_img_each_channel(batch, channel = 1, mot_viz = True):
    '''
    This function can visualize the video clips and present the 
    motions on the frames. 
    The shape of video clips: (1,16,64,64,9)
    The shape of motion: (1,16,2)
    '''

    linear_gt = []
    angle_gt = []

    fig = plt.figure(figsize=(20, 10))
    batch_ch = batch[:,:,:,:,channel].squeeze().cpu().detach().numpy()
    T, H, W = batch_ch.shape

    for t in range(squence_length):
        im = torch.tensor(batch_ch[t:t+1,:,:])
        im = im.view(H, W, 1)
        ax = fig.add_subplot(6, int(squence_length/2), int(squence_length/2)+t+1, xticks=[], yticks=[])

        ax.imshow(im,'gray')

def visual_vid_mmot(batch, motion, rgb=False, mot_viz = True, is_road=False):
    '''
    This function can visualize the video clips and present the 
    motions on the frames. 

    one-hot encoded image:
        The shape of video clips: (16,64,64,9)
    RGB image:
        The shape of video clips: (16,64,64,3)
    
    motion: a list for multiple motion outputs whose shape is (16,2)
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
        ax = fig.add_subplot(6, int(squence_length/2), int(squence_length/2)+t+1, xticks=[], yticks=[])
        ax.imshow(im[:,:],'gray')  

    # The ground-truth unobserved actions (gt robot actions)
    fig1 = plt.figure(figsize=(20, 5))
    i = 0
    for motion in motion:
        i += 1
        motion = motion.squeeze().cpu().detach().numpy()
        linear_gt = []
        angle_gt = []
        for t in range(squence_length):
            if (t >= half_seq_len):            
                x = float(motion[t,0].mean())
                z = float(motion[t,1].mean())
                linear_gt.append(x)
                angle_gt.append(z)
                print('li:%.3f'%x,'ang:%.3f'%z)
        print('==========================================================')

                        
        map_x = 60 - 6 * torch.cumsum(torch.tensor(linear_gt[::int(squence_length/16)]), dim=0)*np.cos(np.pi/4)
        map_z = 32 -0.6 * torch.cumsum(torch.cumsum(torch.tensor(angle_gt[::int(squence_length/16)]), dim=0),dim=0)*np.cos(np.pi/4)
        if mot_viz:
            if (i == 0):
                plt.plot(map_z.tolist(), map_x.tolist(), color='r', marker='o', linestyle='-',linewidth=1.5,markersize=3.5)
            else: 
                plt.plot(map_z.tolist(), map_x.tolist(), color='g', marker='o', linestyle='-',linewidth=1,markersize=3)
            im = torch.tensor(batch_rgb[half_seq_len-1,:,:]).view(H,W,C)
            plt.imshow(im, 'gray') 
        if is_road:
            return(map_z[-1], map_x[-1])



def gray2color(video):
    
    '''The input video size: [16, 9, 64, 64]'''
    video = torch.argmax(video, 1).unsqueeze(1).cpu().numpy() *0.1*255
    video = np.round(video*10)/10
    # print(video[14,0,30])
    video_r = copy.deepcopy(video)
    video_g = copy.deepcopy(video)
    video_b = copy.deepcopy(video)

    ################ Wall ######################
    video_r[video_r== 76.5] = 238 
    video_g[video_g== 76.5] = 213 
    video_b[video_b== 76.5] = 183 

    ################ Person ######################
    video_r[video_r==153] = 212 
    video_g[video_g==153] = 76 
    video_b[video_b==153] = 60 

    ################ Floor ######################
    video_r[video_r==178.5] = 128
    video_g[video_g==178.5] = 128 
    video_b[video_b==178.5] = 128 

    ################ Window ######################
    video_r[video_r==102] = 68
    video_g[video_g==102] = 117 
    video_b[video_b==102] = 122 

    ################ ceiling ######################
    video_r[video_r==25.5] = 220
    video_g[video_g==25.5] = 220 
    video_b[video_b==25.5] = 220 

    ################ Door ##########################
    video_r[video_r==127.5] = 69 
    video_g[video_g==127.5] = 42 
    video_b[video_b==127.5] = 61 

    video_rgb = torch.cat([torch.tensor(video_r),torch.tensor(video_g), torch.tensor(video_b)], dim=1)/255
    return video_rgb

def save_video(frames, motion, output_path, fps=10):
    # 获取帧的宽度和高度（假设所有帧大小一致）
    height, width, _ = frames[0].shape
    
    # 定义视频编码器, 选择一个合适的编码器，例如 MJPG 或 MP4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # 创建 VideoWriter 对象
    out = cv2.VideoWriter(output_path, fourcc, fps, (256, 256))
    start_x, start_y = width // 2, height - 5  # 从视频中心开始

    # 将每一帧写入视频
    for i in range(8):
        frame = frames[i]
        speed = motion[i, 0]  # 前后的移动速度
        angle = motion[i, 1]  # 转角（弧度）
        # 计算箭头的结束位置
        if abs(speed) < 0.01:
            end_x = int(start_x)
            end_y = int(start_y)  

        else:
            # print(np.sin(angle))
            end_x = int(start_x - 30 * np.sin(angle))
            end_y = int(start_y - 25 * np.cos(angle))  

        # 绘制箭头
        arrow_color = (0, 1, 0)  # 绿色箭头
        cv2.arrowedLine(frame, (start_x, start_y), (end_x, end_y), arrow_color, thickness=2, tipLength=0.5)

        # 更新起始位置为当前结束位置
        # start_x, start_y = end_x, end_y

        if frame.dtype == np.float64:
            frame = (frame * 255).astype(np.uint8)
        # 如果帧是 RGB 格式，需要转换为 BGR 格式
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_bgr = cv2.resize(frame_bgr,(256,256))
        # 写入视频文件
        out.write(frame_bgr)
    
    # 释放 VideoWriter 对象
    out.release()
    print(f"视频已保存为 {output_path}")
if __name__ == '__main__':
    a = torch.randn([1, 16, 64, 64, 9])
    b = torch.randn([1,16,2]) 

