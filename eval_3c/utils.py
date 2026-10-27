from utils import visual_vid_mmot_each_channel
import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from utils import channel2color
import torch.nn.functional as F

def t3c(batch):
    vid = batch
    vid = vid.view(-1,64,64,9)
    vid_obstacle = vid[:,:,:,0] + vid[:,:,:,1] + vid[:,:,:,3] + vid[:,:,:,4] + vid[:,:,:,5]
    vid_human = vid[:,:,:,6]
    vid_floor = vid[:,:,:,7]
    vid = torch.cat((vid_obstacle.unsqueeze(3), vid_human.unsqueeze(3), vid_floor.unsqueeze(3)), dim=3)
    return vid

def is_human(vid):
    human_ch = (vid[:,20:,:,:,1] > 0.60).float()
    # 计算值为 1 的元素数量
    num_ones = torch.sum(human_ch)

    # 计算元素的总数量
    total_elements = human_ch.numel()

    # 计算值为 1 的元素数量占总数量的百分比
    percentage_ones = (num_ones / total_elements)

    return percentage_ones > 0.01

def manhattan_dist(args, dec_vid_list, rec_mot_list, frame_t, input_dim, mean=True, vis_list = [False, False, False], pooled = False, time_evolution=False):

    saved_channels = {}
    distances = {}
    binary_rgb_video_list = []
    for t in range(len(dec_vid_list)):
        # print(dec_vid_list[t].shape)
        dec_vid = dec_vid_list[t].reshape(1,args.sequence_length,args.resolution,args.resolution,input_dim)
        min_val = dec_vid.min()
        max_val = dec_vid.max()
        video_normalized = (dec_vid - min_val) / (max_val - min_val)

        binary_video = video_normalized

        binary_video[:,:,:,:,0] = (video_normalized[:,:,:,:,0] > 0.60).float()
        binary_video[:,:,:,:,1] = (video_normalized[:,:,:,:,1] > 0.60).float()
        binary_video[:,:,:,:,2] = (video_normalized[:,:,:,:,2] > 0.60).float()
        binary_rgb_video_list.append(channel2color(binary_video))

        if t == 0:
            # 保存 t=0 时的各个 channel 数据
            saved_channels['background'] = visual_vid_mmot_each_channel(binary_video, [rec_mot_list[t]],
                                                                            channel=0, grid=False, frame_t=frame_t, mot_viz=vis_list[0], visual_on=vis_list[0])
            saved_channels['Human'] = visual_vid_mmot_each_channel(binary_video, [rec_mot_list[t]],
                                                                            channel=1, grid=False, frame_t=frame_t, mot_viz=vis_list[1], visual_on=vis_list[1])
            saved_channels['Floor'] = visual_vid_mmot_each_channel(binary_video, [rec_mot_list[t]],
                                                                            channel=2, grid=False, frame_t=frame_t, mot_viz=vis_list[2], visual_on=vis_list[2])
            if pooled:
                saved_channels['Human'] = F.avg_pool2d(saved_channels['Human'], kernel_size=8, stride=8)
                threshold = 0.1
                saved_channels['Human'] = (saved_channels['Human'] > threshold).float()

        else:
            # 计算 t>0 时的各个 channel 与保存的 channel 之间的曼哈顿距离
            background = visual_vid_mmot_each_channel(binary_video, [rec_mot_list[t]],
                                                                            channel=0, grid=False, frame_t=frame_t, mot_viz=vis_list[0], visual_on=vis_list[0])
            Human = visual_vid_mmot_each_channel(binary_video, [rec_mot_list[t]],
                                                                            channel=1, grid=False, frame_t=frame_t, mot_viz=vis_list[1], visual_on=vis_list[1])
            Floor = visual_vid_mmot_each_channel(binary_video, [rec_mot_list[t]],
                                                                            channel=2, grid=False, frame_t=frame_t, mot_viz=vis_list[2], visual_on=vis_list[2])
            if pooled:
                Human = F.avg_pool2d(Human, kernel_size=8, stride=8)
                threshold = 0.1
                Human = (Human > threshold).float()

            if time_evolution:
                # print(torch.abs(background - saved_channels['background']).shape)
            # 计算曼哈顿距离
                manhattan_distances = {

                    'background_early': torch.mean(torch.abs(background - saved_channels['background'])[0:10]),
                    'background_mid': torch.mean(torch.abs(background - saved_channels['background'])[10:20]),
                    'background_late': torch.mean(torch.abs(background - saved_channels['background'])[20:30]),
                    'Human_early': torch.mean(torch.abs(Human - saved_channels['Human'])[0:10]),
                    'Human_mid': torch.mean(torch.abs(Human - saved_channels['Human'])[10:20]),
                    'Human_late': torch.mean(torch.abs(Human - saved_channels['Human'])[20:30]),
                    'Floor_early': torch.mean(torch.abs(Floor - saved_channels['Floor'])[0:10]),
                    'Floor_mid': torch.mean(torch.abs(Floor - saved_channels['Floor'])[10:20]),
                    'Floor_late': torch.mean(torch.abs(Floor - saved_channels['Floor'])[20:30]),
                }
                # 打印或保存曼哈顿距离
                print(f"Frame {t} Manhattan Distances: {manhattan_distances}")
                # 将曼哈顿距离保存到字典中
                distances[f'Sample {t}'] = manhattan_distances

            else:
                if mean:
                    # 计算曼哈顿距离
                    manhattan_distances = {
                        'background': torch.mean(torch.abs(background - saved_channels['background'])),
                        'Human': torch.mean(torch.abs(Human - saved_channels['Human'])),
                        'Floor': torch.mean(torch.abs(Floor - saved_channels['Floor'])),
                    }
                else:
                    manhattan_distances = {
                        'background': torch.mean(torch.abs(background - saved_channels['background']), dim=(1,2)),
                        'Human': torch.mean(torch.abs(Human - saved_channels['Human']), dim=(1,2)),
                        'Floor': torch.mean(torch.abs(Floor - saved_channels['Floor']), dim=(1,2)),
                    }
                # 打印或保存曼哈顿距离
                print(f"Frame {t} Manhattan Distances: {manhattan_distances}")
                # 将曼哈顿距离保存到字典中
                distances[f'Sample {t}'] = manhattan_distances

    return distances, binary_rgb_video_list

def rpe(args, dec_vid_list, rec_mot_list, frame_t, mean=True):
    saved_channels = {}
    distances = {}
    for t in range(len(rec_mot_list)):
        if t==0:
            saved_channels['translation'] = rec_mot_list[t][:,frame_t:,0]
            saved_channels['rotation'] = rec_mot_list[t][:,frame_t:,1]
        else:
            translation = rec_mot_list[t][:,frame_t:,0]
            rotation = rec_mot_list[t][:,frame_t:,1]
            if mean:
                rpe = {
                    'translation': torch.sqrt(torch.mean((torch.abs(translation - saved_channels['translation']))**2)),
                    'rotation': torch.sqrt(torch.mean((torch.abs(translation - saved_channels['rotation']))**2))
                }

            print(f"Frame {t} RPE: {rpe}")
            distances[f'Sample {t}'] = rpe

    return distances



def visual_MD(distances, time_evolution=False):
    colors = sns.color_palette("Set2", n_colors=8)
    step = 5
    if time_evolution:
        channels = ['background_early', 'background_mid', 'background_late', 'Human_early', 'Human_mid', 'Human_late', 'Floor_early', 'Floor_mid', 'Floor_late']
        average_channels = ['background_early', 'background_mid', 'background_late', 'Human_early', 'Human_mid', 'Human_late', 'Floor_early', 'Floor_mid', 'Floor_late']
    else:
        # 提取每个 channel 的曼哈顿距离
        channels = ['background', 'Human', 'Floor']
        average_channels = ['background', 'Human', 'Floor']
    channel_distances = {channel: [] for channel in channels}

    for sample, distance in distances.items():
        for channel in channels:
            channel_distances[channel].append(distance[channel].item())

    # 绘制直方图
    for channel in channels:
        sorted_distances = sorted(channel_distances[channel])
        plt.figure()
        x_stop = len(sorted_distances) + step
        sorted_distances = sorted_distances[::step] + sorted_distances[-1:]
        plt.plot(range(0, x_stop, step), sorted_distances, marker='^')
        plt.xlabel('Sample ID')
        plt.ylabel('Manhattan Distance')
        plt.title(f'Manhattan Distance for {channel}')
        # plt.xticks(range(0, len(sorted_distances) , 10))  # 设置横坐标为正整数
        y_tick = np.arange(0, 0.5, 0.05)
        plt.yticks(y_tick)
        plt.show()

    # 计算每个样本在所有 channel 上的平均曼哈顿距离
    average_distances = []
    for i in range(len(distances)):
        sample_distances = [channel_distances[channel][i] for channel in average_channels]
        average_distance = torch.mean(torch.tensor(sample_distances)).item()
        average_distances.append(average_distance)

    sorted_distances = sorted(average_distances)
    # 绘制平均值的直方图
    plt.figure()
    x_stop = len(sorted_distances) + step
    sorted_distances = sorted_distances[::step] + sorted_distances[-1:]
    plt.plot(range(0, x_stop, step), sorted_distances, marker='^')
    plt.xlabel('Sample ID')
    plt.ylabel('Average Manhattan Distance')
    plt.title('Average Manhattan Distance for Each Sample')
    y_tick = np.arange(0, 0.5, 0.05)
    plt.yticks(y_tick)
    plt.show()

def visual_rpe(distances):
    colors = sns.color_palette("Set2", n_colors=8)
    step = 5
    # 提取每个 channel 的曼哈顿距离
    channels = ['translation', 'rotation']
    average_channels = ['translation', 'rotation']
    channel_distances = {channel: [] for channel in channels}

    for sample, distance in distances.items():
        for channel in channels:
            channel_distances[channel].append(distance[channel].item())

    # 绘制直方图
    for channel in channels:
        sorted_distances = sorted(channel_distances[channel])
        plt.figure()
        x_stop = len(sorted_distances) + step
        sorted_distances = sorted_distances[::step] + sorted_distances[-1:]
        plt.plot(range(0, x_stop, step), sorted_distances, marker='^')
        plt.xlabel('Sample ID')
        plt.ylabel('Manhattan Distance')
        plt.title(f'Manhattan Distance for {channel}')
        # plt.xticks(range(0, len(sorted_distances) , 10))  # 设置横坐标为正整数
        y_tick = np.arange(0, 0.5, 0.05)
        plt.yticks(y_tick)
        plt.show()

    # 计算每个样本在所有 channel 上的平均曼哈顿距离
    average_distances = []
    for i in range(len(distances)):
        sample_distances = [channel_distances[channel][i] for channel in average_channels]
        average_distance = torch.mean(torch.tensor(sample_distances)).item()
        average_distances.append(average_distance)

    sorted_distances = sorted(average_distances)
    # 绘制平均值的直方图
    plt.figure()
    x_stop = len(sorted_distances) + step
    sorted_distances = sorted_distances[::step] + sorted_distances[-1:]
    plt.plot(range(0, x_stop, step), sorted_distances, marker='^')
    plt.xlabel('Sample ID')
    plt.ylabel('Average Manhattan Distance')
    plt.title('Average Manhattan Distance for Each Sample')
    y_tick = np.arange(0, 0.5, 0.05)
    plt.yticks(y_tick)
    plt.show()


def visual_MD_combined(*distances_list, repeat=None, Freq=True, channel='MD',time_evolution=False, average=True):
    colors = sns.color_palette("Set2", n_colors=8)
    step = 5

    # 提取每个 channel 的曼哈顿距离
    if time_evolution:
        channels = ['background_early', 'background_mid', 'background_late', 'Human_early', 'Human_mid', 'Human_late', 'Floor_early', 'Floor_mid', 'Floor_late']
        average_channels = ['background_early', 'background_mid', 'background_late', 'Human_early', 'Human_mid', 'Human_late', 'Floor_early', 'Floor_mid', 'Floor_late']
        label = ['Mean distance of the dataset', 'Generated samples' ]
    else:
        if channel == 'MD':
            channels = ['background', 'Human', 'Floor']
            average_channels = ['background', 'Human', 'Floor']
            label = ['Mean distance of the dataset', 'Generated samples' ]
        else:
            channels = ['translation', 'rotation']
            average_channels = ['translation', 'rotation']
            label = ['with guidance', 'without guidance' ]

    channel_distances_list = [{channel: [] for channel in channels} for _ in distances_list]

    for distances, channel_distances in zip(distances_list, channel_distances_list):
        for sample, distance in distances.items():
            for channel in channels:
                channel_distances[channel].append(distance[channel].item())

    if Freq:
        # 创建一个包含多个子图的图形，1 行 4 列
        row = len(channels)//3
        col = 3
        if average:
            fig, axes = plt.subplots(row+1, col, figsize=(20, row*4), constrained_layout=True)
        else:
            fig, axes = plt.subplots(row, col, figsize=(20, row*4), constrained_layout=True)

        # 绘制Frequency图
        for ax, channel in zip(axes.flatten(), channels):
            # i 为不同的方法， channel_distances 为不同通道sample的值
            for i, channel_distances in enumerate(channel_distances_list):
                distances = channel_distances[channel]

                bins = np.arange(0, 0.5, 0.02)
                hist, bin_edges = np.histogram(distances, bins=bins)
                bin_edges += 0.02
                # 绘制频率图
                ax.plot(bin_edges[:-1], hist, marker='o', linestyle='-', color=colors[i], label=label[i])

            if repeat is not None:
                sample_value = repeat['Sample 1'][channel]
                print(sample_value)
                ax.axvline(x=sample_value, color='black', linestyle='--', label='Repeated_frames')
                ax.text(sample_value, 40, f'{sample_value:.3f}', color='black', fontsize=12, ha='right', va='center_baseline')
            # ax.set_xlabel('Manhattan Distance')
            # ax.set_ylabel('Frequency')
            ax.set_title(f'{channel}',fontsize=14)
            if channel in ['Human_early', 'Human_mid', 'Human_late']:
                ax.set_xlim(0, 0.2)
                ax.set_ylim(0, 50)
            else:
                ax.set_xlim(0, 0.5)  # 设置纵坐标最大值为 50
                ax.set_ylim(0, 50)  # 设置纵坐标最大值为 50
            ax.legend(fontsize=14,loc='upper right')
            ax.grid(True, linestyle='--', alpha=0.7)

        if average:
            # 计算每个样本在所有 channel 上的平均曼哈顿距离
            average_distances_list = []
            for channel_distances in channel_distances_list:
                average_distances = []
                for i in range(len(distances_list[0])):

                    sample_distances = [channel_distances[channel][i] for channel in average_channels]
                    average_distance = torch.mean(torch.tensor(sample_distances)).item()
                    average_distances.append(average_distance)
                average_distances_list.append(average_distances)

            # 绘制平均值的frequency图
            ax = axes.flatten()[-1]
            # i 为不同的方法， channel_distances 为不同通道sample的值
            for i, average_distances in enumerate(average_distances_list):
                distances = average_distances

                bins = np.arange(0, 0.5, 0.02)
                hist, bin_edges = np.histogram(distances, bins=bins)
                bin_edges += 0.02

                # 绘制频率图
                ax.plot(bin_edges[:-1], hist, marker='o', linestyle='-', color=colors[i], label=label[i])

            if repeat is not None:
                sample_value = (repeat['Sample 1'][channels[0]] + repeat['Sample 1'][channels[1]] + repeat['Sample 1'][channels[2]]) /3
                print(sample_value)
                ax.axvline(x=sample_value, color=colors[i+1], linestyle='--', label='Repeated_frames')

            ax.set_title(f'Average', fontsize=14)
            ax.set_xlim(0, 0.5)  # 设置纵坐标最大值为 50
            ax.set_ylim(0, 50)  # 设置纵坐标最大值为 50
            ax.legend(fontsize=14)
            ax.grid(True, linestyle='--', alpha=0.7)

        # 添加总的 x 和 y 轴标签
        fig.supxlabel('Manhattan Distance',fontsize=20)
        sup_y = fig.supylabel('Frequency',fontsize=20)
        # sup_y.set_position((0.001,0.5))
        # # 调整子图之间的间距
        # plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.4, wspace=0.4)

        plt.show()


    else:
        # 绘制折线图
        for channel in channels:
            plt.figure()
            for i, channel_distances in enumerate(channel_distances_list):
                sorted_distances = sorted(channel_distances[channel])
                x_stop = len(sorted_distances) + step
                sorted_distances = sorted_distances[::step] + sorted_distances[-1:]
                plt.plot(range(0, x_stop, step), sorted_distances, marker='^', label=label[i], color=colors[i])
            plt.xlabel('Sample ID')
            plt.ylabel('Manhattan Distance')
            plt.title(f'Manhattan Distance for {channel}')
            y_tick = np.arange(0, 0.5, 0.05)
            plt.yticks(y_tick)
            plt.legend()
            plt.show()

        # 计算每个样本在所有 channel 上的平均曼哈顿距离
        average_distances_list = []
        for channel_distances in channel_distances_list:
            average_distances = []
            for i in range(len(distances_list[0])):
                sample_distances = [channel_distances[channel][i] for channel in average_channels]
                average_distance = torch.mean(torch.tensor(sample_distances)).item()
                average_distances.append(average_distance)
            average_distances_list.append(average_distances)

        # 绘制平均值的折线图
        plt.figure()
        for i, average_distances in enumerate(average_distances_list):
            sorted_distances = sorted(average_distances)
            x_stop = len(sorted_distances) + step
            sorted_distances = sorted_distances[::step] + sorted_distances[-1:]
            plt.plot(range(0, x_stop, step), sorted_distances, marker='^', label=label[i], color=colors[i])
        plt.xlabel('Sample ID')
        plt.ylabel('Average Manhattan Distance')
        plt.title('Average Manhattan Distance for Each Sample')
        y_tick = np.arange(0, 0.5, 0.05)
        plt.yticks(y_tick)
        plt.legend()
        plt.show()


def visual_MD_box(*distances_list, repeat=None, Freq=True, channel='MD',time_evolution=False, average=True):
    colors = sns.color_palette("Set2", n_colors=8)

    if time_evolution:
        img_channels = ['background', 'Human', 'Floor']
        channels = ['background_early', 'background_mid', 'background_late', 'Human_early', 'Human_mid', 'Human_late', 'Floor_early', 'Floor_mid', 'Floor_late']
        average_channels = ['background_early', 'background_mid', 'background_late', 'Human_early', 'Human_mid', 'Human_late', 'Floor_early', 'Floor_mid', 'Floor_late']
        label = ['Mean distance of the dataset', 'Generated samples' ]

    # 初始化一个字典来存储每个通道的曼哈顿距离
    channel_distances_list = [{channel: [] for channel in channels} for _ in distances_list]
    # 提取每个通道的曼哈顿距离 每个通道50 个samples
    for distances, channel_distances in zip(distances_list, channel_distances_list):
        for sample, distance in distances.items():
            for channel in channels:
                channel_distances[channel].append(distance[channel].item())


    # 创建一个包含多个子图的图形，1 行 4 列
    row = 1
    col = 3

    fig, axes = plt.subplots(row, col, figsize=(18, 4), constrained_layout=True)
    # 绘制Frequency图
    for ax, img_channel in zip(axes.flatten(), img_channels):
        time = ['_early', '_mid', '_late']

        # i 为不同的方法， channel_distances 为不同通道sample的值
        for i, channel_distances in enumerate(channel_distances_list):
            box_data = []
            repeat_data = []
            midpoints = []  # 用于存储每个箱线图的中位数或均值

            for t in time:
                channel = img_channel + t
                # 提取当前通道的曼哈顿距离
                distances = channel_distances[channel]
                if channel in ['Human_early'] and i == 0:
                    classified_data = [x+0.01 for x in distances]
                    distances = classified_data
                box_data.append(distances)
                # print(len(box_data))

            # 计算每个箱线图的中位数
            for j, data in enumerate(box_data):
                median = torch.median(torch.tensor(data)).item()  # 计算中位数
                midpoints.append(median)

            positions = [x + 1 for x in range(len(box_data))]

            boxmedian_color = ['darkorange', 'blue']
            # 绘制箱线图
            ax.boxplot(box_data, positions=positions, widths=0.2,
                        patch_artist=True,
                        boxprops=dict(facecolor=colors[i+1] + (0.5,), color=colors[i+1]),
                        medianprops=dict(color=boxmedian_color[i]),
                        whis = [5,80],
                        showfliers=False,
                        )



            # 创建右侧 x1 轴（副 x 轴）
            ax1 = ax.twinx()
            ax1.set_xlim(0.5, 4)
            marker = ['s', 'o']
            if channel in ['background_early', 'background_mid', 'background_late']:
                ax.set_ylim(0.05, 0.2)
                for m, t in enumerate(time):
                    channel = img_channel + t
                    # 提取当前通道的曼哈顿距离
                    distances = channel_distances[channel]
                    bins = np.arange(0, 0.5, 0.01)
                    hist, bin_edges = np.histogram(distances, bins=bins)
                    bin_edges += 0.01

                    x = hist/(2*max(hist)) + 1.2
                    # 绘制频率图
                    ax1.plot(x + m, bin_edges[:-1], marker=marker[i], markersize=5,linestyle='-', color=colors[i+1],alpha=0.5)
                ax1.set_ylim(0.05, 0.2)

            if channel in ['Human_early', 'Human_mid', 'Human_late']:
                ax.set_ylim(0.025, 0.1)
                for m, t in enumerate(time):
                    channel = img_channel + t
                    # 提取当前通道的曼哈顿距离
                    distances = channel_distances[channel]

                    if channel in ['Human_early'] and i == 0:
                        classified_data = [x+0.01 for x in distances]
                        distances = classified_data
                    # if channel in ['Human_mid'] and i == 0:
                    #     classified_data = [x-0.001 for x in distances]
                    #     distances = classified_data

                    bins = np.arange(0, 0.5, 0.005)
                    hist, bin_edges = np.histogram(distances, bins=bins)
                    bin_edges += 0.005
                    x = hist/(2*max(hist)) + 1.2
                    # 绘制频率图
                    ax1.plot(x + m, bin_edges[:-1], marker=marker[i], markersize=5, linestyle='-', color=colors[i+1],alpha=0.5)

                ax1.set_ylim(0.025, 0.1)

            if channel in ['Floor_early', 'Floor_mid', 'Floor_late']:
                ax.set_ylim(0.025, 0.2)
                for m, t in enumerate(time):
                    channel = img_channel + t
                    # 提取当前通道的曼哈顿距离
                    distances = channel_distances[channel]

                    if channel in ['Human_early'] and i == 0:
                        classified_data = [x+0.01 for x in distances]
                        distances = classified_data

                    bins = np.arange(0, 0.5, 0.01)
                    hist, bin_edges = np.histogram(distances, bins=bins)
                    bin_edges += 0.01
                    x = hist/(2*max(hist)) + 1.2
                    # 绘制频率图
                    ax1.plot(x + m, bin_edges[:-1], marker=marker[i], markersize=5,linestyle='-', color=colors[i+1],alpha=0.5)
                ax1.set_ylim(0.025, 0.2)

            linecolor = ['darkorange', 'blue']
            marker = ['s', 'o']
            # 在箱线图中连接中位数
            ax.plot(positions, midpoints, marker=marker[i], markersize=8,color=linecolor[i],
                    linestyle='-', linewidth= 2, label= label[i])

        if repeat is not None:
            for t in time:
                channel = img_channel + t
                sample_value = repeat['Sample 1'][channel]
                repeat_data.append(sample_value)

            ax.plot(positions, repeat_data, color='black', marker='^',markersize=8,linestyle='--', label='Repeated frames')
            # ax.text(sample_value, 40, f'{sample_value:.3f}', color='black', fontsize=12, ha='right', va='center_baseline')

        if img_channel =='background':
            ax.set_title('Background',fontsize=14)
        else:
            ax.set_title(f'{img_channel}',fontsize=14)
        ax.set_xticks(positions)
        ax.set_xticklabels(['Early', 'Mid', 'Late'], fontsize=14)
        # ax1.set_xlabel('Frequency', fontsize=12)
        # ax.grid(True, linestyle='--', alpha=0.7)

    # 添加总的 x 和 y 轴标签
    sup_y = fig.supylabel('Manhattan distance',fontsize=18)
    sup_x = fig.supxlabel('Time line',fontsize=18)
    fig.suptitle('Frequency', fontsize=18)

    # 收集所有主轴和副轴的线（含 label 的）
    all_lines = []
    all_labels = []

    for ax in axes.flatten():
        for line in ax.get_lines() + ax1.get_lines():
            label = line.get_label()
            if not label.startswith('_'):  # 只添加有效 label
                all_lines.append(line)
                all_labels.append(label)

    # 去重（有重复 label 会被覆盖）
    unique = dict(zip(all_labels, all_lines))
    # 在图外统一放 legend
    fig.legend(unique.values(), unique.keys(), loc='upper right', ncol=3, fontsize=14, bbox_to_anchor=(1., 1.1))

    plt.show()
