from utils import visual_vid_mmot_each_channel
import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from utils import channel2color

def manhattan_dist(args, dec_vid_list, rec_mot_list, frame_t, input_dim, mean=True, vis_list = [False, False, False, False, False, False, False]):

    saved_channels = {}
    distances = {}
    binary_rgb_video_list = []
    for t in range(len(dec_vid_list)):
        dec_vid = dec_vid_list[t].reshape(1,args.sequence_length,args.resolution,args.resolution,input_dim)
        min_val = dec_vid.min()
        max_val = dec_vid.max()
        video_normalized = (dec_vid - min_val) / (max_val - min_val)

        binary_video = video_normalized

        binary_video[:,:,:,:,0] = (video_normalized[:,:,:,:,0] > 0.65).float()
        binary_video[:,:,:,:,1] = (video_normalized[:,:,:,:,1] > 0.58).float()
        binary_video[:,:,:,:,2] = (video_normalized[:,:,:,:,2] > 0.90).float()
        binary_video[:,:,:,:,3] = (video_normalized[:,:,:,:,3] > 0.56).float()
        binary_video[:,:,:,:,4] = (video_normalized[:,:,:,:,4] > 0.60).float()
        binary_video[:,:,:,:,5] = (video_normalized[:,:,:,:,5] > 0.70).float()
        binary_video[:,:,:,:,6] = (video_normalized[:,:,:,:,6] > 0.58).float()
        binary_video[:,:,:,:,7] = (video_normalized[:,:,:,:,7] > 0.60).float()
        binary_video[:,:,:,:,8] = (video_normalized[:,:,:,:,8] > 0.90).float()
        binary_rgb_video_list.append(channel2color(binary_video))

        # if t > 0:
        #     # 创建一个与 video_normalized 形状相同的二值张量
        #     binary_video = torch.zeros_like(video_normalized)

        #     # 遍历每个 channel
        #     for i in range(video_normalized.shape[-1]):
        #         for time in range(video_normalized.shape[1]):
        #             # 获取当前 channel 的数据
        #             channel_data = video_normalized[:,time,:,:,i]
        #             # 将数据展平为一维数组
        #             flattened_data = channel_data.reshape(-1)
        #             # print(channel_data[:,-1])

        #             if i == 1:
        #                 threshold_index = int(flattened_data.numel() * 0.35)
        #             elif i == 3:
        #                 threshold_index = int(flattened_data.numel() * 0.15)
        #             elif i == 5:
        #                 threshold_index = int(flattened_data.numel() * 0.01)

        #             elif i == 6:
        #                 threshold_index = int(flattened_data.numel() * 0.05)
        #             elif i == 7:
        #                 threshold_index = int(flattened_data.numel() * 0.25)
        #             else:
        #                 threshold_value = torch.tensor([1.0])
        #                 threshold_index = int(flattened_data.numel() * 0.01)

        #             # print(threshold_index)
        #             threshold_value = torch.topk(flattened_data, threshold_index, largest=True).values

        #             # 根据阈值设置二值张量
        #             binary_video[:,time,:,:,i] = (channel_data >= threshold_value[-1]).float()
        #             print(i, threshold_value, threshold_value.shape)
        #             # print(binary_video[:,time,:,:,i])

        #     # print(binary_video[:,18, 32,:,6])
        #     binary_rgb_video_list.append(channel2color(binary_video))
        #     # binary_video = (video_normalized > 0.6).float() # 转换为浮点数
        # else:
        #     binary_rgb_video_list.append(channel2color(binary_video))

        if t == 0:
            # 保存 t=0 时的各个 channel 数据
            saved_channels['undetected_obs'] = visual_vid_mmot_each_channel(binary_video, [rec_mot_list[t]], 
                                                                            channel=0, grid=False, frame_t=frame_t, mot_viz=vis_list[0], visual_on=vis_list[0])
            saved_channels['ceiling'] = visual_vid_mmot_each_channel(binary_video, [rec_mot_list[t]], 
                                                                            channel=1, grid=False, frame_t=frame_t, mot_viz=vis_list[1], visual_on=vis_list[1])
            saved_channels['wall'] = visual_vid_mmot_each_channel(binary_video, [rec_mot_list[t]], 
                                                                            channel=3, grid=False, frame_t=frame_t, mot_viz=vis_list[2], visual_on=vis_list[2])
            saved_channels['detected_furniture'] = visual_vid_mmot_each_channel(binary_video, [rec_mot_list[t]], 
                                                                            channel=4, grid=False, frame_t=frame_t, mot_viz=vis_list[3], visual_on=vis_list[3])
            saved_channels['door'] = visual_vid_mmot_each_channel(binary_video, [rec_mot_list[t]], 
                                                                            channel=5, grid=False, frame_t=frame_t, mot_viz=vis_list[4], visual_on=vis_list[4])
            saved_channels['Human'] = visual_vid_mmot_each_channel(binary_video, [rec_mot_list[t]], 
                                                                            channel=6, grid=False, frame_t=frame_t, mot_viz=vis_list[5], visual_on=vis_list[5])
            saved_channels['Floor'] = visual_vid_mmot_each_channel(binary_video, [rec_mot_list[t]], 
                                                                            channel=7, grid=False, frame_t=frame_t, mot_viz=vis_list[6], visual_on=vis_list[6])
        else:
            # 计算 t>0 时的各个 channel 与保存的 channel 之间的曼哈顿距离
            undetected_obs = visual_vid_mmot_each_channel(binary_video, [rec_mot_list[t]], 
                                                                            channel=0, grid=False, frame_t=frame_t, mot_viz=vis_list[0], visual_on=vis_list[0])
            ceiling = visual_vid_mmot_each_channel(binary_video, [rec_mot_list[t]], 
                                                                            channel=1, grid=False, frame_t=frame_t, mot_viz=vis_list[1], visual_on=vis_list[1])
            wall = visual_vid_mmot_each_channel(binary_video, [rec_mot_list[t]], 
                                                                            channel=3, grid=False, frame_t=frame_t, mot_viz=vis_list[2], visual_on=vis_list[2])
            detected_furniture = visual_vid_mmot_each_channel(binary_video, [rec_mot_list[t]], 
                                                                            channel=4, grid=False, frame_t=frame_t, mot_viz=vis_list[3], visual_on=vis_list[3])
            door = visual_vid_mmot_each_channel(binary_video, [rec_mot_list[t]], 
                                                                            channel=5, grid=False, frame_t=frame_t, mot_viz=vis_list[4], visual_on=vis_list[4])
            Human = visual_vid_mmot_each_channel(binary_video, [rec_mot_list[t]], 
                                                                            channel=6, grid=False, frame_t=frame_t, mot_viz=vis_list[5], visual_on=vis_list[5])
            Floor = visual_vid_mmot_each_channel(binary_video, [rec_mot_list[t]], 
                                                                            channel=7, grid=False, frame_t=frame_t, mot_viz=vis_list[6], visual_on=vis_list[6])
            if mean:
                # 计算曼哈顿距离
                manhattan_distances = {
                    'undetected_obs': torch.mean(torch.abs(undetected_obs - saved_channels['undetected_obs'])),
                    'ceiling': torch.mean(torch.abs(ceiling - saved_channels['ceiling'])),
                    'wall': torch.mean(torch.abs(wall - saved_channels['wall'])),
                    'detected_furniture': torch.mean(torch.abs(detected_furniture - saved_channels['detected_furniture'])),
                    'door': torch.mean(torch.abs(door - saved_channels['door'])),
                    'Human': torch.mean(torch.abs(Human - saved_channels['Human'])),
                    'Floor': torch.mean(torch.abs(Floor - saved_channels['Floor']))
                }
            else:
                manhattan_distances = {
                    'undetected_obs': torch.mean(torch.abs(undetected_obs - saved_channels['undetected_obs']), dim=(1,2)),
                    'ceiling': torch.mean(torch.abs(ceiling - saved_channels['ceiling']), dim=(1,2)),
                    'wall': torch.mean(torch.abs(wall - saved_channels['wall']), dim=(1,2)),
                    'detected_furniture': torch.mean(torch.abs(detected_furniture - saved_channels['detected_furniture']), dim=(1,2)),
                    'door': torch.mean(torch.abs(door - saved_channels['door']), dim=(1,2)),
                    'Human': torch.mean(torch.abs(Human - saved_channels['Human']), dim=(1,2)),
                    'Floor': torch.mean(torch.abs(Floor - saved_channels['Floor']), dim=(1,2))
                }
            # 打印或保存曼哈顿距离
            print(f"Frame {t} Manhattan Distances: {manhattan_distances}")
            # 将曼哈顿距离保存到字典中
            distances[f'Sample {t}'] = manhattan_distances

    return distances, binary_rgb_video_list


def visual_MD(distances):
    colors = sns.color_palette("Set2", n_colors=8)
    step = 5
    # 提取每个 channel 的曼哈顿距离
    channels = ['undetected_obs', 'ceiling', 'wall', 'detected_furniture', 'door', 'Human', 'Floor']
    average_channels = [ 'undetected_obs', 'ceiling', 'wall', 'Floor']
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




def visual_MD_combined(*distances_list, Freq=True):
    colors = sns.color_palette("Set2", n_colors=8)
    step = 5

    # 提取每个 channel 的曼哈顿距离
    channels = ['undetected_obs', 'ceiling', 'wall', 'detected_furniture', 'door', 'Human', 'Floor']
    average_channels = ['undetected_obs', 'ceiling', 'wall', 'Floor']
    # label = ['One picked data v.s. Other data in dataset', 'Mean distance of the dataset', 'Real data v.s. Generated data']
    label = ['Mean distance of the dataset', 'Generated samples distance' ]

    channel_distances_list = [{channel: [] for channel in channels} for _ in distances_list]

    for distances, channel_distances in zip(distances_list, channel_distances_list):
        for sample, distance in distances.items():
            for channel in channels:
                channel_distances[channel].append(distance[channel].item())

    # print(channel_distances_list)
    if Freq:
        # 创建一个包含多个子图的图形，2 行 4 列
        fig, axes = plt.subplots(2, 4, figsize=(20, 8),constrained_layout=True)

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

            # ax.set_xlabel('Manhattan Distance')
            # ax.set_ylabel('Frequency')
            ax.set_title(f'{channel}',fontsize=14)
            ax.set_xlim(0, 0.5)  # 设置纵坐标最大值为 50
            ax.set_ylim(0, 50)  # 设置纵坐标最大值为 50
            ax.legend(fontsize=14)
            ax.grid(True, linestyle='--', alpha=0.7)


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
