from utils import visual_vid_mmot_each_channel
import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def manhattan_dist(args, dec_vid_list, rec_mot_list, frame_t, input_dim, mean=True):

    saved_channels = {}
    distances = {}
    for t in range(len(dec_vid_list)):
        dec_vid = dec_vid_list[t].reshape(1,args.sequence_length,args.resolution,args.resolution,input_dim)
        min_val = dec_vid.min()
        max_val = dec_vid.max()
        video_normalized = (dec_vid - min_val) / (max_val - min_val)
        binary_video = (video_normalized > 0.6).float() # 转换为浮点数

        if t == 0:
            # 保存 t=0 时的各个 channel 数据
            saved_channels['undetected_obs'] = visual_vid_mmot_each_channel(binary_video, [rec_mot_list[t]], 
                                                                            channel=0, grid=False, frame_t=frame_t, mot_viz=False, visual_on=False)
            saved_channels['ceiling'] = visual_vid_mmot_each_channel(binary_video, [rec_mot_list[t]], 
                                                                            channel=1, grid=False, frame_t=frame_t, mot_viz=False, visual_on=False)
            saved_channels['wall'] = visual_vid_mmot_each_channel(binary_video, [rec_mot_list[t]], 
                                                                            channel=3, grid=False, frame_t=frame_t, mot_viz=False, visual_on=False)
            saved_channels['detected_furniture'] = visual_vid_mmot_each_channel(binary_video, [rec_mot_list[t]], 
                                                                            channel=4, grid=False, frame_t=frame_t, mot_viz=False, visual_on=False)
            saved_channels['door'] = visual_vid_mmot_each_channel(binary_video, [rec_mot_list[t]], 
                                                                            channel=5, grid=False, frame_t=frame_t, mot_viz=False, visual_on=False)
            saved_channels['Human'] = visual_vid_mmot_each_channel(binary_video, [rec_mot_list[t]], 
                                                                            channel=6, grid=False, frame_t=frame_t, mot_viz=False, visual_on=False)
            saved_channels['Floor'] = visual_vid_mmot_each_channel(binary_video, [rec_mot_list[t]], 
                                                                            channel=7, grid=False, frame_t=frame_t, mot_viz=False, visual_on=False)
        else:
            # 计算 t>0 时的各个 channel 与保存的 channel 之间的曼哈顿距离
            undetected_obs = visual_vid_mmot_each_channel(binary_video, [rec_mot_list[t]], 
                                                                            channel=0, grid=False, frame_t=frame_t, mot_viz=False, visual_on=False)
            ceiling = visual_vid_mmot_each_channel(binary_video, [rec_mot_list[t]], 
                                                                            channel=1, grid=False, frame_t=frame_t, mot_viz=False, visual_on=False)
            wall = visual_vid_mmot_each_channel(binary_video, [rec_mot_list[t]], 
                                                                            channel=3, grid=False, frame_t=frame_t, mot_viz=False, visual_on=False)
            detected_furniture = visual_vid_mmot_each_channel(binary_video, [rec_mot_list[t]], 
                                                                            channel=4, grid=False, frame_t=frame_t, mot_viz=False, visual_on=False)
            door = visual_vid_mmot_each_channel(binary_video, [rec_mot_list[t]], 
                                                                            channel=5, grid=False, frame_t=frame_t, mot_viz=False, visual_on=False)
            Human = visual_vid_mmot_each_channel(binary_video, [rec_mot_list[t]], 
                                                                            channel=6, grid=False, frame_t=frame_t, mot_viz=False, visual_on=False)
            Floor = visual_vid_mmot_each_channel(binary_video, [rec_mot_list[t]], 
                                                                            channel=7, grid=False, frame_t=frame_t, mot_viz=False, visual_on=False)
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

    return distances


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

    # sample_num = len(distances)
    # y_tick = np.arange(0, sample_num, 5)
    # # 绘制直方图
    # for channel, color in zip(channels, colors):
    #     distances = channel_distances[channel]
    #     bins = np.arange(0, 0.5, 0.05)  # 设置横坐标为 0 到 0.4，间隔为 0.1
    #     hist, bin_edges = np.histogram(distances, bins=bins)

    #     plt.figure()
    #     plt.bar(bin_edges[:-1], hist, width=0.05, align='edge', color=color, alpha = 0.3)
    #     plt.xlabel('Manhattan Distance')
    #     plt.ylabel('Frequency')
    #     plt.title(f'Manhattan Distance Frequency for {channel}')
    #     plt.xticks(bins)  # 设置横坐标刻度
    #     plt.yticks(y_tick)  # 设置纵坐标刻度
    #     plt.show()

    # # 绘制平均值的直方图
    # distances = average_distances
    # bins = np.arange(0, 0.5, 0.05)  # 设置横坐标为 0 到 0.4，间隔为 0.1
    # hist, bin_edges = np.histogram(distances, bins=bins)

    # plt.figure()
    # plt.plot(bin_edges[:-1], hist, marker='o')
    # plt.xlabel('Manhattan Distance')
    # plt.ylabel('Frequency')
    # plt.title('Average Manhattan Distance Frequency')
    # plt.xticks(bins)  # 设置横坐标刻度
    # plt.yticks(y_tick)  # 设置纵坐标刻度
    # plt.show()


def visual_MD_combined(*distances_list):
    colors = sns.color_palette("Set2", n_colors=8)
    step = 5
    # 提取每个 channel 的曼哈顿距离
    channels = ['undetected_obs', 'ceiling', 'wall', 'detected_furniture', 'door', 'Human', 'Floor']
    average_channels = ['undetected_obs', 'ceiling', 'wall', 'Floor']
    label = ['One picked data v.s. Other data in dataset', 'Mean distance of the dataset', 'One picked data v.s. Generated data']

    channel_distances_list = [{channel: [] for channel in channels} for _ in distances_list]

    for distances, channel_distances in zip(distances_list, channel_distances_list):
        for sample, distance in distances.items():
            for channel in channels:
                channel_distances[channel].append(distance[channel].item())

    # 绘制折线图
    for channel in channels:
        plt.figure()
        for i, channel_distances in enumerate(channel_distances_list):
            sorted_distances = sorted(channel_distances[channel])
            x_stop = len(sorted_distances) + step
            sorted_distances = sorted_distances[::step] + sorted_distances[-1:]
            plt.plot(range(0, x_stop, step), sorted_distances, marker='^', label=label[i])
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
        plt.plot(range(0, x_stop, step), sorted_distances, marker='^', label=label[i])
    plt.xlabel('Sample ID')
    plt.ylabel('Average Manhattan Distance')
    plt.title('Average Manhattan Distance for Each Sample')
    y_tick = np.arange(0, 0.5, 0.05)
    plt.yticks(y_tick)
    plt.legend()
    plt.show()