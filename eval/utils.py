from utils import visual_vid_mmot_each_channel
import torch
import matplotlib.pyplot as plt

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
        plt.bar(range(1, len(sorted_distances) + 1), sorted_distances)
        plt.xlabel('Sample')
        plt.ylabel('Manhattan Distance')
        plt.title(f'Manhattan Distance for {channel}')
        # plt.xticks(range(1, len(sorted_distances) + 2, 10))  # 设置横坐标为正整数
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
    plt.bar(range(1, len(sorted_distances) + 1), sorted_distances)
    plt.xlabel('Sample')
    plt.ylabel('Average Manhattan Distance')
    plt.title('Average Manhattan Distance for Each Sample')
    plt.show()