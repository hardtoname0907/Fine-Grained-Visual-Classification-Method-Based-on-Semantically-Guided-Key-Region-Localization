import torch
from torch import nn
import torch.nn.functional as F
from networks import resnet
from config import pretrain_path, coordinates_cat, iou_threshs, window_nums_sum, ratios, N_list
import numpy as np
from utils.AOLM import AOLM,AOLM2
from utils.CBAM import CBAM
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
from .gcn import *
from networks.qkv_attn import FeatureFusionAttention


def nms(scores_np, proposalN, iou_threshs, coordinates):
    if not (type(scores_np).__module__ == 'numpy' and len(scores_np.shape) == 2 and scores_np.shape[1] == 1):
        raise TypeError('score_np is not right')

    windows_num = scores_np.shape[0] # 窗口总数
    indices_coordinates = np.concatenate((scores_np, coordinates), 1) # 合并，形成新的矩阵 [windows_num, 5]，每行是 [score, x1, y1, x2, y2]

    indices = np.argsort(indices_coordinates[:, 0]) # 升序排列，返回分数列的排序索引
    indices_coordinates = np.concatenate((indices_coordinates, np.arange(0,windows_num).reshape(windows_num,1)), 1)[indices]   #[339,6]
    # 首先，将一个从 0 到 windows_num-1 的索引列添加到 indices_coordinates，用于在筛选后保留原始窗口索引
    # 其次，用排序索引重新排列 indices_coordinates，此时indices_coordinates形状为 [windows_num, 6]，每行是 [score, x1, y1, x2, y2, original_index]
    indices_results = [] # 保存最终保留的窗口索引

    res = indices_coordinates # res: 剩余的候选窗口

    while res.any():
        indice_coordinates = res[-1] # res[-1]是当前剩余窗口中分数最高的（因为按分数升序排序）
        indices_results.append(indice_coordinates[5]) # 将其索引（original_index）添加到 indices_results

        if len(indices_results) == proposalN:
            return np.array(indices_results).reshape(1,proposalN).astype(int) 
        # 如果保留的窗口数量达到 proposalN，直接返回结果
        res = res[:-1] # 从剩余窗口移除最后一个元素（最高分）

        # Exclude anchor boxes with selected anchor box whose iou is greater than the threshold
        # 计算剩余窗口与最高分窗口的 IoU
        start_max = np.maximum(res[:, 1:3], indice_coordinates[1:3])
        end_min = np.minimum(res[:, 3:5], indice_coordinates[3:5])
        lengths = end_min - start_max + 1
        intersec_map = lengths[:, 0] * lengths[:, 1]
        intersec_map[np.logical_or(lengths[:, 0] < 0, lengths[:, 1] < 0)] = 0
        iou_map_cur = intersec_map / ((res[:, 3] - res[:, 1] + 1) * (res[:, 4] - res[:, 2] + 1) +
                                      (indice_coordinates[3] - indice_coordinates[1] + 1) *
                                      (indice_coordinates[4] - indice_coordinates[2] + 1) - intersec_map)
        res = res[iou_map_cur <= iou_threshs] # 保留 IoU 小于等于 iou_threshs 的窗口

    while len(indices_results) != proposalN:
        indices_results.append(indice_coordinates[5])
    # 如果循环结束时保留的窗口不足 proposalN，通过重复最后一个窗口的索引补足数量。

    return np.array(indices_results).reshape(1, -1).astype(int)

class APPM(nn.Module):
    def __init__(self):
        super(APPM, self).__init__()
        self.avgpools = [nn.AvgPool2d(ratios[i], 1) for i in range(len(ratios))] # 多个池化部件

    def forward(self, proposalN, x, ratios, window_nums_sum, N_list, iou_threshs, DEVICE='cuda'):
        batch, channels, _, _ = x.size()
        avgs = [self.avgpools[i](x) for i in range(len(ratios))]
        # 列表，每个元素对应一种窗口大小的输出，形状为 [batch, 1, new_height, new_width]
        # feature map sum
        fm_sum = [torch.sum(avgs[i], dim=1) for i in range(len(ratios))] 
        # 不同窗口大小生成的特征图列表，每个元素形状为 [batch, new_height, new_width]

        all_scores = torch.cat([fm_sum[i].view(batch, -1, 1) for i in range(len(ratios))], dim=1) # [batch, total_num_windows, 1]
        # fm_sum[i].view(batch, -1, 1)——》[batch, num_windows, 1]，num_windows = new_height * new_width
        # 再沿着num_windows连接起来，总数成为total_num_windows
        windows_scores_np = all_scores.data.cpu().numpy() # [batch, num_windows, 1]
        window_scores = torch.from_numpy(windows_scores_np).to(DEVICE).reshape(batch, -1) # [batch, total_num_windows]
        # 感觉这块有点问题，为啥用的是np啊？难道不该用不带np的吗，需要的是三维还是二维？
        # nms
        proposalN_indices = []
        for i, scores in enumerate(windows_scores_np):
            indices_results = []
            for j in range(len(window_nums_sum)-1): # 这个数是4-1=3，表示大中小共3组窗口，遍历 window_nums_sum 的每个元素（除了最后一个）
                """
                scores[sum(window_nums_sum[:j + 1]):sum(window_nums_sum[:j + 2])]：获取当前比例组的得分。
                N_list[j]：获取当前比例组的提议窗口数量。
                iou_threshs[j]：获取当前比例组的IoU阈值。
                coordinates_cat[sum(window_nums_sum[:j + 1]):sum(window_nums_sum[:j + 2])]：获取当前比例组的坐标信息。
                sum(window_nums_sum[:j + 1])：计算当前比例组前所有比例组的窗口数量和，用于调整索引
                """
                indices_results.append(nms(scores[sum(window_nums_sum[:j+1]):sum(window_nums_sum[:j+2])], proposalN=N_list[j], iou_threshs=iou_threshs[j],
                                           coordinates=coordinates_cat[sum(window_nums_sum[:j+1]):sum(window_nums_sum[:j+2])]) + sum(window_nums_sum[:j+1]))
            # indices_results.reverse()
            proposalN_indices.append(np.concatenate(indices_results, 1))   # reverse
            # 将每个样本的最终候选窗口索引保存到 proposalN_indices

        proposalN_indices = np.array(proposalN_indices).reshape(batch, proposalN)
        proposalN_indices = torch.from_numpy(proposalN_indices).to(DEVICE) 
        # proposalN_indices 形状为 [batch, proposalN]。每一行对应一个样本的窗口索引
        proposalN_windows_scores = torch.cat(
            [torch.index_select(all_score, dim=0, index=proposalN_indices[i]) for i, all_score in enumerate(all_scores)], 0).reshape(
            batch, proposalN)
        # 在得分张量中，按照第 i 个样本的窗口索引选取对应的窗口得分，输出形状为 [proposalN]
        return proposalN_indices, proposalN_windows_scores, window_scores


class MainNet(nn.Module):
    def __init__(self, proposalN, num_classes, channels):
        # nn.Module子类的函数必须在构造函数中执行父类的构造函数
        super(MainNet, self).__init__()
        self.num_classes = num_classes
        self.proposalN = proposalN
        # self.pretrained_model = resnet.resnet101(pretrained=True, pth_path=pretrain_path)
        self.pretrained_model = resnet.resnet50(pretrained=True, pth_path=pretrain_path)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.dropout = nn.Dropout(0.5)
        self.rawcls_net = nn.Linear(channels, num_classes)
        self.CBAM = CBAM(in_channels=channels)
        self.APPM = APPM()
        self.gcn_model = WindowGCN(input_dim=channels, hidden_dim=512, output_dim=256, num_classes=num_classes)
        self.fusion4 = FeatureFusionAttention(2048, 1024, num_heads=8, head_dim=64, dropout=0.2)
        self.fusion3 = FeatureFusionAttention(2048, 512, num_heads=8, head_dim=32, dropout=0.2)

    def forward(self, x, epoch=0, batch_idx=0, status='test', DEVICE='cuda'):
        fm, embedding, conv5_b,conv3_d,conv4_f = self.pretrained_model(x)
        batch_size, channel_size, side_size, _ = fm.shape
        assert channel_size == 2048
        # 使用高层特征引导融合低层特征
        fused_4 = self.fusion4(fm, conv4_f)  # [B, 1024, 14, 14]
        fused_3 = self.fusion3(fm, conv3_d)  # [B, 512, 14, 14]
        # 融合所有特征
        fused_features = torch.cat([fused_4, fused_3, fm], dim=1)  # [B, 2048+1024+512, 14, 14]
        # raw branch
        raw_logits = self.rawcls_net(embedding) # delete 1

        #SCDA
        # coordinates = torch.tensor(AOLM(fused_features.detach(), conv5_b.detach()))
        # coordinates = torch.tensor(AOLM(fm.detach(), conv5_b.detach()))

        coordinates,intersections,intermasks_np = AOLM2(fused_features.detach(), conv5_b.detach())  # 获取坐标和交集
        # # 将列表转换为 tensor
        coordinates = torch.tensor(coordinates).to(DEVICE)  # 将坐标转换为 tensor，并移动到正确的设备
        # # intersections 可能包含布尔值或者 0 和 1 的值，直接转换为 tensor
        # intersections = torch.tensor(intersections).to(DEVICE)  # 同样将交集转换为 tensor
        intermasks = torch.tensor(intermasks_np).to(DEVICE)  # 同样将mask转换为 tensor

        local_imgs = torch.zeros([batch_size, 3, 448, 448]).to(DEVICE)  # [N, 3, 448, 448]
        for i in range(batch_size):
            [x0, y0, x1, y1] = coordinates[i]
            # if epoch <= 45:
            #     local_imgs[i:i + 1] = F.interpolate(x[i:i + 1, :, x0:(x1+1), y0:(y1+1)], size=(448, 448),
            #                                     mode='bilinear', align_corners=True)  # [N, 3, 224, 224]
            # else:
            masked_x = x[i:i + 1, :, :, :] * intermasks[i:i + 1].unsqueeze(0)
            # 截取坐标范围内的区域
            cropped_x = masked_x[:, :, x0:(x1+1), y0:(y1+1)]  # [1, 3, height, width]
            # 使用插值调整裁剪区域大小为 [B, 3, 448, 448]
            local_imgs[i:i + 1] = F.interpolate(cropped_x, size=(448, 448), mode='bilinear', align_corners=True)


            # local_imgs[i:i + 1] = F.interpolate(x[i:i + 1, :, x0:(x1+1), y0:(y1+1)], size=(448, 448),
            #                                     mode='bilinear', align_corners=True)  # [N, 3, 224, 224]
        local_fm, local_embeddings, _ ,_,_= self.pretrained_model(local_imgs.detach())  # [N, 2048]
        local_fm=self.CBAM(local_fm)
        local_logits = self.rawcls_net(local_embeddings)  # [N, 200]
        # fm均为[B,2048,14,14]
        proposalN_indices, proposalN_windows_scores, window_scores \
            = self.APPM(self.proposalN, local_fm.detach(), ratios, window_nums_sum, N_list, iou_threshs, DEVICE)

        if status == "train":
            # window_imgs cls
            window_imgs = torch.zeros([batch_size, self.proposalN, 3, 224, 224]).to(DEVICE)  # [N, 4, 3, 224, 224]
            for i in range(batch_size):
                for j in range(self.proposalN):
                    [x0, y0, x1, y1] = coordinates_cat[proposalN_indices[i, j]]
                    window_imgs[i:i + 1, j] = F.interpolate(local_imgs[i:i + 1, :, x0:(x1 + 1), y0:(y1 + 1)], size=(224, 224),
                                                                mode='bilinear',
                                                                align_corners=True)  # [N, 4, 3, 224, 224]

            window_imgs = window_imgs.reshape(batch_size * self.proposalN, 3, 224, 224)  # [N*4, 3, 224, 224]
            window_fm, window_embeddings, _,_,_ = self.pretrained_model(window_imgs.detach())  # [N*4, 2048]
            proposalN_windows_logits = self.rawcls_net(window_embeddings)  # [N* 4, 200]
        else:
             # window_imgs cls
            window_imgs = torch.zeros([batch_size, self.proposalN, 3, 224, 224]).to(DEVICE)  # [N, 4, 3, 224, 224]
            for i in range(batch_size):
                for j in range(self.proposalN):
                    [x0, y0, x1, y1] = coordinates_cat[proposalN_indices[i, j]]
                    window_imgs[i:i + 1, j] = F.interpolate(local_imgs[i:i + 1, :, x0:(x1 + 1), y0:(y1 + 1)], size=(224, 224),
                                                                mode='bilinear',
                                                                align_corners=True)  # [N, 4, 3, 224, 224]

            window_imgs = window_imgs.reshape(batch_size * self.proposalN, 3, 224, 224)  # [N*4, 3, 224, 224]
            window_fm, window_embeddings, _,_,_ = self.pretrained_model(window_imgs.detach())  # [N*4, 2048]
            proposalN_windows_logits = torch.zeros([batch_size * self.proposalN, self.num_classes]).to(DEVICE)

        # Step 2: Process with GCN
        # proposalN_indices 的形状是 [batch_size, proposalN]
        # 将它展平成一维索引数组
        flattened_indices = proposalN_indices.reshape(-1)
        # 从 coordinates_cat 中提取相应的坐标
        window_coordinates = coordinates_cat[flattened_indices.cpu().numpy()]
        # Flatten to [N*proposalN, 4]
        window_coordinates = torch.tensor(window_coordinates).to(DEVICE)
        gcn_logits = process_windows_with_gcn(batch_size, self.proposalN, window_embeddings, window_coordinates, self.gcn_model, DEVICE)
        # [B,200]

        return proposalN_windows_scores, proposalN_windows_logits, proposalN_indices, \
               window_scores, coordinates, raw_logits, local_logits, local_imgs,gcn_logits #,intersections,intermasks_np
