import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
import numpy as np

class WindowGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_classes):
        super(WindowGCN, self).__init__()
        # 第一层 GCN
        self.conv1 = GCNConv(input_dim, hidden_dim) # 2048——512
        # 第二层 GCN
        # self.conv2 = GCNConv(hidden_dim, hidden_dim)
        # 第三层 GCN
        self.conv3 = GCNConv(hidden_dim, output_dim) #512——256
        # 分类层
        self.classifier = torch.nn.Linear(output_dim, num_classes)  # 最终用于分类200

    def forward(self, x, edge_index, batch):
        # 第一层 GCN + 激活函数 + Dropout
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        # # 第二层 GCN + 激活函数 + Dropout
        # x = self.conv2(x, edge_index)
        # x = F.relu(x)
        # x = F.dropout(x, p=0.5, training=self.training)
        
        # 第三层 GCN
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        
        # 全局均值池化，将节点特征聚合为图特征
        x = global_mean_pool(x, batch)  # [batch_size, output_dim]
        
        # 图级别分类
        x = self.classifier(x)  # [batch_size, num_classes]
        
        return x


def compute_adjacency_matrix(coordinates, k):
    """
    Compute adjacency matrix using k-NN based on Euclidean distance.
    :param coordinates: Tensor of shape [num_nodes, 4] containing window coordinates [x0, y0, x1, y1].
    :param k: Number of nearest neighbors to connect.
    :return: edge_index for PyG.
    """
    num_nodes = coordinates.size(0)
    distances = torch.cdist(coordinates[:, :2].float(), coordinates[:, :2].float(), p=2)  # Use top-left corner
    knn_indices = distances.topk(k + 1, largest=False).indices[:, 1:]  # Exclude self-loop

    edge_index = []
    for i in range(num_nodes):
        for j in knn_indices[i]:
            edge_index.append([i, j.item()])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return edge_index

def process_windows_with_gcn(batch_size, proposalN, window_embeddings, window_coordinates, gcn_model, device):
    """
    Process window embeddings with GCN.
    :param batch_size: Number of samples in the batch.
    :param proposalN: Number of proposal windows per sample.
    :param window_embeddings: Tensor of shape [N*proposalN, 2048].
    :param window_coordinates: Tensor of shape [N*proposalN, 4].
    :param gcn_model: GCN model instance.
    :param device: Device to perform computation on.
    :return: GCN outputs.
    """
    
    # batch_indices = torch.arange(batch_size).repeat_interleave(proposalN).to(device)  # Batch indices for PyG
    # for onnx 
    # 更改repeat算子后实验torch测试准确率的情况————已测试96.98%
    batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(batch_size, proposalN).contiguous().view(-1)

    edge_index = compute_adjacency_matrix(window_coordinates, k=3).to(device)  # k-NN graph k=3 best

    gcn_outputs = gcn_model(window_embeddings, edge_index, batch_indices)  # [batch_size, output_dim]
    return gcn_outputs

# Integration with existing pipeline
class ImprovedModel(nn.Module):
    def __init__(self, pretrained_model, rawcls_net, gcn_model):
        super(ImprovedModel, self).__init__()
        self.pretrained_model = pretrained_model
        self.rawcls_net = rawcls_net
        self.gcn_model = gcn_model

    def forward(self, local_imgs, coordinates, proposalN_indices, batch_size, proposalN, device):
        # Step 1: Extract window embeddings and coordinates
        window_imgs = torch.zeros([batch_size, proposalN, 3, 224, 224]).to(device)
        for i in range(batch_size):
            for j in range(proposalN):
                [x0, y0, x1, y1] = coordinates[proposalN_indices[i, j]]
                window_imgs[i:i + 1, j] = F.interpolate(local_imgs[i:i + 1, :, x0:(x1 + 1), y0:(y1 + 1)], size=(224, 224),
                                                        mode='bilinear', align_corners=True)

        window_imgs = window_imgs.reshape(batch_size * proposalN, 3, 224, 224)
        window_fm, window_embeddings, _ = self.pretrained_model(window_imgs.detach())

        # Step 2: Process with GCN
        window_coordinates = coordinates[proposalN_indices.reshape(-1)]  # Flatten to [N*proposalN, 4]
        gcn_outputs = process_windows_with_gcn(batch_size, proposalN, window_embeddings, window_coordinates, self.gcn_model, device)

        # Step 3: Classification
        logits = self.rawcls_net(gcn_outputs)  # Final classification logits
        return logits

