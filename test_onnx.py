# coding=utf-8
"""
test_onnx.py
====================================================================================
使用 trans_onnx.py 导出的 ONNX 骨干 + 原始 PyTorch 的 AOLM2/APPM/GCN，
复刻 MainNet.forward(status='test') 的完整推理链路，并在数据集上评测
"object 分支(local_logits)" 与 "GCN 分支(gcn_logits)" 的准确率。

【整体数据流】（与 networks/model.py 的 MainNet.forward 完全对齐）
  x[B,3,448,448]
    │ (ONNX backbone)
    ├─> fm, embedding, conv5_b, fused_features
    │
    ├─ raw_logits = rawcls_net(embedding)                    # 全局分支(评测未用，仅保留)
    │
    │ (PyTorch AOLM2)  fused_features, conv5_b -> coordinates, intermasks
    ├─ 按坐标+掩码裁剪并插值 -> local_imgs[B,3,448,448]
    │
    │ (ONNX backbone, 复用)  local_imgs -> local_fm, local_embeddings
    │ (ONNX local_head)      local_fm, local_embeddings -> cbam_fm, local_logits
    │
    │ (PyTorch APPM+NMS)     cbam_fm -> proposalN_indices, ...
    ├─ 按窗口坐标裁剪 local_imgs -> window_imgs[B*proposalN,3,224,224]
    │
    │ (ONNX backbone, 复用)  window_imgs -> window_embeddings
    │ (PyTorch GCN)          window_embeddings + 窗口坐标 -> gcn_logits
    ▼
  评测 local_logits / gcn_logits

【为什么 GCN 仍走 PyTorch】
WindowGCN 依赖 torch_geometric 的 GCNConv / global_mean_pool，不在 ONNX 标准算子集内；
且其 edge_index 由 kNN 在运行时动态计算。按需求"不修改原代码"，GCN 保留 PyTorch 推理。

【依赖】
  torch, onnxruntime (或 onnxruntime-gpu), 以及本工程的 networks/utils/config/datasets。
  torch_geometric 仍需安装(GCN 用)。

【运行前提】
  1. 已执行 trans_onnx.py，生成 ./onnx/backbone.onnx 与 ./onnx/local_head.onnx
  2. 已放入真实权重 ./checkpoints/best.pth（GCN 等 PyTorch 子模块需要它）
  3. 数据集目录结构符合 datasets.dataset.Geners 的 ImageFolder 要求（root/test/<类名>/*.jpg）

【运行】
  python test_onnx.py
====================================================================================
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# ----------------------------------------------------------------------------------
# 工程内模块
# ----------------------------------------------------------------------------------
from config import (
    input_size, proposalN, channels,
    coordinates_cat, iou_threshs, window_nums_sum, ratios, N_list,
)
from networks.model import MainNet
from utils.AOLM import AOLM2
from utils.auto_laod_resume import auto_load_resume
from utils.read_dataset import read_dataset


# ==================================================================================
# 配置区（相对路径）
# ==================================================================================
NUM_CLASSES = 8                                  # 茅台 8 类
SET = 'GEN'                                       # 用 Geners(ImageFolder) 读取，与 config 默认一致
ROOT = './datasets/maotai'                        # 数据集根目录(需含 test/ 子目录)；按需修改
PTH_PATH = './checkpoints/best.pth'               # PyTorch 权重(GCN 等子模块需要)
BACKBONE_ONNX = './onnx/backbone.onnx'
LOCAL_HEAD_ONNX = './onnx/local_head.onnx'
BATCH_SIZE = 1                                     # 茅台测试 batch=1

CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if CUDA else 'cpu')


# ==================================================================================
# ONNX Runtime 推理封装
# ==================================================================================
def build_ort_session(onnx_path):
    """创建 onnxruntime 推理会话，优先用 GPU(若可用)。"""
    import onnxruntime as ort

    if not os.path.exists(onnx_path):
        sys.exit('[test_onnx] 找不到 ONNX 文件: {}，请先运行 trans_onnx.py'.format(onnx_path))

    providers = []
    avail = ort.get_available_providers()
    if CUDA and 'CUDAExecutionProvider' in avail:
        providers.append('CUDAExecutionProvider')
    providers.append('CPUExecutionProvider')

    sess = ort.InferenceSession(onnx_path, providers=providers)
    print('[test_onnx] 载入 ONNX: {}  providers={}'.format(onnx_path, sess.get_providers()))
    return sess


def run_backbone(sess, x_tensor):
    """
    运行 backbone.onnx。
    输入:  x_tensor [B,3,H,W] (torch.Tensor, 任意 device)
    输出:  (fm, embedding, conv5_b, fused_features) 均为 torch.Tensor，放到 DEVICE 上。
    """
    x_np = x_tensor.detach().cpu().numpy().astype(np.float32)
    fm, embedding, conv5_b, fused_features = sess.run(
        ['fm', 'embedding', 'conv5_b', 'fused_features'],
        {'input': x_np},
    )
    to = lambda a: torch.from_numpy(a).to(DEVICE)
    return to(fm), to(embedding), to(conv5_b), to(fused_features)


def run_local_head(sess, local_fm, local_embedding):
    """
    运行 local_head.onnx。
    输入:  local_fm [B,2048,14,14], local_embedding [B,2048] (torch.Tensor)
    输出:  (cbam_fm, local_logits) torch.Tensor，放到 DEVICE 上。
    """
    fm_np = local_fm.detach().cpu().numpy().astype(np.float32)
    emb_np = local_embedding.detach().cpu().numpy().astype(np.float32)
    cbam_fm, local_logits = sess.run(
        ['cbam_fm', 'local_logits'],
        {'local_fm': fm_np, 'local_embedding': emb_np},
    )
    to = lambda a: torch.from_numpy(a).to(DEVICE)
    return to(cbam_fm), to(local_logits)


# ==================================================================================
# 推理主体：复刻 MainNet.forward(status='test')，骨干换成 ONNX
# ==================================================================================
class OnnxInferencer(object):
    """
    持有:
      - 两个 ONNX 会话(backbone / local_head)
      - 一个 PyTorch MainNet 实例(仅用其 APPM / gcn_model / rawcls_net 等"非骨干"子模块及其权重)
    用 ONNX 跑骨干、用 PyTorch 跑 AOLM2/APPM/GCN，串联出与原 forward 一致的输出。
    """

    def __init__(self):
        # 1) ONNX 会话
        self.backbone_sess = build_ort_session(BACKBONE_ONNX)
        self.local_head_sess = build_ort_session(LOCAL_HEAD_ONNX)

        # 2) PyTorch 模型(加载完整权重，供 GCN / APPM / rawcls_net 使用)
        self.model = MainNet(proposalN=proposalN, num_classes=NUM_CLASSES, channels=channels).to(DEVICE)
        if os.path.exists(PTH_PATH):
            epoch = auto_load_resume(self.model, PTH_PATH, status='test')
            print('[test_onnx] 已加载 PyTorch 权重: {} (epoch={})'.format(PTH_PATH, epoch))
        else:
            print('[test_onnx] 警告: 未找到权重 {}，GCN/分类头为随机权重，准确率无意义。'.format(PTH_PATH))
        self.model.eval()

        self.proposalN = proposalN
        self.num_classes = NUM_CLASSES

    @torch.no_grad()
    def infer(self, x):
        """
        单个 batch 推理。
        输入:  x [B,3,448,448] (已归一化, torch.Tensor)
        返回:  local_logits [B,num_classes], gcn_logits [B,num_classes]
        """
        x = x.to(DEVICE)
        batch_size = x.size(0)

        # ---------- 1. 全局骨干(ONNX) ----------
        fm, embedding, conv5_b, fused_features = run_backbone(self.backbone_sess, x)
        # raw_logits 评测未使用，但保留以对齐原 forward 语义
        raw_logits = self.model.rawcls_net(embedding)  # noqa: F841

        # ---------- 2. AOLM2 定位(PyTorch, 原始实现) ----------
        # 注意: AOLM2 内部用 .cpu().numpy()，输入需为 tensor。与原 forward 一致传 detach 后的张量。
        coordinates, intersections, intermasks_np = AOLM2(fused_features.detach(), conv5_b.detach())
        coordinates = torch.tensor(coordinates).to(DEVICE)           # [B,4]
        intermasks = torch.tensor(np.array(intermasks_np)).to(DEVICE)  # [B,448,448]

        # ---------- 3. 裁剪生成 local_imgs(与原 forward 的 else 分支一致) ----------
        # 原 forward: epoch<=45 走直接裁剪，否则走 mask 裁剪。test.py 调用时 epoch 来自循环下标 i，
        # 评测语义上等价于"训练后期(epoch>45)"的 mask 裁剪路径，这里固定走 mask 裁剪以匹配最终模型行为。
        local_imgs = torch.zeros([batch_size, 3, 448, 448]).to(DEVICE)
        for i in range(batch_size):
            x0, y0, x1, y1 = [int(v) for v in coordinates[i]]
            masked_x = x[i:i + 1, :, :, :] * intermasks[i:i + 1].unsqueeze(0)
            cropped_x = masked_x[:, :, x0:(x1 + 1), y0:(y1 + 1)]
            local_imgs[i:i + 1] = torch.nn.functional.interpolate(
                cropped_x, size=(448, 448), mode='bilinear', align_corners=True)

        # ---------- 4. 局部骨干(ONNX 复用) + 局部分类头(ONNX) ----------
        local_fm, local_embeddings, _, _ = run_backbone(self.backbone_sess, local_imgs)
        cbam_fm, local_logits = run_local_head(self.local_head_sess, local_fm, local_embeddings)

        # ---------- 5. APPM 窗口提议(PyTorch, 原始实现) ----------
        # 原 forward: self.APPM(self.proposalN, local_fm.detach(), ratios, window_nums_sum, N_list, iou_threshs, DEVICE)
        # 这里的 local_fm 已被 CBAM 处理(对应原 forward 的 local_fm=self.CBAM(local_fm))，故传 cbam_fm。
        proposalN_indices, proposalN_windows_scores, window_scores = self.model.APPM(
            self.proposalN, cbam_fm.detach(), ratios, window_nums_sum, N_list, iou_threshs, DEVICE)

        # ---------- 6. 裁剪生成 window_imgs(与原 forward else 分支一致) ----------
        window_imgs = torch.zeros([batch_size, self.proposalN, 3, 224, 224]).to(DEVICE)
        for i in range(batch_size):
            for j in range(self.proposalN):
                x0, y0, x1, y1 = [int(v) for v in coordinates_cat[proposalN_indices[i, j]]]
                window_imgs[i:i + 1, j] = torch.nn.functional.interpolate(
                    local_imgs[i:i + 1, :, x0:(x1 + 1), y0:(y1 + 1)], size=(224, 224),
                    mode='bilinear', align_corners=True)
        window_imgs = window_imgs.reshape(batch_size * self.proposalN, 3, 224, 224)

        # ---------- 7. 窗口骨干(ONNX 复用) ----------
        _, window_embeddings, _, _ = run_backbone(self.backbone_sess, window_imgs)

        # ---------- 8. GCN 分支(PyTorch, 原始实现) ----------
        # 与原 forward 完全一致：展平索引 -> 取窗口坐标 -> process_windows_with_gcn
        from networks.gcn import process_windows_with_gcn
        flattened_indices = proposalN_indices.reshape(-1)
        window_coordinates = coordinates_cat[flattened_indices.cpu().numpy()]
        window_coordinates = torch.tensor(window_coordinates).to(DEVICE)
        gcn_logits = process_windows_with_gcn(
            batch_size, self.proposalN, window_embeddings, window_coordinates,
            self.model.gcn_model, DEVICE)

        return local_logits, gcn_logits


# ==================================================================================
# 评测主流程（对齐 test.py 的统计口径）
# ==================================================================================
def main():
    # 读取测试集。Geners 的 __getitem__ 返回 (img, target)，与 test.py 中 set!='CUB' 分支一致。
    _, testloader = read_dataset(input_size, BATCH_SIZE, ROOT, SET)

    inferencer = OnnxInferencer()

    object_correct = 0
    gcn_correct = 0
    total = len(testloader.dataset)

    print('[test_onnx] 开始评测，共 {} 张测试图...'.format(total))
    for data in tqdm(testloader):
        # Geners 返回 (img, target)
        x, y = data
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        local_logits, gcn_logits = inferencer.infer(x)

        # object 分支(local)
        pred = local_logits.max(1, keepdim=True)[1]
        object_correct += pred.eq(y.view_as(pred)).sum().item()

        # gcn 分支
        gcn_pred = gcn_logits.max(1, keepdim=True)[1]
        gcn_correct += gcn_pred.eq(y.view_as(gcn_pred)).sum().item()

    print('\nObject branch accuracy: {}/{} ({:.2f}%)'.format(
        object_correct, total, 100. * object_correct / total))
    print('GCN branch accuracy:    {}/{} ({:.2f}%)'.format(
        gcn_correct, total, 100. * gcn_correct / total))


if __name__ == '__main__':
    main()
