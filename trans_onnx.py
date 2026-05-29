# coding=utf-8
"""
trans_onnx.py
====================================================================================
将 SG-KRL (MainNet) 中可被 ONNX 导出的"骨干计算部分"导出为 ONNX 模型。

【为什么是分段导出，而不是整模型一把梭】
MainNet.forward 里混杂了三类无法被 torch.onnx.export 处理的内容：
  1. AOLM2: 基于 skimage 连通域分析 + cv2.resize 的纯 CPU/NumPy 算法，且含数据依赖控制流；
  2. APPM + nms: 含 NumPy 排序、Python while/for 循环的非极大值抑制；
  3. WindowGCN: 依赖 torch_geometric 的 GCNConv / global_mean_pool，不在 ONNX 标准算子集内，
     且 edge_index 由 kNN 动态计算（双重 Python for 循环）。
这三部分都属于"控制流 / 图操作 / 第三方稀疏算子"，trace/script 均无法导出。

【因此本脚本只导出纯卷积/注意力计算密集型的骨干子图】，共两个 ONNX：
  - backbone.onnx :  ResNet50 主干 -> (fm, embedding, conv5_b, conv3_d, conv4_f)
                     再接 fusion4 / fusion3，输出 fused_features。
                     对应 forward 中对 self.pretrained_model(x) + 两个 fusion 的调用。
                     支持动态 batch / 动态 H、W（因为骨干在全局/局部/窗口三处被复用，输入尺寸不同）。
  - local_head.onnx: CBAM(local_fm) + rawcls_net(embedding) 的分类头部分。
                     注意：CBAM 的输入是骨干输出的 local_fm，rawcls_net 的输入是 embedding，
                     这里把"对 14x14 特征图做 CBAM"和"对 2048 向量做线性分类"两件事打包。

AOLM2 / APPM-nms / GCN 在 test_onnx.py 中继续用原始 PyTorch 代码执行（不改原代码）。

【依赖】
  torch >= 1.8, onnx, (导出本身不需要 onnxruntime)
  本工程的 networks / utils / config 模块需可被 import（即在工程根目录运行本脚本）。

【运行】
  python trans_onnx.py
导出的 onnx 文件默认写到 ./onnx/ 目录下（相对工程根目录）。
====================================================================================
"""

import os
import sys

import torch
import torch.nn as nn

# ----------------------------------------------------------------------------------
# 工程内模块。要求在工程根目录运行，使 networks / utils / config 可被正常 import。
# ----------------------------------------------------------------------------------
from config import proposalN, channels, input_size
from networks.model import MainNet
from utils.auto_laod_resume import auto_load_resume


# ==================================================================================
# 配置区（全部相对路径，产物落在工程目录内）
# ==================================================================================
# 茅台 8 类
NUM_CLASSES = 8

# 训练好的权重路径（相对工程根目录）。若不存在，则用随机初始化权重导出，
# 仅用于打通流程 / 校验结构；真正部署务必放入真实权重。
PTH_PATH = './checkpoints/best.pth'

# ONNX 产物输出目录
ONNX_DIR = './onnx'
BACKBONE_ONNX = os.path.join(ONNX_DIR, 'backbone.onnx')
LOCAL_HEAD_ONNX = os.path.join(ONNX_DIR, 'local_head.onnx')

# ONNX opset。13 能较好支持 AdaptiveAvgPool / softmax / matmul 等算子。
OPSET = 13

DEVICE = torch.device('cpu')  # 导出在 CPU 上做即可，权重已在 CPU


# ==================================================================================
# 子模块包装：从已加载好的 MainNet 中"借用"其子模块权重，组成可导出的子图。
# 这样做的好处是——不复制权重、不改原模型定义，仅在导出期重组前向计算。
# ==================================================================================
class BackboneONNX(nn.Module):
    """
    封装骨干 + 双路特征融合，对应 MainNet.forward 的前半段：
        fm, embedding, conv5_b, conv3_d, conv4_f = pretrained_model(x)
        fused_4 = fusion4(fm, conv4_f)
        fused_3 = fusion3(fm, conv3_d)
        fused_features = cat([fused_4, fused_3, fm], dim=1)

    输入:
        x: [B, 3, H, W]  (全局/局部分支 H=W=448，窗口分支 H=W=224)
    输出:
        fm             : [B, 2048, H/32, W/32]
        embedding      : [B, 2048]
        conv5_b        : [B, 2048, H/32, W/32]  (layer4 前两个 block 的输出，AOLM2 的次掩码来源)
        fused_features : [B, 3584, H/32, W/32]  (2048+1024+512, 仅在 448 输入时形状有意义)

    说明:
      - 窗口分支(224 输入)实际只用到 embedding，但为保持单一 ONNX 复用，
        统一输出全部张量；调用方按需取用即可。
      - fusion4/fusion3 内部用了 F.adaptive_avg_pool2d，对动态 H/W 友好。
    """

    def __init__(self, main_net: MainNet):
        super(BackboneONNX, self).__init__()
        self.pretrained_model = main_net.pretrained_model
        self.fusion4 = main_net.fusion4
        self.fusion3 = main_net.fusion3

    def forward(self, x):
        fm, embedding, conv5_b, conv3_d, conv4_f = self.pretrained_model(x)
        fused_4 = self.fusion4(fm, conv4_f)          # [B, 1024, h, w]
        fused_3 = self.fusion3(fm, conv3_d)          # [B, 512,  h, w]
        fused_features = torch.cat([fused_4, fused_3, fm], dim=1)  # [B, 3584, h, w]
        return fm, embedding, conv5_b, fused_features


class LocalHeadONNX(nn.Module):
    """
    封装局部分支的分类头部分，对应 MainNet.forward 中：
        local_fm = self.CBAM(local_fm)          # 仅对特征图做注意力(本工程未把它接入后续分类，仅保留计算)
        local_logits = self.rawcls_net(local_embeddings)

    输入:
        local_fm        : [B, 2048, 14, 14]   骨干对 local_imgs 提取的特征图
        local_embedding : [B, 2048]           骨干对 local_imgs 的全局向量
    输出:
        cbam_fm     : [B, 2048, 14, 14]        CBAM 注意力后的特征图(与原 forward 行为一致地计算出来)
        local_logits: [B, num_classes]         局部分支分类 logits（test.py 用它做 object 分支准确率）

    说明:
      - 原 forward 里 local_fm 经 CBAM 后并未再喂给 rawcls_net（分类用的是 local_embeddings），
        CBAM 结果实际只参与了 APPM 的窗口打分(self.APPM(..., local_fm.detach(), ...))。
        因此这里把 CBAM 输出也返回，供 test_onnx.py 喂给 APPM。
    """

    def __init__(self, main_net: MainNet):
        super(LocalHeadONNX, self).__init__()
        self.CBAM = main_net.CBAM
        self.rawcls_net = main_net.rawcls_net

    def forward(self, local_fm, local_embedding):
        cbam_fm = self.CBAM(local_fm)                     # [B, 2048, 14, 14]
        local_logits = self.rawcls_net(local_embedding)   # [B, num_classes]
        return cbam_fm, local_logits


# ==================================================================================
# 工具函数
# ==================================================================================
def build_main_net():
    """构造 MainNet 并尽力加载权重。返回 eval 模式的模型。"""
    model = MainNet(proposalN=proposalN, num_classes=NUM_CLASSES, channels=channels)
    model = model.to(DEVICE)

    if os.path.exists(PTH_PATH):
        # auto_load_resume 内部会剥离 'module.' 前缀，兼容 DataParallel 保存的权重
        epoch = auto_load_resume(model, PTH_PATH, status='test')
        print('[trans_onnx] 已加载权重: {}  (epoch={})'.format(PTH_PATH, epoch))
    else:
        print('[trans_onnx] 警告: 未找到权重 {}，将使用随机初始化权重导出。'
              '\n             导出的 ONNX 仅可用于结构/流程校验，部署前请放入真实权重后重新导出。'
              .format(PTH_PATH))

    model.eval()
    return model


def export_backbone(main_net):
    """导出 backbone.onnx（动态 batch + 动态 H/W）。"""
    backbone = BackboneONNX(main_net).to(DEVICE).eval()

    # 用 448 输入做 trace（全局/局部分支尺寸）。窗口分支 224 通过 dynamic_axes 兼容。
    dummy = torch.randn(1, 3, input_size, input_size, device=DEVICE)

    input_names = ['input']
    output_names = ['fm', 'embedding', 'conv5_b', 'fused_features']
    dynamic_axes = {
        'input':          {0: 'batch', 2: 'height', 3: 'width'},
        'fm':             {0: 'batch', 2: 'fh', 3: 'fw'},
        'embedding':      {0: 'batch'},
        'conv5_b':        {0: 'batch', 2: 'fh', 3: 'fw'},
        'fused_features': {0: 'batch', 2: 'fh', 3: 'fw'},
    }

    with torch.no_grad():
        torch.onnx.export(
            backbone,
            dummy,
            BACKBONE_ONNX,
            export_params=True,
            opset_version=OPSET,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )
    print('[trans_onnx] 已导出: {}'.format(BACKBONE_ONNX))


def export_local_head(main_net):
    """导出 local_head.onnx（CBAM + rawcls_net，动态 batch）。"""
    head = LocalHeadONNX(main_net).to(DEVICE).eval()

    # local_fm: [B,2048,14,14]; local_embedding: [B,2048]
    feat_side = input_size // 32  # 448/32 = 14
    dummy_fm = torch.randn(1, channels, feat_side, feat_side, device=DEVICE)
    dummy_emb = torch.randn(1, channels, device=DEVICE)

    input_names = ['local_fm', 'local_embedding']
    output_names = ['cbam_fm', 'local_logits']
    dynamic_axes = {
        'local_fm':        {0: 'batch'},
        'local_embedding': {0: 'batch'},
        'cbam_fm':         {0: 'batch'},
        'local_logits':    {0: 'batch'},
    }

    with torch.no_grad():
        torch.onnx.export(
            head,
            (dummy_fm, dummy_emb),
            LOCAL_HEAD_ONNX,
            export_params=True,
            opset_version=OPSET,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )
    print('[trans_onnx] 已导出: {}'.format(LOCAL_HEAD_ONNX))


def try_check_onnx():
    """若安装了 onnx 包，做一次结构合法性检查（不依赖 onnxruntime）。"""
    try:
        import onnx
    except ImportError:
        print('[trans_onnx] 未安装 onnx 包，跳过结构检查。可 pip install onnx 后重试。')
        return

    for path in (BACKBONE_ONNX, LOCAL_HEAD_ONNX):
        model = onnx.load(path)
        onnx.checker.check_model(model)
        print('[trans_onnx] 结构检查通过: {}'.format(path))


def main():
    os.makedirs(ONNX_DIR, exist_ok=True)

    main_net = build_main_net()

    export_backbone(main_net)
    export_local_head(main_net)

    try_check_onnx()

    print('\n[trans_onnx] 完成。')
    print('  - 骨干(含特征融合)     : {}'.format(BACKBONE_ONNX))
    print('  - 局部分类头(CBAM+FC)   : {}'.format(LOCAL_HEAD_ONNX))
    print('  注意: AOLM2 / APPM-NMS / GCN 不可导出，已在 test_onnx.py 中以原始 PyTorch 执行。')


if __name__ == '__main__':
    main()
