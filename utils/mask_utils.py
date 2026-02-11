import numpy as np
import torch

def generate_bbox_mask(bbox, shape=(14, 14)):
    # bbox 格式: [min_row, min_col, max_row, max_col]
    min_row, min_col, max_row, max_col = bbox
    
    # 创建一个全为0的掩码
    mask = np.zeros(shape, dtype=int)
    
    # 将边界框内的区域设置为1
    mask[min_row:max_row, min_col:max_col] = 1
    
    return mask

def adjust_mask(intersections, coordinates):
    # 确保输入是 numpy 数组
    intersections = intersections.astype(int)
    coordinates = coordinates.astype(int)

    # 获取掩码的高度和宽度
    height, width = intersections.shape

    # 初始化结果掩码
    result_mask = np.zeros_like(intersections)

    for i in range(height):
        # 获取当前行的 coordinates 和 intersections 的索引
        coord_row = coordinates[i]
        inter_row = intersections[i]

        # 找到 intersections 中第一个和最后一个 `1` 的索引
        inter_indices = np.where(inter_row == 1)[0]
        if len(inter_indices) == 0:
            continue

        inter_first, inter_last = inter_indices[0], inter_indices[-1]

        # 根据 intersections 的范围在 coordinates 中扩展
        for j in range(inter_first, inter_last + 1):
            result_mask[i, j] = 1  # intersections 范围内，coordinates 也为 1

        # 处理 intersections 此行之前的1个patch
        if inter_first - 1 >= 0 and coord_row[inter_first - 1] == 1:
            result_mask[i, inter_first - 1] = 1

        # 处理 intersections 此行之后的1个patch
        if inter_last + 1 <= width-1 and coord_row[inter_last + 1] == 1:
            result_mask[i, inter_last + 1] = 1

    return result_mask
