import torch
from skimage import measure
import cv2
import numpy as np
from utils.mask_utils import generate_bbox_mask, adjust_mask


def AOLM(fms, fm1):
    A = torch.sum(fms, dim=1, keepdim=True) # 沿通道维度求和，得到特征强度
    a = torch.mean(A, dim=[2, 3], keepdim=True) # 求每张特征图的全局均值
    M = (A > a).float() # 按均值二值化，生成主掩码 M

    A1 = torch.sum(fm1, dim=1, keepdim=True)
    a1 = torch.mean(A1, dim=[2, 3], keepdim=True)
    M1 = (A1 > a1).float() # 生成次掩码 M1


    coordinates = []
    for i, m in enumerate(M):
        mask_np = m.cpu().numpy().reshape(14, 14)
        component_labels = measure.label(mask_np)
        # 对 M 掩码进行 连通域分析，提取不同的关注区域（组件）
        properties = measure.regionprops(component_labels)
        areas = []
        for prop in properties:
            areas.append(prop.area)
        max_idx = areas.index(max(areas)) 
        # 找到面积最大的区域后，与 M1 掩码求交集

        intersection = ((component_labels==(max_idx+1)).astype(int) + (M1[i][0].cpu().numpy()==1).astype(int)) ==2
        prop = measure.regionprops(intersection.astype(int))
        if len(prop) == 0:
            bbox = [0, 0, 14, 14]
            print('there is one img no intersection')
        else:
            bbox = prop[0].bbox

        # 将交集区域的边界框从特征图空间映射回原始图像空间（假设缩放比例是 32）
        x_lefttop = bbox[0] * 32 - 1
        y_lefttop = bbox[1] * 32 - 1
        x_rightlow = bbox[2] * 32 - 1
        y_rightlow = bbox[3] * 32 - 1
        # for image
        if x_lefttop < 0:
            x_lefttop = 0
        if y_lefttop < 0:
            y_lefttop = 0
        coordinate = [x_lefttop, y_lefttop, x_rightlow, y_rightlow]
        coordinates.append(coordinate)
    return coordinates

def AOLM2(fms, fm1):
    A = torch.sum(fms, dim=1, keepdim=True) # 沿通道维度求和，得到特征强度 [B,1,14,14]
    a = torch.mean(A, dim=[2, 3], keepdim=True) # 求每张特征图的全局均值 [B,1,1,1]
    M = (A > a).float() # 按均值为阈值进行二值化，生成主掩码 M ,[B,1,14,14]，由0和1构成

    A1 = torch.sum(fm1, dim=1, keepdim=True)
    a1 = torch.mean(A1, dim=[2, 3], keepdim=True)
    M1 = (A1 > a1).float() # 生成次掩码 M1 ,[B,1,14,14]，由0和1构成


    coordinates = []
    intersections=[]
    max_fm_regions=[]
    inter_masks=[]
    for i, m in enumerate(M):
        mask_np = m.cpu().numpy().reshape(14, 14)
        component_labels = measure.label(mask_np) # [14,14]
        # 对 M 掩码进行 连通域分析，提取不同的关注区域（组件）
        properties = measure.regionprops(component_labels)
        areas = []
        for prop in properties:
            areas.append(prop.area)
        max_idx = areas.index(max(areas)) 
        # 找到面积最大的区域后，与 M1 掩码求交集作为最终选定的区域
        # (component_labels==(max_idx+1)).astype(int)表示最大连通区域的二值掩码，[14,14]
        max_fm_region = (component_labels==(max_idx+1)).astype(int)
        intersection = ((component_labels==(max_idx+1)).astype(int) + (M1[i][0].cpu().numpy()==1).astype(int)) ==2
        # 交集掩码，[14,14]
        
        prop = measure.regionprops(intersection.astype(int))
        if len(prop) == 0:
            bbox = [0, 0, 14, 14]
            print('there is one img no intersection')
        else:
            bbox = prop[0].bbox
        
        bbox_coo=generate_bbox_mask(bbox, shape=(14, 14))
        inter_mask=adjust_mask(intersection, bbox_coo)

        inter_masks.append(inter_mask) # 处理后的交集掩膜
        intersections.append(intersection) # 处理前的交集掩膜
        max_fm_regions.append(max_fm_region) # 最大连通区域，不交集

        # 将交集区域的边界框从特征图空间映射回原始图像空间（假设缩放比例是 32）
        x_lefttop = bbox[0] * 32 - 1
        y_lefttop = bbox[1] * 32 - 1
        x_rightlow = bbox[2] * 32 - 1
        y_rightlow = bbox[3] * 32 - 1
        # for image
        if x_lefttop < 0:
            x_lefttop = 0
        if y_lefttop < 0:
            y_lefttop = 0
        coordinate = [x_lefttop, y_lefttop, x_rightlow, y_rightlow]
        coordinates.append(coordinate)
    # 对 intersections 进行放大（32倍）
    intersections_resized = [cv2.resize(intersection.astype(np.float32), None, fx=32, fy=32, interpolation=cv2.INTER_NEAREST)
                            for intersection in intersections] # [448,448]的列表
    # 对 intersections 进行放大（32倍）
    intermasks_resized = [cv2.resize(inter_mask.astype(np.float32), None, fx=32, fy=32, interpolation=cv2.INTER_NEAREST)
                            for inter_mask in inter_masks] # [448,448]的列表
    return coordinates ,intersections_resized,intermasks_resized

