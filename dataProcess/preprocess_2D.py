# 该文件的2D预处理仅负责xy维度重采样+窗宽窗位固定值调整/肿瘤区域调整/normalization应用强度信息调整
# 使用raw_data数据+cropped_data中保存的pkl信息
#                 对应加入键值spacing_after_resample, tumor_index[], tumor_size[], tumor_volume
# 后续补充加入了准备2D分割数据的cropped CT的XY轴重采样和nnu强度标准化
# ******************************************************************************************************************** #
import glob
import pickle
import shutil
from batchgenerators.utilities.file_and_folder_operations import subfiles
from multiprocessing import Pool
from dataProcess.utils import get_case_identifier_from_niigz, load_nii, get_and_save_properties, window_transform, \
    split_dataset_list, save_to_different_set, load_pickle, get_case_identifier_from_npz, load_cropped_npz
from path import raw_data, cropped_data, resampled_data_2D, seg_data_2D,\
    preprocessing_output_dir, window_data_2D, nnu_window_data_2D, divided_data_dir
import SimpleITK as sitk
import os
from os.path import join
import numpy as np
import pandas as pd
from skimage.transform import resize
from collections import OrderedDict

def do_case_nnu_window_normalize(case_identifier, window_data_2D, resampled_data_2D):
    plans = load_pickle(join(preprocessing_output_dir, "preprocess_plans_3D.pkl"))
    intensityproperties = plans['dataset_properties']['intensityproperties']
    image, _ = load_nii(resampled_data_2D, case_identifier) # , properties
    data = sitk.GetArrayFromImage(image)
    mean_intensity = intensityproperties[0]['mean'] # 58.03475
    std_intensity = intensityproperties[0]['sd'] # 24.691725
    lower_bound = intensityproperties[0]['percentile_00_5'] # 3.0
    upper_bound = intensityproperties[0]['percentile_99_5'] # 139.0
    data = np.clip(data, lower_bound, upper_bound)
    data = (data - mean_intensity) / std_intensity
    res_image = sitk.GetImageFromArray(data)
    res_image.SetSpacing(image.GetSpacing())
    res_image.SetOrigin(image.GetOrigin())
    res_image.SetDirection(image.GetDirection())
    sitk.WriteImage(res_image, join(window_data_2D, "images", case_identifier+".nii.gz"))

def prepare_nnu_window(resampled_data_2D, window_data_2D):
    # if os.path.isdir(join(window_data_2D, "masks")):
    #     shutil.rmtree(join(window_data_2D, "masks"))
    # shutil.copytree(join(resampled_data_2D, "masks"), join(window_data_2D, "masks"))
    list_of_resampled_data_files = subfiles(join(resampled_data_2D, "images"), True, None, ".nii.gz", True)
    all_args = []
    for j, case in enumerate(list_of_resampled_data_files): # [1476:]
        case_identifier = get_case_identifier_from_niigz(case)
        args = case_identifier, window_data_2D, resampled_data_2D
        all_args.append(args)
        do_case_nnu_window_normalize(*args)
        print("Done ", j)
    # p = Pool(8)
    # p.starmap(do_case_window_normalize, all_args)
    # p.close()

def cal_window(ct_mask, ct_data):   # 这一层开始调整窗宽窗位by tumor(recommended)
    seg_tumor = ct_mask == 1
    ct_tumor = ct_data * seg_tumor
    tumor_min = ct_tumor.min()
    tumor_max = ct_tumor.max()
    tumor_wide = tumor_max - tumor_min
    tumor_center = (tumor_max + tumor_min) / 2
    return tumor_wide, tumor_center

def do_case_window_normalize(case_identifier, window_data_2D, resampled_data_2D):
    image, mask = load_nii(resampled_data_2D, case_identifier) # , properties
    image_npz = sitk.GetArrayFromImage(image)
    mask_npz = sitk.GetArrayFromImage(mask)
    tumor_wide, tumor_center = 300, 50 # cal_window(mask_npz, image_npz)
    pancreatic_wl_image = window_transform(image_npz, tumor_wide, tumor_center, normal=True)
    res_image = sitk.GetImageFromArray(pancreatic_wl_image)
    res_image.SetSpacing(image.GetSpacing())
    res_image.SetOrigin(image.GetOrigin())
    res_image.SetDirection(image.GetDirection())
    sitk.WriteImage(res_image, join(window_data_2D, "images", case_identifier+".nii.gz"))


def prepare_window(resampled_data_2D, window_data_2D):
    if os.path.isdir(join(window_data_2D, "masks")):
        shutil.rmtree(join(window_data_2D, "masks"))
    shutil.copytree(join(resampled_data_2D, "masks"), join(window_data_2D, "masks"))
    list_of_resampled_data_files = subfiles(join(resampled_data_2D, "images"), True, None, ".nii.gz", True)
    all_args = []
    for j, case in enumerate(list_of_resampled_data_files): # [1476:]
        case_identifier = get_case_identifier_from_niigz(case)
        args = case_identifier, window_data_2D, resampled_data_2D
        all_args.append(args)
        # do_case_window_normalize(*args)
        # print("Done ", j)
    p = Pool(8)
    p.starmap(do_case_window_normalize, all_args)
    p.close()


def do_case_resample_XY(target_spacing, case_identifier, output_folder_stage_resample, raw_data_path):
    image, mask = load_nii(raw_data_path, case_identifier) # , properties
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    # 计算重采样的缩放比例
    scale_factors = [original_spacing[i] / target_spacing[i] if target_spacing[i] != 0 else 1 for i in range(3)]
    new_size = [int(original_size[i] * scale_factors[i]) for i in range(3)]
    target_spacing[2] = original_spacing[2]
    # 创建重采样滤波器
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    # 执行重采样
    resampler.SetInterpolator(sitk.sitkLinear)
    resampled_image = resampler.Execute(image)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampled_mask = resampler.Execute(mask)
    get_and_save_properties(mask, resampled_mask, output_folder_stage_resample, case_identifier)
    sitk.WriteImage(resampled_image, join(output_folder_stage_resample, "images", case_identifier+".nii.gz"))
    sitk.WriteImage(resampled_mask, join(output_folder_stage_resample, "masks", case_identifier+".nii.gz"))
    print("Finish resampling %s" % case_identifier)

def prepare_resample_2D(raw_data_path, target_spacing, output_folder_stage_resample):
    list_of_raw_data_files = subfiles(join(raw_data_path, "images"), True, None, ".nii.gz", True)
    all_args = []
    for j, case in enumerate(list_of_raw_data_files[20:]): # [206:] [1476:]
        case_identifier = get_case_identifier_from_niigz(case)
        args = target_spacing, case_identifier, output_folder_stage_resample, raw_data_path
        all_args.append(args)
        print("Doing case ", j + 20, ": ", case_identifier)
        do_case_resample_XY(*args)
    # p = Pool(8)
    # p.starmap(do_case_resample_XY, all_args)
    # p.close()
    # p.join()

def resize_segmentation(segmentation, new_shape, order=0):
    '''
    Resizes a segmentation map. Supports all orders (see skimage documentation). Will transform segmentation map to one
    hot encoding which is resized and transformed back to a segmentation map.
    This prevents interpolation artifacts ([0, 0, 2] -> [0, 1, 2])
    :param order: 0 表示最邻近插值
    '''
    tpe = segmentation.dtype
    return resize(segmentation.astype(float), new_shape, order, mode="edge", clip=True, anti_aliasing=False).astype(tpe)

def do_case_seg_resample_XY(data, new_shape, is_seg):
    if is_seg:
        order = 0
        resize_fn = resize_segmentation
        kwargs = OrderedDict()
    else:
        order = 3
        resize_fn = resize
        kwargs = {'mode': 'edge', 'anti_aliasing': False}
    dtype_data = data.dtype
    shape = np.array(data.shape)
    new_shape = np.array(new_shape)
    if np.any(shape != new_shape):
        data = data.astype(float)
        reshaped_seg = []
        for slice_id in range(shape[0]): # slice数目 单张slice做归一化
            reshaped_seg.append(resize_fn(data[slice_id], new_shape[1:], order, **kwargs).astype(dtype_data))
        reshaped_seg = np.stack(reshaped_seg, 0)
        return reshaped_seg
    else:
        return data

def do_case_seg_norm(data):
    plans = load_pickle(join(preprocessing_output_dir, "preprocess_plans_3D.pkl"))
    intensityproperties = plans['dataset_properties']['intensityproperties']
    mean_intensity = intensityproperties[0]['mean'] # 58.03475
    std_intensity = intensityproperties[0]['sd'] # 24.691725
    lower_bound = intensityproperties[0]['percentile_00_5'] # 3.0
    upper_bound = intensityproperties[0]['percentile_99_5'] # 139.0
    data = np.clip(data, lower_bound, upper_bound)
    data = (data - mean_intensity) / std_intensity
    return data

def do_case_seg_resample_norm_XY(target_spacing, case_identifier, output_folder_seg_data, raw_data_path):
    image, mask = load_cropped_npz(raw_data_path, case_identifier)
    properties = load_pickle(join(raw_data_path, case_identifier+".pkl"))
    original_spacing = properties["original_spacing"]
    target_spacing[0] = original_spacing[0]
    shape = np.array(image.shape)
    new_shape = np.round(((np.array(original_spacing) / np.array(target_spacing)).astype(float) * shape)).astype(int)
    resample_image = do_case_seg_resample_XY(image, new_shape, False)
    resample_mask = do_case_seg_resample_XY(mask, new_shape, True)

    data = np.stack((resample_image, resample_mask), axis=0)
    np.savez_compressed(os.path.join(output_folder_seg_data, "resample_xy", "%s.npz"
                                     % case_identifier), data=data.astype(np.float32)) # 保存仅重采样的 此行说明
    properties["size_after_resampling"] = resample_image.shape
    properties["spacing_after_resampling"] = target_spacing

    norm_image = do_case_seg_norm(resample_image)
    all_data = np.stack((norm_image, resample_mask), axis=0)
    np.savez_compressed(os.path.join(output_folder_seg_data, "%s.npz" % case_identifier),
                        data=all_data.astype(np.float32))
    with open(os.path.join(output_folder_seg_data, "%s.pkl" % case_identifier), 'wb') as f:
        pickle.dump(properties, f)

def prepare_2d_seg_resample_norm(cropped_data, seg_data_2D, target_spacing=None):
    """
    这次读入npz完成重采样的操作 其中img=npz[0] seg=npz[-1]
    :param cropped_data: 原CT文件经过去机床和裁剪黑边操作
    :param seg_data_2D: 依次保留病人的重采样结果和标准化结果
    :param target_spacing: Z轴不进行采样 标记为0待更改为图片自身的Z轴体素
    :return: 保存结果
    """
    if target_spacing is None:
        target_spacing = [0, 0.7421875, 0.7421875]
    list_of_cropped_data_files = subfiles(cropped_data, True, None, ".npz", True)
    all_args = []
    for j, case in enumerate(list_of_cropped_data_files): #  [32:] [1476:] 10186807_2_v
        case_identifier = get_case_identifier_from_npz(case)
        args = target_spacing, case_identifier, seg_data_2D, cropped_data
        all_args.append(args)
        print("Doing case: ", j, case_identifier)
        do_case_seg_resample_norm_XY(*args)

def main():

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_preprocess", required=False, default=False,
                        help="set this flag to preprocess raw data including resampling, adjust window and normalization!")
    parser.add_argument("--seg_data_preprocess", required=False, default=True,
                        help="set this flag to use cropped data to do 2D resample and normalization")
    parser.add_argument("--target_spacing", type=list, required=False, default=[0, 0.72656, 0.72656]) # SEC [0.72656 0.72656 4] PR [0.75781202, 0.75781202 5] SD [0.7421875, 0.7421875, 0]
    parser.add_argument("--divided_patient_sets_path", type=str, required=False,
                        default="/data/hsx/project_pc_multimodality_addsize/Task_Pancreas/data_info/divided_patient_sets.xlsx", # "../Task_Pancreas/data_info/divided_patient_sets.xlsx",
                        help="set this path to get different separation of dataset")
    parser.add_argument("--divided_ways", type=str, default="622")
    args = parser.parse_args()

    # 经过检查有10416611_1_a & 10656805_0_d在重采样中出现错误 现单独重采样
    # 10617936_0_a
    # params = args.target_spacing, '10617936_0_a', resampled_data_2D, raw_data
    # do_case_resample_XY(*params) # 重采样默认只用一次
    # params = args.target_spacing, '10186807_2_v', seg_data_2D, cropped_data
    # do_case_seg_resample_norm_XY(*params) # 窗宽窗位调整仅一次

    if args.raw_data_preprocess:
        prepare_resample_2D(raw_data, args.target_spacing, resampled_data_2D)
        prepare_window(resampled_data_2D, window_data_2D) # 这个是固定窗宽窗位调整加标准化 因为任务是病灶分类 不便于纳入病灶先验信息
        # prepare_nnu_window(resampled_data_2D, nnu_window_data_2D)

    if args.seg_data_preprocess: # 做2d分割前的预处理
        prepare_2d_seg_resample_norm(cropped_data, seg_data_2D, args.target_spacing)


    ### 做分类 划分数据集 ###
    # trainList, valList, testList = split_dataset_list(args.divided_patient_sets_path, args.divided_ways)
    # # 这里目前使用仅resample后的数据
    # save_to_different_set(trainList, valList, testList, divided_data_dir, window_data_2D, join(resampled_data_2D, "images"))

if __name__ == "__main__":
    main()