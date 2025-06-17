"""
************************************* 这里写准备病灶和non-病灶slice的操作 ************************************************ #
"""
import math
import random
from os.path import join
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import subfiles, load_pickle, write_pickle
from sklearn.feature_selection import VarianceThreshold

from dataProcess.path import window_data_2D, resampled_data_2D, lesion_data_dir
from dataProcess.utils import get_case_identifier_from_npz, get_tumor_box_from_mask, load_nii
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from openpyxl import load_workbook
import seaborn as sns
from sklearn.decomposition import NMF

def save_non_tumor_pancreas(save_path):
    """
    此函数将image切片，读入经过重采样和[300, 50]调整的图像
    :param data: data[0]=image data[-1]=mask
    :param index: 含有肿瘤的索引 现在保存不含肿瘤区域的 最近 各2切片
    :param save_path: /non_tumor_pancreas
    """
    slice_sum = 0
    nii_file_list = subfiles(join(window_data_2D, "images"), True, None, ".nii.gz", True)
    for j, case in enumerate(nii_file_list):
        case_identifier = case.split("/")[-1][:-7]
        data_i, data_m = load_nii(window_data_2D, case_identifier)
        image, mask = sitk.GetArrayFromImage(data_i), sitk.GetArrayFromImage(data_m)
        pkl = load_pickle(join(resampled_data_2D, "images", case_identifier+".pkl"))
        index = pkl["tumor_index"]
        new_index = [index[0]-2, index[0]-1, index[-1]+1, index[-1]+2]
        pan_index = [i for i in new_index if i >= 0 and i < image.shape[0]]
        print("size: ", image.shape[0], "index: ", pan_index)
        pkl["pan_index"] = np.array(pan_index)
        write_pickle(pkl,join(resampled_data_2D, "images", case_identifier+".pkl")) # 将胰腺组织的索引
        for i in pan_index:
            slice_sum += 1
            file_name = case_identifier + '_' + str(i)
            np.savez_compressed(join(save_path, "non_tumor_pancreas", "%s.npz" % file_name), data=image[i].astype(np.float32))
            print("Save ", file_name)
    print(slice_sum) # 10465

def save_npz_to_slice(save_path):
    """
    此函数将image切片，读入经过重采样和[300, 50]调整的图像
    :param data: data[0]=image data[-1]=mask
    :param index: 含有肿瘤的索引
    :param save_path: /images  /masks
    """
    slice_sum = 0
    nii_file_list = subfiles(join(window_data_2D, "images"), True, None, ".nii.gz", True)
    for j, case in enumerate(nii_file_list):
        case_identifier = case.split("/")[-1][:-7]
        data_i, data_m = load_nii(window_data_2D, case_identifier)
        image, mask = sitk.GetArrayFromImage(data_i), sitk.GetArrayFromImage(data_m)
        pkl = load_pickle(join(resampled_data_2D, "images", case_identifier+".pkl"))
        index = pkl["tumor_index"]
        for i in index:
            slice_sum += 1
            file_name = case_identifier + '_' + str(i)
            np.savez_compressed(join(save_path, "images", "%s.npz" % file_name), data=image[i].astype(np.float32))
            np.savez_compressed(join(save_path, "masks", "%s.npz" % file_name), data=mask[i].astype(np.float32))
            print("Save ", file_name)
    print(slice_sum) # 10465

def cut_non_lesion_image(image, x_center, y_center, d):
    d = int(d / 1.5) # 选择裁剪非肿瘤组织半径为d 实际大小2d*2d框是模仿和组织区域的量级相同 但容易裁出区域 顾适当缩小半径但也大于贴边直径
    # 计算非病灶组织中心的种子点范围,是当前肿瘤直径为d 是贴边的肿瘤直径大小
    radius = random.choice([1.45, 1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 2]) * d # , 1.75, 1.8, 1.85, 1.9, 2,
    angle = random.uniform(0, 2 * math.pi) # 在圆周上随机选择一个角度
    new_point_x = x_center + radius * math.cos(angle)
    new_point_y = y_center + radius * math.sin(angle)
    resizer = (slice(max(0,int(new_point_y - d)), min(image.shape[0],int(new_point_y + d))),
               slice(max(0,int(new_point_x - d)), min(image.shape[1],int(new_point_x + d))))
    rect = (max(0,int(new_point_x - d)), min(image.shape[0],int(new_point_y + d)), 2*d)
    return image[resizer], rect # 左下角

def is_intersecting(rectangle1, rectangle2):
    x1, y1, d1 = rectangle1  # 第一个方形区域的左下角坐标 (x1, y1) 和宽度 d1
    x2, y2, d2 = rectangle2  # 第二个方形区域的左下角坐标 (x2, y2) 和宽度 d2
    if (x1 < x2 + d2 and x1 + d1 > x2 and y1 > y2 - d2 and y1 - d1 < y2):
    # if max(x1, x2) <= min(x1+d1, x2+d2) and max(y1, y2) <= min(y1-d1, y2-d2): # 相交矩形的左下角的点 相交矩形的右上角的点
        return True
    else:
        return False

def calculate_zero_percentage(image):
    count_zero = 0
    total_elements = 0
    for row in image:
        for element in row:
            if element == 0:
                count_zero += 1
            total_elements += 1
    zero_percentage = count_zero / total_elements
    return zero_percentage

def save_non_lesion_slice():
    lesion_npz_file_list = subfiles(join(lesion_data_dir, "images"), True, None, ".npz", True)
    for j, case in enumerate(lesion_npz_file_list):
        case_identifier = case.split("/")[-1][:12]
        properties = load_pickle(join(resampled_data_2D, "images", case_identifier+".pkl"))
        image = np.load(case)['data']
        x_center, y_center = properties["tumor_center"][2], properties["tumor_center"][1]
        max_length = properties["tumor_max_length"]
        rect1 = (properties["tumor_box"][2][0], properties["tumor_box"][1][1], max_length) # 左下坐标是x小y大
        non_lesion_image, rect2 = cut_non_lesion_image(image, x_center, y_center, max_length / 2)
        zero_percent = 0.25
        choose_cnt = 1
        while is_intersecting(rect1, rect2) or calculate_zero_percentage(non_lesion_image) > zero_percent:
            # plt.imshow(non_lesion_image, cmap='gray')
            # plt.show()
            if choose_cnt % 50 == 0: zero_percent += 0.02
            choose_cnt += 1
            print(case_identifier, "Intersecting or Zero")
            non_lesion_image, rect2 = cut_non_lesion_image(image, x_center, y_center, max_length / 2)
        plt.imshow(non_lesion_image, cmap='gray')
        plt.show()
        np.savez_compressed(join(lesion_data_dir, "non_lesion_images", case.split("/")[-1]), data=non_lesion_image.astype(np.float32))
        print("save ", j, case_identifier)

def cut_target_index_image(case_identifier, properties, tum_index, pan_index):
    """
    :param lesion_npz_file_path:
    :param pan_npz_file_path:
    :param tum_index:
    :param pan_index:
    :return:
    """
    lesion_npz_file_path = join(lesion_data_dir, "images")
    pan_npz_file_path = join(lesion_data_dir, "non_tumor_pancreas")
    x_center, y_center = properties["tumor_center"][2], properties["tumor_center"][1]
    max_length = properties["tumor_max_length"]
    rect1 = (properties["tumor_box"][2][0], properties["tumor_box"][1][1], max_length) # 左下坐标是x小y大

    for index in tum_index: # 按照适当缩小裁剪框 在肿瘤矩形框附近不重叠区域裁剪
        case = case_identifier+'_'+str(index)+'.npz'
        image = np.load(join(lesion_npz_file_path, case))['data']
        non_lesion_image, rect2 = cut_non_lesion_image(image, x_center, y_center, max_length)
        zero_percent = 0.2
        choose_cnt = 1
        while is_intersecting(rect1, rect2) or calculate_zero_percentage(non_lesion_image) > zero_percent:
            # plt.imshow(non_lesion_image, cmap='gray')
            # plt.show()
            if choose_cnt % 50 == 0: zero_percent += 0.02
            choose_cnt += 1
            print(case_identifier, "Intersecting or Zero")
            non_lesion_image, rect2 = cut_non_lesion_image(image, x_center, y_center, max_length)
        plt.imshow(non_lesion_image, cmap='gray')
        plt.show()
        np.savez_compressed(join(lesion_data_dir, "non_lesion_images_2", case), data=non_lesion_image.astype(np.float32))
        print("save ", case)

    for index in pan_index: # 直接按照肿瘤中心的2d*2d框进行裁剪
        case = case_identifier+'_'+str(index)+'.npz'
        image = np.load(join(pan_npz_file_path, case))['data']
        resizer = (slice(max(0,int(y_center - max_length)), min(image.shape[0],int(y_center + max_length))),
                   slice(max(0,int(x_center - max_length)), min(image.shape[1],int(x_center + max_length))))
        non_lesion_image = image[resizer]
        np.savez_compressed(join(lesion_data_dir, "non_lesion_images_2", case), data=non_lesion_image.astype(np.float32))
        print("save ", case)

def save_non_lesion_and_pancreas_images():
    """
    这里负责改良非病灶区域的裁剪 综合非病灶区域的胰腺
    1.直接纳入ct的pkl文件入文件列表，读取对应的tumor层和pancreas层index，根据算法均匀提取得到新的index，分别在两文件夹读取
    2.当index数目>=7时，全部纳入pancreas 替换对应数目的tumor
      当index数目<=6时，各占一半 tumor占index/2,pancreas占len(index)-index/2
    :return:
    """
    pkl_file_list = subfiles(join(resampled_data_2D, "images"), True, None, ".pkl", True)
    for j, case in enumerate(pkl_file_list):
        case_identifier = case.split("/")[-1][:12]
        properties = load_pickle(case)
        tum_index = properties["tumor_index"]
        pan_index = properties["pan_index"]
        if len(tum_index) >= 7: # 根据原index的长度计算纳入多少非肿瘤胰腺区 如果>=7则全纳入
            tum_index = tum_index[np.linspace(0, len(tum_index)-1, len(tum_index) - len(pan_index), dtype='int')]
        else: # 反之则各取一半
            tum_index = tum_index[np.linspace(0, len(tum_index)-1, len(tum_index) // 2, dtype='int')] # tumor类向下取整
            pan_index = pan_index[np.linspace(0, len(pan_index)-1, math.ceil(len(properties["tumor_index"]) / 2), dtype='int')] # pan类向上取整
        cut_target_index_image(case_identifier, properties, tum_index, pan_index) # 各自裁剪并保存


def center_crop(image, output_size_rate):
    height, width = image.shape #(y, x)
    new_width, new_height = int(height * output_size_rate), int(width * output_size_rate) # (0.8y, 0.8x)
    left = (width - new_width) // 2
    top = (height - new_height) // 2 # 左上
    right = left + new_width
    bottom = top + new_height # 右下
    cropped_image = image[top:bottom, left:right]
    return cropped_image


def main():
    save_npz_to_slice('/data/hsx/project_pc_multimodality_addsize/Task_Pancreas_sec/lesion_data_2D')

    # 这里完成 data 的切片保存
    # save_non_tumor_pancreas(lesion_data_dir)
    # 这里完成non-lesion-data 的切片保存
    # save_non_lesion_and_pancreas_images()
    # 以下是检查
    # lesion_npz_file_list = subfiles(join(lesion_data_dir, "non_tumor_pancreas"), True, None, ".npz", True)
    # for i, case in enumerate(lesion_npz_file_list):
    #     image = np.load(case)['data']
    #     plt.imshow(image, cmap='gray')
    #     plt.show()
    #     zero_percentage = calculate_zero_percentage(image)
    #     if zero_percentage >= 0.3: # 80%范围中心裁剪
    #         print(i, "case: ", case.split("/")[-1], "percent: ", zero_percentage)
    #         plt.imshow(image, cmap='gray')
    #         plt.show()
    #         cropped = center_crop(image, 0.8)
    #         plt.imshow(cropped, cmap='gray')
    #         plt.show()
    #         np.savez_compressed(join(lesion_data_dir, "non_lesion_images", case.split("/")[-1]), data=cropped.astype(np.float32))

    # 现完成对于影像组学数据的降维处理
    data = pd.read_excel('./Pancreas-Radiomics-features.xlsx', sheet_name="Sheet1")
    # print(data)
    X = data.iloc[:, 1:]
    id = data.iloc[:, 0]
    variance = data.var()

    threshold_84_value = np.percentile(variance, 84)
    threshold_87_value = np.percentile(variance, 87)
    threshold_90_value = np.percentile(variance, 90)

    selected_84_features = variance[variance >= threshold_84_value]
    selected_87_features = variance[variance >= threshold_87_value]
    selected_90_features = variance[variance >= threshold_90_value]

    selector = VarianceThreshold(threshold=threshold_87_value)

    X_filtered = selector.fit_transform(X) #
    nmf_model = NMF(n_components=60)
    X_reduced = nmf_model.fit_transform(X_filtered)

if __name__ == "__main__":
    main()