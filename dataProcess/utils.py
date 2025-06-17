# 此函数封装预处理中独立的工具函数
import pickle
import re
import shutil

import skimage
import numpy as np
import pandas as pd
import json
import glob
import os
from os.path import join
import SimpleITK as sitk
from collections import OrderedDict
from batchgenerators.utilities.file_and_folder_operations import subfiles
from dataProcess.path import cropped_data, preprocessing_output_dir
from skimage.transform import resize
from scipy.ndimage.interpolation import map_coordinates


"""
*********************************************  my preprocess tools   ***************************************************
"""
def get_case_identifier_from_niigz(case):
    case_identifier = case.split("/")[-1][:-7]
    return case_identifier

def get_case_identifier_from_npz(case):
    case_identifier = case.split("/")[-1][:-4]
    return case_identifier

def load_nii(file_path, case_identifier):
    image = sitk.ReadImage(join(file_path, "images", "%s.nii.gz" % case_identifier))
    mask = sitk.ReadImage(join(file_path, "masks", "%s.nii.gz" % case_identifier))
    return image, mask

def load_cropped_npz(file_path, case_identifier):
    data = np.load(join(file_path, "%s.npz" % case_identifier))
    image = data['data'][0]
    mask = data['data'][-1]
    return image, mask

def get_tumor_box_from_mask(mask, outside_value=0):
    """
    边界，寻找mask中xyz轴维度里不为0的范围
    """
    mask_voxel_coords = np.where(mask == 1)#!= outside_value) # 非肿瘤区域全0 [z, y, x]
    if mask.size == 0:
        print("---------MASK SIZE IS NONE---------")
        return None
    minzidx = int(np.min(mask_voxel_coords[0]))
    maxzidx = int(np.max(mask_voxel_coords[0])) + 1
    minyidx = int(np.min(mask_voxel_coords[1]))
    maxyidx = int(np.max(mask_voxel_coords[1])) + 1
    minxidx = int(np.min(mask_voxel_coords[2]))
    maxxidx = int(np.max(mask_voxel_coords[2])) + 1
    return [[minzidx, maxzidx], [minyidx, maxyidx], [minxidx, maxxidx]]

def get_and_save_properties(mask, resampled_mask, save_path, case_identifier):
    properties = dict()
    properties['case_identifier'] = case_identifier
    properties['itk_origin'] = mask.GetOrigin()
    properties['itk_spacing'] = mask.GetSpacing()
    properties['itk_direction'] = mask.GetDirection()
    properties['itk_size'] = mask.GetSize()
    properties['size_after_resample'] = resampled_mask.GetSize()
    properties['spacing_after_resample'] = resampled_mask.GetSpacing()
    res_mk_npz = sitk.GetArrayFromImage(resampled_mask)
    index = np.where(np.any(res_mk_npz == 1, axis=(1, 2)))[0]
    properties['tumor_index'] = index
    counts = []  # 存储每一层中数值为1的像素总数的列表
    for z in range(len(index)):     # 遍历每一层
        layer = res_mk_npz[int(index[z]), :, :]  # 获取当前层
        count = np.count_nonzero(layer == 1)  # 统计数值为1的像素个数
        counts.append(count)  # 将统计结果添加到列表中
    properties['tumor_area'] = counts
    box = get_tumor_box_from_mask(res_mk_npz)
    if box != None:
        properties['tumor_box'] = box
        point_z, point_y, point_x = (box[0][0] + box[0][1]) / 2, (box[1][0] + box[1][1]) / 2, (box[2][0] + box[2][1]) / 2
        properties['tumor_center'] = [point_z, point_y, point_x]
        properties['tumor_max_length'] = max((box[0][1]-box[0][0]), (box[1][1]-box[1][0]), (box[2][1]-box[2][0]))
    write_pickle(properties, join(save_path, "images", case_identifier+".pkl"))

def window_transform(ct_array, windowWidth, windowCenter, normal=False):
    """
    return: trucated image according to window center and window width and normalized to [0,1]
    """
    minWindow = float(windowCenter) - 0.5*float(windowWidth)
    maxWindow = float(windowCenter) + 0.5*float(windowWidth)
    ct_array[ct_array < minWindow] = minWindow
    ct_array[ct_array > maxWindow] = maxWindow
    newing = (ct_array - minWindow) / float(windowWidth)
    if not normal:
        newing = (newing *255).astype('uint8')
    return newing

def split_dataset_list(divided_patient_sets_path, divided_ways, separate_stage=False, delete_patient=False ):
    sets = pd.read_excel(divided_patient_sets_path, sheet_name=divided_ways) # dataset dataframe
    trainList, valList, testList = [], [], []
    for i in range(len(sets)):
        # print(i, sets.loc[i]) # 10619760+20210817+CT
        sets.loc[i] = sets.loc[i].str.split('+')
        trainList.append(sets.loc[i]['train'][0])
        if i < sum(sets['val'].notna()):
            valList.append(sets.loc[i]['val'][0])
        if i < sum(sets['test'].notna()):
            testList.append(sets.loc[i]['test'][0])
    if separate_stage:
        print("仅纳入转移分期病人")
        stage_df = pd.read_excel("/data/hsx/project_pc_multimodality_addsize/Task_Pancreas/data_info/stage.xlsx",sheet_name='Sheet1')
        stage = stage_df[stage_df['stage'] == 2]
        id_values = stage['id'].tolist()
        ids = [str(num) for num in id_values]
        trainList = list(set(trainList) & set(ids))
        valList = list(set(valList) & set(ids))
        testList = list(set(testList) & set(ids))
    if delete_patient:
        print("去除部分病人")
        # delList = ['10433483', '10607973', '10609672', '10196015', '10607778', '10606753', '10612941', '10630517', '10674917',
        #            '10316003', '10334021', '10477579', '10698837', '10523116', '10422111', '10458166', '10526030', '10590703',
        #            '10598937', '10675323']
        delList = ['10433483', '10609672', '10196015', '10607778', '10674917',
                   '10316003', '10698837',
                   '10675323']
        trainList = [item for item in trainList if item not in delList]
        valList = [item for item in valList if item not in delList]
        testList = [item for item in testList if item not in delList]
    return trainList, valList, testList

def get_files_with_name_and_extension(folder_path, name_list, extension):
    file_paths = []
    file_pattern = os.path.join(folder_path, f"*.{extension}")
    files = glob.glob(file_pattern)
    for file_path in files: # 遍历文件列表，检查文件名是否包含在指定的名字列表中
        file_name = os.path.basename(file_path)
        for name in name_list:
            if name in file_name:
                file_paths.append(file_path)
    return file_paths

def is_tr_v_te(trainList, valList, testList, case_identifier):
    id = case_identifier[:8]
    if id in trainList:
        print(id, " in Train")
        return case_identifier[-1], 'train'
    elif id in valList:
        print(id, " in Val")
        return case_identifier[-1], 'val'
    elif id in testList:
        print(id, " in Test")
        return case_identifier[-1], 'test'
    else:
        print("ERROR! The ", id, " is not in the tr_v_te list")

def save_to_different_set(trainList, valList, testList, target_dir, source_dir, pkl_path):
    """
    :param setList: id
    :param tr_v_te: 用于标记当前List是训练还是验证还是测试
    :param target_dir: .../divided_set 函数内拼接/a v d/tr v te
    :param source_dir: 选择使用的待分类文件夹,文件夹内即是img与pkl的混合
    :param pkl_path: 数据pkl信息路径 得到该数据集的信息文件列表，然后读取，得到元组字典
                    sliceDictTrain[saved_filename] = (tumor_index, tumor_area, tumor_center, tumor_box, tumor_size)
    """
    sliceDictTrain, sliceDictVal, sliceDictTest = {}, {}, {}
    for filename in os.listdir(join(source_dir, "masks")): # 遍历filepath文件夹下所有子文件 filename = id+fup+p
        case_identifier = get_case_identifier_from_niigz(filename)
        property = load_pickle(join(pkl_path, case_identifier+'.pkl'))
        case_property = {"tumor_index":property["tumor_index"], "tumor_area":property["tumor_area"],
                         "tumor_box":property["tumor_box"], "tumor_center":property["tumor_center"],
                         "tumor_max_length":property["tumor_max_length"]}
        set, set_class = is_tr_v_te(trainList, valList, testList, case_identifier) # set is avd; set_class is tr_v_te
        shutil.copy(join(source_dir, "images", filename), join(target_dir, set+"_set", set_class)) # 复制这个文件到指定adv处
        if set_class == 'train': sliceDictTrain[filename] = case_property
        if set_class == 'val': sliceDictVal[filename] = case_property
        if set_class == 'test': sliceDictTest[filename] = case_property
    print("length of sets: ", len(sliceDictTrain), len(sliceDictVal), len(sliceDictTest)) # 1167 402 389
    write_pickle(sliceDictTrain, join(target_dir, "trainSliceDict.pkl"))
    write_pickle(sliceDictVal, join(target_dir, "valSliceDict.pkl"))
    write_pickle(sliceDictTest, join(target_dir, "testSliceDict.pkl"))

"""
*********************************************   tools   ****************************************************************
"""

def resize_segmentation(segmentation, new_shape, order=3):
    '''
    Resizes a segmentation map. Supports all orders (see skimage documentation). Will transform segmentation map to one
    hot encoding which is resized and transformed back to a segmentation map.
    This prevents interpolation artifacts ([0, 0, 2] -> [0, 1, 2])
    :param segmentation:
    :param new_shape:
    :param order:
    :return:
    '''
    tpe = segmentation.dtype
    unique_labels = np.unique(segmentation)
    assert len(segmentation.shape) == len(new_shape), "new shape must have same dimensionality as segmentation"
    if order == 0:
        return resize(segmentation.astype(float), new_shape, order, mode="edge", clip=True, anti_aliasing=False).astype(tpe)
    else:
        reshaped = np.zeros(new_shape, dtype=segmentation.dtype)
        for i, c in enumerate(unique_labels):
            mask = segmentation == c
            reshaped_multihot = resize(mask.astype(float), new_shape, order, mode="edge", clip=True, anti_aliasing=False)
            reshaped[reshaped_multihot >= 0.5] = c
        return reshaped

def determine_normalization_scheme(dataset_properties):
    schemes = OrderedDict()
    modalities = dataset_properties['modalities']
    num_modalities = len(list(modalities.keys()))
    for i in range(num_modalities):
        if modalities[i] == "CT" or modalities[i] == 'ct':
            schemes[i] = "CT"
        elif modalities[i] == 'noNorm':
            schemes[i] = "noNorm"
        else:
            schemes[i] = "nonCT"
    return schemes

def save_my_plans(file, plans):
    with open(file, 'wb') as f:
        pickle.dump(plans, f)

def load_pickle(file: str, mode: str = 'rb'):
    with open(file, mode) as f:
        a = pickle.load(f)
    return a

def write_pickle(obj, file: str, mode: str = 'wb') -> None:
    with open(file, mode) as f:
        pickle.dump(obj, f)

def txt2dict(file_path):
    dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.replace('\n', '')
            parts = re.split(r'[:\---]', line)
            print(parts[1])
            print(parts[-1])
            dict[parts[1]] = parts[-1]
            dict[parts[-1]] = parts[1]
    with open("./fewData/id2num.pkl", "wb") as tf:
        pickle.dump(dict,tf)

def load_case_from_list_of_files(case_identifier, data_files, seg_file=None):
    """
    读入nii.gz文件，保存原始数据信息
    :param data_files:
    :param seg_file:
    :return:
    """
    assert isinstance(data_files, list) or isinstance(data_files, tuple), "case must be either a list or a tuple"
    properties = OrderedDict()
    data_itk = [sitk.ReadImage(f) for f in data_files]

    properties["case_identifier"] = case_identifier
    properties["original_size_of_raw_data"] = np.array(data_itk[0].GetSize())[[2, 1, 0]]
    properties["original_spacing"] = np.array(data_itk[0].GetSpacing())[[2, 1, 0]]
    properties["list_of_data_files"] = data_files
    properties["seg_file"] = seg_file
    properties["itk_origin"] = data_itk[0].GetOrigin()
    properties["itk_spacing"] = data_itk[0].GetSpacing()
    properties["itk_direction"] = data_itk[0].GetDirection()

    data_npy = np.vstack([sitk.GetArrayFromImage(d)[None] for d in data_itk]) # 之前这里跑stack会报错 维度不一致 是因为是逐个病人处理的
    if seg_file is not None:
        seg_itk = sitk.ReadImage(seg_file)
        seg_npy = sitk.GetArrayFromImage(seg_itk)[None].astype(np.float32)
    else:
        seg_npy = None
    return data_npy.astype(np.float32), seg_npy, properties

def create_lists_from_splitted_dataset(base_folder_splitted):
    """
    从json文件中整理提取的img-mask匹配列表
    :param base_folder_splitted: 输入原始数据所在的大文件 dataset.json文件存在的地方
    :return: img-mask匹配列表
    """
    lists = []
    json_file = join(base_folder_splitted, "dataset.json")
    with open(json_file) as jsn:
        d = json.load(jsn)
        training_files = d['training']
    num_modalities = len(d['modality'].keys())
    for tr in training_files:
        cur_pat = []
        for mod in range(num_modalities): # /10036753_0_a.nii.gz
            cur_pat.append(join(base_folder_splitted, "images", tr['image'].split("/")[-1])) # [:-7]
            # + "_%04.0d.nii.gz" % mod)) # 10036753_0_a_0000.nii.gz是他的改名 这里暂时不需要
        cur_pat.append(join(base_folder_splitted, "masks", tr['label'].split("/")[-1]))
        lists.append(cur_pat)
    return lists, {int(i): d['modality'][str(i)] for i in d['modality'].keys()}

def get_case_identifier(case): # pancreas_0258_0000.nii.gz ---> 10036753_0_a_0000.nii.gz ---> 10036753_0_a.nii.gz
    case_identifier = case[0].split("/")[-1].split(".nii.gz")[0] #[:-5] # 不需要去除原先的模态部分
    return case_identifier

def get_patient_identifiers_from_cropped_files(folder):
    return [i.split("/")[-1][:-4] for i in subfiles(folder, join=True, suffix=".npz")]

def get_data_json_file(raw_data_file_path):
    # 定义数据集信息
    dataset_info = {
        "name": "Pancreas",
        "description": "cancer segmentation",
        "tensorImageSize": "3D",
        "modality": {
            "0": "CT"
        },
        "labels": {
            "0": "background",
            "1": "cancer"
        },
        "numTraining": 288,
        "numTest": 0,
        "training": [],
        "test": []
    }
    # 添加训练集样本
    for tr_file in os.listdir(raw_data_file_path+"/images"):
        sample = {
            "image": "./images/"+tr_file,
            "label": "./masks/"+tr_file
        }
        dataset_info["training"].append(sample)
    # for ts_file in os.listdir(raw_data_file_path+"imagesTr"):     # 添加测试集样本
    #     sample = "./imagesTs/"+ts_file
    #     dataset_info["test"].append(sample)
    with open(join(raw_data_file_path, "dataset.json"), "w") as f:     # 保存为JSON文件
        json.dump(dataset_info, f, indent=4)

"""
************************************************   Cropping   **********************************************************
"""

def create_nonzero_mask(data):
    """
    找一个阈值划分黑区域和组织 大于-800的划为组织区域 其余是0 黑区域
    :param data:
    :return:
    """
    from scipy.ndimage import binary_fill_holes
    nonzero_mask = np.zeros(data.shape[1:], dtype=bool) # 和数据一样的false mask
    for c in range(data.shape[0]): # 单模态中shape[0]就是1,取data[c]就是唯一的ct数据data[0](ZYX),但是多模态时就是依次取每一个模态
        this_mask = data[c] > -800 # 0 # 但CT原数据全黑无数据的地方是-2048而不是0 需要改动 !=0
        nonzero_mask = nonzero_mask | this_mask
    nonzero_mask = binary_fill_holes(nonzero_mask) # 防止就是中间实际值就是0但是却误以为是黑色背景的情形
    return nonzero_mask

def get_nonbed_mask_from_nonzero_mask(nonzero_mask):
    """
    这里使用腐蚀膨胀的方式去除机床 对分好了0 1的nonzero_mask做腐蚀膨胀
    :param nonzero_mask:
    :return:
    """
    ero_mask = skimage.morphology.erosion(nonzero_mask, skimage.morphology.ball(radius=8))
    nonbed_mask = skimage.morphology.dilation(ero_mask, skimage.morphology.ball(radius=7))
    return nonbed_mask

def get_bbox_from_mask(mask, outside_value=0):
    """
    边界，寻找mask中xyz轴维度里不为0的范围
    :param mask:
    :param outside_value:
    :return:
    """
    mask_voxel_coords = np.where(mask != outside_value) # 已经划定范围 外界值-2048的全黑区域是0
    minzidx = int(np.min(mask_voxel_coords[0]))
    maxzidx = int(np.max(mask_voxel_coords[0])) + 1
    minxidx = int(np.min(mask_voxel_coords[1]))
    maxxidx = int(np.max(mask_voxel_coords[1])) + 1
    minyidx = int(np.min(mask_voxel_coords[2]))
    maxyidx = int(np.max(mask_voxel_coords[2])) + 1
    return [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]

def crop_to_bbox(image, bbox):
    assert len(image.shape) == 3, "only supports 3d images"
    resizer = (slice(bbox[0][0], bbox[0][1]), slice(bbox[1][0], bbox[1][1]), slice(bbox[2][0], bbox[2][1]))
    return image[resizer]

def crop_to_nonzero(data, seg=None, nonzero_label=-1):
    """
    crop的流程函数-裁去大范围无用黑边，留下器官组织区域，并将余下非标注区域中的黑背景设置为-1
    :param data: 是单独的[c z y x]ct npz seg同理
    :param nonzero_label: this will be written into the segmentation map
    """
    nonzero_mask = create_nonzero_mask(data) # 经由此，可以获得含有机床和组织连接的 0 1 mask
    nonbed_mask = get_nonbed_mask_from_nonzero_mask(nonzero_mask)
    for c in range(data.shape[0]):
        data[c][nonbed_mask != 1] = -1024 # 把去床的mask盖上ct 由于显示灰度的特殊性，全黑并不是0 而是-1024为保险灰度值
    # 将ct 原标注mask nonzero_mask均进行裁剪
    bbox = get_bbox_from_mask(nonbed_mask, 0) # 找mask中不等于0的区域 即使组织区域
    cropped_data, cropped_seg = [], []
    for c in range(data.shape[0]):
        cropped = crop_to_bbox(data[c], bbox)
        cropped_data.append(cropped[None])
    cropped_data = np.vstack(cropped_data)
    nonzero_mask = crop_to_bbox(nonbed_mask, bbox)
    if seg is not None:
        for c in range(seg.shape[0]):
            cropped = crop_to_bbox(seg[c], bbox)
            cropped_seg.append(cropped[None])
        cropped_seg = np.vstack(cropped_seg)
        # 设置label中-1的黑域
        cropped_seg[(cropped_seg == 0) & (nonzero_mask == 0)] = nonzero_label # 最终实现-1(黑景) 0(组织) 1(标注)
    else:
        cropped_seg = nonzero_mask
    return cropped_data, cropped_seg, bbox

def do_case_crop(data, seg, properties):
    """
    打印crop前后信息，调用crop_to_nonzero，修正crop结果，补充properties信息
    :return:
    """
    shape_before = data.shape
    data, seg, bbox = crop_to_nonzero(data, seg, nonzero_label=-1)
    shape_after = data.shape
    print(properties["case_identifier"], "before crop:", shape_before, "after crop:", shape_after, "spacing:",
          np.array(properties["original_spacing"]), "\n")
    properties["crop_bbox"] = bbox
    properties['classes'] = np.unique(seg)
    seg[seg < -1] = 0 # 但是seg中不是没有小于-1的区域吗 意在何为？---单纯修正作用
    properties["size_after_cropping"] = data[0].shape
    return data, seg, properties

def load_crop_save(case, case_identifier, overwrite_existing=False):
    """
    crop的传入的是单个case 对应一张CT(所有模态)和对应的mask，case应该是同一个序号标识，比如一个人的 这个函数是多进程调用的
    """
    output_folder = cropped_data
    print(case_identifier)
    if overwrite_existing \
            or (not os.path.isfile(join(output_folder, "%s.npz" % case_identifier))
                or not os.path.isfile(os.path.join(output_folder, "%s.pkl" % case_identifier))):

        data, seg, properties = load_case_from_list_of_files(case_identifier, case[:-1], case[-1]) # 此处注意case[:-1]是取的除了标注的其他CT文件，如果有多个模态，则case[:-1]并不等价于case[0]
        data, seg, properties = do_case_crop(data, seg, properties)
        all_data = np.vstack((data, seg))
        np.savez_compressed(join(output_folder, "%s.npz" % case_identifier), data=all_data)
        with open(join(output_folder, "%s.pkl" % case_identifier), 'wb') as f:
            pickle.dump(properties, f)

"""
***********************************************   Resampling   *********************************************************
"""

def get_target_spacings(spacings, sizes):
    """
    per default we use the 50th percentile=median for the target spacing. Higher spacing results in smaller data
    and thus faster and easier training. Smaller spacing results in larger data and thus longer and harder training

    For some datasets the median is not a good choice. Those are the datasets where the spacing is very anisotropic
    (for example ACDC with (10, 1.5, 1.5)). These datasets still have examples with a spacing of 5 or 6 mm in the low
    resolution axis. Choosing the median here will result in bad interpolation artifacts that can substantially
    impact performance (due to the low number of slices).
    """
    target = np.percentile(np.vstack(spacings), 50, 0)
    # This should be used to determine the new median shape. The old implementation is not 100% correct.
    # Fixed in 2.4
    # sizes = [np.array(i) / target * np.array(j) for i, j in zip(spacings, sizes)]
    target_size = np.percentile(np.vstack(sizes), 50, 0)
    target_size_mm = np.array(target) * np.array(target_size)
    # we need to identify datasets for which a different target spacing could be beneficial. These datasets have
    # the following properties:
    # - one axis which much lower resolution than the others
    # - the lowres axis has much less voxels than the others
    # - (the size in mm of the lowres axis is also reduced)
    worst_spacing_axis = np.argmax(target)
    other_axes = [i for i in range(len(target)) if i != worst_spacing_axis]
    other_spacings = [target[i] for i in other_axes]
    other_sizes = [target_size[i] for i in other_axes]

    has_aniso_spacing = target[worst_spacing_axis] > (3 * max(other_spacings))
    has_aniso_voxels = target_size[worst_spacing_axis] * 3 < min(other_sizes)
    # we don't use the last one for now
    #median_size_in_mm = target[target_size_mm] * RESAMPLING_SEPARATE_Z_ANISOTROPY_THRESHOLD < max(target_size_mm)

    if has_aniso_spacing and has_aniso_voxels:
        spacings_of_that_axis = np.vstack(spacings)[:, worst_spacing_axis]
        target_spacing_of_that_axis = np.percentile(spacings_of_that_axis, 10)
        # don't let the spacing of that axis get higher than the other axes
        if target_spacing_of_that_axis < max(other_spacings):
            target_spacing_of_that_axis = max(max(other_spacings), target_spacing_of_that_axis) + 1e-5
        target[worst_spacing_axis] = target_spacing_of_that_axis
    return target

def load_cropped(cropped_output_dir, case_identifier):
    all_data = np.load(os.path.join(cropped_output_dir, "%s.npz" % case_identifier))['data']
    data = all_data[:-1].astype(np.float32)
    seg = all_data[-1:]
    with open(os.path.join(cropped_output_dir, "%s.pkl" % case_identifier), 'rb') as f:
        properties = pickle.load(f)
    return data, seg, properties

def resample_data_or_seg(data, new_shape, is_seg, axis=None, order=3, do_separate_z=False, order_z=0):
    """
    separate_z=True will resample with order 0 along z 使用最临近插值 也许默认时双线性插值等
    :param data:
    :param new_shape:
    :param is_seg:
    :param axis:
    :param order:
    :param do_separate_z:
    :param order_z: only applies if do_separate_z is True
    :return:
    """
    assert len(data.shape) == 4, "data must be (c, x, y, z)"
    assert len(new_shape) == len(data.shape) - 1
    if is_seg:
        resize_fn = resize_segmentation
        kwargs = OrderedDict()
    else:
        resize_fn = resize
        kwargs = {'mode': 'edge', 'anti_aliasing': False} # 影响重采样过程中的像素值计算和边缘处理
        # mode edge表示使用边缘像素进行填充。这意味着将超过原始图像边界的像素值重复到边缘上，以处理边缘情况
        # anti_aliasing：该参数控制是否在重采样过程中应用抗锯齿滤波。默认值为 False，表示不进行抗锯齿处理
    dtype_data = data.dtype
    shape = np.array(data[0].shape)
    new_shape = np.array(new_shape)
    if np.any(shape != new_shape):
        data = data.astype(float)
        if do_separate_z:
            print("separate z, order in z is", order_z, "order inplane is", order)
            assert len(axis) == 1, "only one anisotropic axis supported"
            axis = axis[0]
            if axis == 0:
                new_shape_2d = new_shape[1:]
            elif axis == 1:
                new_shape_2d = new_shape[[0, 2]]
            else:
                new_shape_2d = new_shape[:-1]

            reshaped_final_data = []
            for c in range(data.shape[0]):
                reshaped_data = []
                for slice_id in range(shape[axis]):
                    if axis == 0:
                        reshaped_data.append(resize_fn(data[c, slice_id], new_shape_2d, order, **kwargs).astype(dtype_data))
                    elif axis == 1:
                        reshaped_data.append(resize_fn(data[c, :, slice_id], new_shape_2d, order, **kwargs).astype(dtype_data))
                    else:
                        reshaped_data.append(resize_fn(data[c, :, :, slice_id], new_shape_2d, order, **kwargs).astype(dtype_data))
                reshaped_data = np.stack(reshaped_data, axis)
                if shape[axis] != new_shape[axis]:

                    # The following few lines are blatantly copied and modified from sklearn's resize()
                    rows, cols, dim = new_shape[0], new_shape[1], new_shape[2]
                    orig_rows, orig_cols, orig_dim = reshaped_data.shape

                    row_scale = float(orig_rows) / rows
                    col_scale = float(orig_cols) / cols
                    dim_scale = float(orig_dim) / dim

                    map_rows, map_cols, map_dims = np.mgrid[:rows, :cols, :dim]
                    map_rows = row_scale * (map_rows + 0.5) - 0.5
                    map_cols = col_scale * (map_cols + 0.5) - 0.5
                    map_dims = dim_scale * (map_dims + 0.5) - 0.5

                    coord_map = np.array([map_rows, map_cols, map_dims])
                    if not is_seg or order_z == 0:
                        reshaped_final_data.append(map_coordinates(reshaped_data, coord_map, order=order_z,
                                                                   mode='nearest')[None].astype(dtype_data))
                    else:
                        unique_labels = np.unique(reshaped_data)
                        reshaped = np.zeros(new_shape, dtype=dtype_data)

                        for i, cl in enumerate(unique_labels):
                            reshaped_multihot = np.round(
                                map_coordinates((reshaped_data == cl).astype(float), coord_map, order=order_z,
                                                mode='nearest'))
                            reshaped[reshaped_multihot > 0.5] = cl
                        reshaped_final_data.append(reshaped[None].astype(dtype_data))
                else:
                    reshaped_final_data.append(reshaped_data[None].astype(dtype_data))
            reshaped_final_data = np.vstack(reshaped_final_data)
        else:
            print("no separate z, order", order)
            reshaped = []
            for c in range(data.shape[0]):
                reshaped.append(resize_fn(data[c], new_shape, order, **kwargs)[None].astype(dtype_data))
            reshaped_final_data = np.vstack(reshaped)
        return reshaped_final_data.astype(dtype_data)
    else:
        print("no resampling necessary")
        return data

def get_do_separate_z(spacing, anisotropy_threshold=3):
    """
    spacing的最大值和最小值的比值，大于3表明z轴间距与其他轴间距差异大，这种差异会导致重采样（插值）引入伸缩变形，所以z轴独立进行重采样
    :param spacing:
    :param anisotropy_threshold: 设定的判别阈值
    """
    do_separate_z = (np.max(spacing) / np.min(spacing)) > anisotropy_threshold
    return do_separate_z

def get_lowres_axis(new_spacing): # array([5.        , 0.76599997, 0.76599997])
    axis = np.where(max(new_spacing) / np.array(new_spacing) == 1)[0]  # find which axis is anisotropic array([0])
    return axis

def resample_patient(data, seg, original_spacing, target_spacing, order_data=3, order_seg=0, force_separate_z=False,
                     order_z_data=0, order_z_seg=0, separate_z_anisotropy_threshold=3):
    """
    :param order_data:
    :param order_seg:
    :param force_separate_z: if None then we dynamically decide how to resample along z, if True/False then always
    /never resample along z separately
    :param order_z_seg: only applies if do_separate_z is True
    :param order_z_data: only applies if do_separate_z is True
    :param separate_z_anisotropy_threshold: if max_spacing > separate_z_anisotropy_threshold * min_spacing (per axis)
    then resample along lowres axis with order_z_data/order_z_seg instead of order_data/order_seg
    :return:
    """
    if data is not None:
        shape = np.array(data[0].shape)
    else:
        shape = np.array(seg[0].shape)
    new_shape = np.round(((np.array(original_spacing) / np.array(target_spacing)).astype(float) * shape)).astype(int)

    if force_separate_z is not None:
        do_separate_z = force_separate_z
        if force_separate_z:
            axis = get_lowres_axis(original_spacing)
        else:
            axis = None
    else:
        if get_do_separate_z(original_spacing, separate_z_anisotropy_threshold): # 通常是走这条分支
            do_separate_z = True
            axis = get_lowres_axis(original_spacing)
        elif get_do_separate_z(target_spacing, separate_z_anisotropy_threshold):
            do_separate_z = True
            axis = get_lowres_axis(target_spacing)
        else:
            do_separate_z = False
            axis = None

    if axis is not None:
        if len(axis) == 3:
            # every axis has the spacing, this should never happen, why is this code here?
            do_separate_z = False
        elif len(axis) == 2:
            # this happens for spacings like (0.24, 1.25, 1.25) for example. In that case we do not want to resample
            # separately in the out of plane axis
            do_separate_z = False
        else:
            pass

    if data is not None:
        data_reshaped = resample_data_or_seg(data, new_shape, False, axis, order_data, do_separate_z,
                                             order_z=order_z_data)
    else:
        data_reshaped = None
    if seg is not None:
        seg_reshaped = resample_data_or_seg(seg, new_shape, True, axis, order_seg, do_separate_z, order_z=order_z_seg)
    else:
        seg_reshaped = None
    return data_reshaped, seg_reshaped

def resample_and_normalize(case_identifier, data, seg, target_spacing, properties, transpose_forward, force_separate_z=None):
    """
    data and seg must already have been transposed by transpose_forward. properties are the un-transposed values
    (spacing etc)
    :param data:
    :param target_spacing: target_spacing is already transposed, properties["original_spacing"] is not so we need to transpose it!
    :param properties:
    :param seg:
    :param force_separate_z:
    :return:
    """
    # data, seg are already transposed. Double check this using the properties
    plans = load_pickle(join(preprocessing_output_dir, "preprocess_plans_3D.pkl")) # 这是保存的plans 预处理关键信息
    normalization_schemes = plans['normalization_schemes']
    use_nonzero_mask_for_normalization = plans['use_mask_for_norm']
    intensityproperties = plans['dataset_properties']['intensityproperties']

    original_spacing_transposed = np.array(properties["original_spacing"])[transpose_forward]
    before = {
        'spacing': properties["original_spacing"],
        'spacing_transposed': original_spacing_transposed,
        'data.shape (data is transposed)': data.shape
    }
    data[np.isnan(data)] = 0     # remove nans
    data, seg = resample_patient(data, seg, np.array(original_spacing_transposed), target_spacing, 3, 1, # 3 1是默认设置
                                 force_separate_z=force_separate_z, order_z_data=0, order_z_seg=0,
                                 separate_z_anisotropy_threshold=3)
    after = {
        'spacing': target_spacing,
        'data.shape (data is resampled)': data.shape
    }
    print("before:", before, "\nafter: ", after, "\n")
    np.savez_compressed(os.path.join(preprocessing_output_dir, "preprocess_plans_3D", "stage_resample", "%s.npz"
                                     % case_identifier), data=data.astype(np.float32))

    properties["size_after_resampling"] = data[0].shape
    properties["spacing_after_resampling"] = target_spacing
    use_nonzero_mask = use_nonzero_mask_for_normalization
    assert len(normalization_schemes) == len(data), "self.normalization_scheme_per_modality " \
                                                    "must have as many entries as data has " \
                                                    "modalities"
    assert len(use_nonzero_mask_for_normalization) == len(data), "self.use_nonzero_mask must have as many entries as data" \
                                                                 " has modalities"
    for c in range(len(data)):
        scheme = normalization_schemes[c]
        if scheme == "CT":
            # clip to lb and ub from train data foreground and use foreground mn and sd from training data
            assert intensityproperties is not None, "ERROR: if there is a CT then we need intensity properties"
            mean_intensity = intensityproperties[c]['mean']
            std_intensity = intensityproperties[c]['sd']
            lower_bound = intensityproperties[c]['percentile_00_5']
            upper_bound = intensityproperties[c]['percentile_99_5']
            data[c] = np.clip(data[c], lower_bound, upper_bound)
            data[c] = (data[c] - mean_intensity) / std_intensity
            if use_nonzero_mask[c]:
                data[c][seg[-1] < 0] = 0
        elif scheme == "CT2":
            # clip to lb and ub from train data foreground, use mn and sd form each case for normalization
            assert intensityproperties is not None, "ERROR: if there is a CT then we need intensity properties"
            lower_bound = intensityproperties[c]['percentile_00_5']
            upper_bound = intensityproperties[c]['percentile_99_5']
            mask = (data[c] > lower_bound) & (data[c] < upper_bound)
            data[c] = np.clip(data[c], lower_bound, upper_bound)
            mn = data[c][mask].mean()
            sd = data[c][mask].std()
            data[c] = (data[c] - mn) / sd
            if use_nonzero_mask[c]:
                data[c][seg[-1] < 0] = 0
        elif scheme == 'noNorm':
            print('no intensity normalization')
            pass
        else:
            if use_nonzero_mask[c]:
                mask = seg[-1] >= 0
                data[c][mask] = (data[c][mask] - data[c][mask].mean()) / (data[c][mask].std() + 1e-8)
                data[c][mask == 0] = 0
            else:
                mn = data[c].mean()
                std = data[c].std()
                # print(data[c].shape, data[c].dtype, mn, std)
                data[c] = (data[c] - mn) / (std + 1e-8)
    return data, seg, properties

def do_case_resample_normalize(target_spacing, case_identifier, output_folder_stage, cropped_output_dir, transpose_forward,
                               force_separate_z, all_classes):
    """
    这个函数完成经过线程函数后单个文件的导入与执行以及该阶段的数据保存
    :param target_spacing:
    :param case_identifier:
    :param output_folder_stage: nnUNetData_plans_v2.1_stage0
    :param cropped_output_dir: preprocessed_out_dir
    :param force_separate_z: 默认传进来的是None,以便自适应
    :param all_classes: 1
    :return:
    """
    data, seg, properties = load_cropped(cropped_output_dir, case_identifier)
    data = data.transpose((0, *[i + 1 for i in transpose_forward]))
    seg = seg.transpose((0, *[i + 1 for i in transpose_forward])) # [0,1,2]
    data, seg, properties = resample_and_normalize(case_identifier, data, seg, target_spacing, properties, transpose_forward, force_separate_z)
    all_data = np.vstack((data, seg)).astype(np.float32)
    num_samples = 10000
    min_percent_coverage = 0.01 # at least 1% of the class voxels need to be selected, otherwise it may be too sparse
    rndst = np.random.RandomState(1234)
    class_locs = {}
    for c in all_classes:
        all_locs = np.argwhere(all_data[-1] == c)
        if len(all_locs) == 0:
            class_locs[c] = []
            continue
        target_num_samples = min(num_samples, len(all_locs))
        target_num_samples = max(target_num_samples, int(np.ceil(len(all_locs) * min_percent_coverage)))
        selected = all_locs[rndst.choice(len(all_locs), target_num_samples, replace=False)]
        class_locs[c] = selected
        print(c, target_num_samples)
    properties['class_locations'] = class_locs
    print("saving: ", os.path.join(output_folder_stage, "%s.npz" % case_identifier))
    np.savez_compressed(os.path.join(output_folder_stage, "%s.npz" % case_identifier),
                        data=all_data.astype(np.float32))
    with open(os.path.join(output_folder_stage, "%s.pkl" % case_identifier), 'wb') as f:
        pickle.dump(properties, f)

def plan_resample(cropped_data_path):
    use_nonzero_mask_for_normalization = {0:False} # 函数中如果识别模态是CT就是 false
    list_of_cropped_npz_files = subfiles(cropped_data_path, True, None, ".npz", True)
    dataset_properties = load_pickle(join(cropped_data_path, "dataset_properties.pkl"))
    spacings = dataset_properties['all_spacings']
    sizes = dataset_properties['all_sizes']
    all_classes = dataset_properties['all_classes']
    modalities = dataset_properties['modalities']
    num_modalities = len(list(modalities.keys()))
    # 此处给出中值体素值 并计算转化后对应shape
    target_spacing = get_target_spacings(spacings, sizes)
    new_shapes = [np.array(i) / target_spacing * np.array(j) for i, j in zip(spacings, sizes)]
    # 输出数据集中值信息
    median_shape = np.median(np.vstack(new_shapes), 0)
    print("the median shape of the dataset is ", median_shape)
    max_shape = np.max(np.vstack(new_shapes), 0)
    print("the max shape in the dataset is ", max_shape)
    min_shape = np.min(np.vstack(new_shapes), 0)
    print("the min shape in the dataset is ", min_shape)

    transpose_forward = [0, 1, 2] # [max_spacing_axis] + remaining_axes
    target_spacing_transposed = np.array(target_spacing)[transpose_forward]
    median_shape_transposed = np.array(median_shape)[transpose_forward]
    print("the transposed median shape of the dataset is ", median_shape_transposed)

    normalization_schemes = determine_normalization_scheme(dataset_properties)
    # 这只针对预处理计划
    plans = {'num_modalities': num_modalities, 'modalities': modalities, 'normalization_schemes': normalization_schemes,
             'dataset_properties': dataset_properties, 'list_of_npz_files': list_of_cropped_npz_files,
             'original_spacings': spacings, 'original_sizes': sizes,
             'target_spacings': target_spacing_transposed,
             'preprocessed_data_folder': preprocessing_output_dir, 'num_classes': len(all_classes),
             'all_classes': all_classes, 'use_mask_for_norm': use_nonzero_mask_for_normalization,
             'transpose_forward': transpose_forward, 'data_identifier': 'preprocess_plans_3D'
             }
    save_my_plans(join(preprocessing_output_dir, 'preprocess_plans_3D.pkl'), plans)