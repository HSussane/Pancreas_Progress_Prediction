from os.path import join
import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np
import pickle
import pandas as pd
from batchgenerators.utilities.file_and_folder_operations import subfiles
# from dataProcess.utils import load_nii

#%% 计算 人口统计学和临床特征 中位数和P值
file_path = r"E:\pycharm-code\A_HSX\project_pc_multimodality_addsize\Task_Pancreas\data_info"

# df_treatment = pd.read_excel(join(file_path, "treatment.xlsx")).set_index('id')
df_clinic = pd.read_excel(join(file_path, "pancreas_clinical_info_cal.xlsx"), sheet_name='info')
tr_df = pd.read_excel(join(file_path, "pancreas_clinical_info_cal.xlsx"), sheet_name='tr')
in_df = pd.read_excel(join(file_path, "pancreas_clinical_info_cal.xlsx"), sheet_name='in')
ex_df = pd.read_excel(join(file_path, "pancreas_clinical_info_cal.xlsx"), sheet_name='ex')

# df_clinic['sex'] = df_clinic['sex'].replace({'男': 1, '女': 2})

#%% 计算中位数和范围
# 选取不同队列的ID出来 把他们分成三个队列分别保存

tr_df['id'] = tr_df['id'].apply(lambda x: x.split('+')[0] if isinstance(x, str) and '+' in x else x)

#%%
with pd.ExcelWriter(join(file_path, "pancreas_clinical_info_cohort.xlsx")) as writer:
    # df_clinic.to_excel(writer, sheet_name='info')
    # tr_df.to_excel(writer, sheet_name='tr')
    # in_df.to_excel(writer, sheet_name='in')
    # ex_df.to_excel(writer, sheet_name='ex')
    df_clinic_tr.to_excel(writer, sheet_name='tr')
    df_clinic_in.to_excel(writer, sheet_name='in')
    df_clinic_ex.to_excel(writer, sheet_name='ex')


#%%
df_clinic_tr = df_clinic[df_clinic['ID'].isin(tr_df['id'].astype(int))]
df_clinic_in = df_clinic[df_clinic['ID'].isin(in_df['id'].astype(int))]
df_clinic_ex = df_clinic[df_clinic['ID'].isin(ex_df['id'].astype(int))]

#%%
# 计算每列的数值范围（最大值 - 最小值）

df = df_clinic_ex

range_max = df.apply(lambda x: x.max() if pd.api.types.is_numeric_dtype(x) else None)
range_min = df.apply(lambda x: x.min() if pd.api.types.is_numeric_dtype(x) else None)
median_values = df.apply(lambda x: x.median() if pd.api.types.is_numeric_dtype(x) else None)

# 创建一个包含统计结果的 DataFrame
stats_df = pd.DataFrame({
    '数值范围MAX': range_max,
    '数值范围MIN': range_min,
    '中位数': median_values
})



#%%
def load_nii(file_path, case_identifier):
    image = sitk.ReadImage(join(file_path, "images", "%s.nii.gz" % case_identifier))
    mask = sitk.ReadImage(join(file_path, "masks", "%s.nii.gz" % case_identifier))
    return image, mask

def load_pkl(file_path, case_identifier, mode: str = 'rb'):
    with open(join(file_path, case_identifier), mode) as f:
        a = pickle.load(f)
    return a

#%% 原image
raw_path = '/data/hsx/project_pc_multimodality_addsize/Task_Pancreas/raw_data/images'
name = '10598754_0_a' #'10322248_0_a.nii.gz'
raw_data = sitk.ReadImage(join(raw_path, name+'.nii.gz'))
raw_npz = sitk.GetArrayFromImage(raw_data)
plt.imshow(raw_npz[45,:,:], cmap='gray')
plt.show()

#%% 裁剪
cropped_path = '/data/hsx/project_pc_multimodality_addsize/Task_Pancreas/cropped_data'
name = '10598754_0_a'
data = np.load(join(cropped_path, name+'.npz'))
res = data['data'][0]
seg = data['data'][1]
plt.imshow(res[45,:,:], cmap='gray')
plt.show()

#%% 重采样&标准化 np.where(np.any(seg == 1, axis=(1, 2)))
processed_path = '/data/hsx/project_pc_multimodality_addsize/Task_Pancreas/preprocessed/preprocess_plans_3D'
name = '10036753_0_a'
data = np.load(join(processed_path, name+'.npz'))
res = data['data'][0]
seg = data['data'][1]
plt.imshow(res[20,:,:], cmap='gray')
plt.show()
#%%
lb_data = sitk.ReadImage(join(cropped_path, 'gt_segmentations', name+'.nii.gz'))
lb_npz = sitk.GetArrayFromImage(lb_data)
plt.imshow(lb_npz[46,:,:], cmap='gray')
plt.show()

#%% 哪些重采样改变了z轴维度
list_of_pkl_files = subfiles(processed_path, True, None, ".pkl", True)
cnt = 0
for pkl_name in list_of_pkl_files:
    with open(join(processed_path, pkl_name), 'rb') as f:
        pkl = pickle.load(f)
        if abs(float(pkl['original_spacing'][0])-float(pkl['spacing_after_resampling'][0])) >= 0.1:
            print(pkl['case_identifier'], pkl['original_spacing'], pkl['original_size_of_raw_data'], pkl['size_after_resampling'])
            cnt += 1

#%% pkl 内容
resample_xy_path = '/data/hsx/project_pc_multimodality_addsize/Task_Pancreas/preprocessed/preprocess_plans_2D/stage_resample_xy'
with open(join(resample_xy_path, "images", '10660298_0_a.pkl'), 'rb') as f:
    resampled_pkl = pickle.load(f)

#%%
def ImageResample(sitk_image, new_spacing, is_label=False):
    '''
    sitk_image:
    new_spacing: x,y,z
    is_label: if True, using Interpolator `sitk.sitkNearestNeighbor`
    '''
    size = np.array(sitk_image.GetSize())
    spacing = np.array(sitk_image.GetSpacing())
    new_spacing[2] = spacing[2]
    new_spacing = np.array(new_spacing)
    new_size = size * spacing / new_spacing
    new_spacing_refine = size * spacing / new_size
    new_spacing_refine = [float(s) for s in new_spacing_refine]
    new_size = [int(s) for s in new_size]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputDirection(sitk_image.GetDirection())
    resample.SetOutputOrigin(sitk_image.GetOrigin())
    resample.SetSize(new_size)
    resample.SetOutputSpacing(new_spacing_refine)

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        #resample.SetInterpolator(sitk.sitkBSpline)
        resample.SetInterpolator(sitk.sitkLinear)
    newimage = resample.Execute(sitk_image)
    return newimage

#%%
raw_data = '/data/hsx/project_pc_multimodality_addsize/Task_Pancreas/raw_data'
sitk_image, sitk_mask = load_nii(raw_data, '10656805_0_a')
newimage = ImageResample(sitk_image, [0.7421875, 0.7421875, 0], False)
newmask = ImageResample(sitk_mask, [0.7421875, 0.7421875, 0], True)
save_path = '/data/hsx/project_pc_multimodality_addsize/Task_Pancreas/preprocessed/test_res_data'
sitk.WriteImage(newmask, join(save_path, "10656805_0_a"+".nii.gz"))
#%%
pkl_path = '/data/hsx/project_pc_multimodality_addsize/Task_Pancreas/preprocessed/preprocess_plans_2D/stage_resample_xy'
window_path = '/data/hsx/project_pc_multimodality_addsize/Task_Pancreas/preprocessed/preprocess_plans_2D/stage_window'
case_identifier = '10036753_0_a'
resam_image, resam_mask = load_nii(window_path, case_identifier)
pkl = load_pkl(join(pkl_path, "images"), case_identifier+'.pkl')
mk_npz = sitk.GetArrayFromImage(resam_mask)
ct_npz = sitk.GetArrayFromImage(resam_image)
index = np.where(np.any(mk_npz == 1, axis=(1, 2)))[0]
#%%
plt.imshow(mk_npz[10,:,:], cmap='gray')
plt.show()


#%%
# 这里计算一下最大的边长是多少
pkl_file_pattern = os.path.join(resampled_data_2D, "images", f"*.{'pkl'}")
pkl_file_paths = glob.glob(pkl_file_pattern)
cnt_max150, cnt_max100, cnt_max50 = 0, 0, 0
for file_path in pkl_file_paths:
    pkl = load_pickle(file_path)
    ct_max_length = pkl["tumor_max_length"]
    if 50 <= ct_max_length < 150: # 这里的最小是 一个肿瘤最大横截面的最小
        cnt_max100 += 1
    if ct_max_length < 50:
        cnt_max50 += 1
print(cnt_max150, cnt_max100, cnt_max50)
# maxLenth[146 15] cnt_max150=42, cnt_max100=937, cnt_max50=979 也就是说50为分界 各自占一半

#%% 计算0 1在训练集验证集中slice数目
lb_path = "/data/hsx/project_pc_multimodality_addsize/result/transTrain300"
tr_lb_path = join(lb_path, "y_epoch_tr1.csv")
val_lb_path = join(lb_path, "y_epoch1.csv")
tr_lb = pd.read_csv(tr_lb_path, header=None)
val_lb = pd.read_csv(val_lb_path, header=None)
#%%
val_lb_arr = val_lb[0].values
count_0 = np.count_nonzero(val_lb_arr == 0) # 1116
count_1 = np.count_nonzero(val_lb_arr == 1) # 328

#%%
# 读取 CSV 文件
df = pd.read_csv('E:\pycharm-code\A_HSX\project_pc_multimodality_addsize\Task_Pancreas\data_info\labels_CT.csv')
df_follows = df.iloc[:, -8:] # 243 全部病人的随访信息了
# 然后统计非NULL的格子数 创新新列 记录为该病人所拥有的随访次数
df['ct_nums'] = df_follows.count(axis=1)
df['ct_nums'] = df['ct_nums'].add(1)
#%%
print(df.head()) # 前几行
df_sort = df.sort_values('ct_nums', ascending=True)
print(df_sort.head())
df_sort = df_sort.drop(df_sort.columns[1:5], axis=1)
print(df_sort.head())
#%%
def split_dataset_list(divided_patient_sets_path, divided_ways):
    sets = pd.read_excel(divided_patient_sets_path, sheet_name=divided_ways) # dataset dataframe
    trainList, valList, testList = [], [], []
    for i in range(len(sets)):
        print(i, sets.loc[i]) # 10619760+20210817+CT
        sets.loc[i] = sets.loc[i].str.split('+')
        trainList.append(sets.loc[i]['train'][0])
        if i < sum(sets['val'].notna()):
            valList.append(sets.loc[i]['val'][0])
        if i < sum(sets['test'].notna()):
            testList.append(sets.loc[i]['test'][0])
    return trainList, valList, testList
trainList, valList, testList = split_dataset_list("E:\pycharm-code\A_HSX\project_pc_multimodality_addsize\Task_Pancreas\data_info\divided_patient_sets.xlsx", "622")
#%%
df_sort['id'] = df_sort['id'].astype(str)
train_df = df_sort[df_sort['id'].isin(trainList)]
val_df = df_sort[df_sort['id'].isin(valList)]
test_df = df_sort[df_sort['id'].isin(testList)]
#%%
test_df.to_csv("E:/pycharm-code/A_HSX/project_pc_multimodality_addsize/Task_Pancreas/data_info/test_df.csv", index=True)

#%% 这里对2D分割任务进行实验与检查
# 检查cropped_data的匹配性
cropped_path = '/data/hsx/project_pc_multimodality_addsize/Task_Pancreas/cropped_data'
seg_path = join(cropped_path, 'gt_segmentations')
case_identifier = '10036753_0_a'
seg = sitk.ReadImage(join(seg_path, "%s.nii.gz" % case_identifier))
ori_seg_npz = sitk.GetArrayFromImage(seg)
pkl = load_pkl(cropped_path, case_identifier+".pkl")
img = np.load(join(cropped_path, "%s.npz" % case_identifier))
img_npz = img['data'][0]
new_seg_npz = img['data'][1]
#%% 对npz文件进行X Y维度重采样实验
# 计算新shape
original_spacing = pkl['original_spacing']
target_spacing = [original_spacing[0], 0.7421875, 0.7421875] # 0表示在z维度不进行重采样
shape = np.array(img_npz.shape)
new_shape = np.round(((np.array(original_spacing) / np.array(target_spacing)).astype(float) * shape)).astype(int)
#%%
from skimage.transform import resize
from collections import OrderedDict

# 预留seg_resize函数
def resize_segmentation(segmentation, new_shape, order=0):
    '''
    Resizes a segmentation map. Supports all orders (see skimage documentation). Will transform segmentation map to one
    hot encoding which is resized and transformed back to a segmentation map.
    This prevents interpolation artifacts ([0, 0, 2] -> [0, 1, 2])
    :param segmentation:
    :param new_shape:
    :param order: 0 表示最邻近插值
    :return:
    '''
    tpe = segmentation.dtype
    return resize(segmentation.astype(float), new_shape, order, mode="edge", clip=True, anti_aliasing=False).astype(tpe)

resize_fn = resize_segmentation # resize
kwargs = OrderedDict() # {'mode': 'edge', 'anti_aliasing': False}
order = 0 # data用3 三次样条插值 seg用0最邻近插值
dtype_data = new_seg_npz.dtype # 考虑到data和seg的数据类型不一样
#%% 正式实验
reshaped_seg = []
for slice_id in range(shape[0]): # slice数目 单张slice做归一化
    reshaped_seg.append(resize_fn(new_seg_npz[slice_id], new_shape[1:], order, **kwargs).astype(dtype_data))
reshaped_seg = np.stack(reshaped_seg, 0)
#%% 对比和nii格式重采样的结果
resample_xy_path = '/data/hsx/project_pc_multimodality_addsize/Task_Pancreas/preprocessed/preprocess_plans_2D/stage_resample_xy'
samp = sitk.ReadImage(join(resample_xy_path, "images", "%s.nii.gz" % case_identifier))
samp_npz = sitk.GetArrayFromImage(samp)
#%% 对比标准化结果
seg_path = '/data/hsx/project_pc_multimodality_addsize/Task_Pancreas/preprocessed/preprocess_plans_2D/stage_seg'
data = np.load(join(seg_path, "%s.npz" % case_identifier))
data_npz = sitk.GetArrayFromImage(data)

#%% 把肿瘤的病灶非病灶数据集重新采集可视化
ds_path = '/data/hsx/project_pc_multimodality_addsize/Task_Pancreas/lesion_data_2D/non_lesion_images_2'
name = '10124246_0_a_9'
npz = np.load(join(ds_path, name+'.npz'))['data']
plt.imshow(npz, cmap='gray')
plt.show()
#%%
tm_path = '/data/hsx/project_pc_multimodality_addsize/Task_Pancreas/lesion_data_2D/images'
name = '10124246_1_a_0'
npz = np.load(join(tm_path, name+'.npz'))['data']
plt.imshow(npz, cmap='gray')
plt.show()