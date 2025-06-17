import collections
import os
import random
import math
import pandas as pd
from pandas import DataFrame
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm
import cv2
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import torch
from batchgenerators.utilities.file_and_folder_operations import subfiles, load_pickle
from torch.utils.data import Dataset
from dataProcess.path import seg_data_dir, seg_data_2D
import seaborn as sns
from sklearn.decomposition import NMF


# *********************************************** my lesion dataset ************************************************** #
def get_dataset_slice_file_list(data_file_list, id_list, fup=None):
    """
    根据slice数据集文件列表和id_list列表筛选出目标数据集的slice文件列表
    :param data_file_list: 10036753_0_a_37.npz
    :param id_list: 10036753
    """
    file_list = []
    for i, file_name in enumerate(data_file_list):
        id = file_name[:8]
        if id in id_list:
            file_list.append(file_name)
    if fup != None: # 这个字符串将指引提取某一时相的特征
        file_fup_list = []
        for file_name in file_list:
            p = file_name.split("_",3)[2]
            if fup  == p.split(".",2)[0]:  # a d v
                file_fup_list.append(file_name)
        file_list = file_fup_list
    return file_list

def get_file_paths(folder_path, file_names):
    file_paths = [os.path.join(folder_path, file_name) for file_name in file_names]
    return file_paths

class my2DLesionDataset(Dataset):
    def __init__(self, data_path_list, pkl_path, transform=None):
        self.image_file_list = data_path_list
        self.transform = transform
        self.pkl_path = pkl_path

    def __len__(self):
        return len(self.image_file_list)

    def __getitem__(self, idx):
        file_path = self.image_file_list[idx]
        file_name, image_type = file_path.split("/")[-1], file_path.split("/")[-2] # /images/id_tp_fup_sl.npz
        image = np.load(file_path)['data']
        label = 1 if image_type == "images" else 0
        if image_type == "images":
            image = self.cut_by_tumor_center(file_name[:12], image)
        image = cv2.resize(image, (224, 224))
        image = np.expand_dims(image, axis=0)
        return file_name, image, label

    def cut_by_tumor_center(self, case_identifier, image):
        pkl = load_pickle(join(self.pkl_path, case_identifier+".pkl"))
        case_identifier = list(case_identifier)
        case_identifier[-3] = '0'
        case_identifier = ''.join(case_identifier)
        if os.path.exists(join(self.pkl_path, case_identifier+".pkl")): # 如果存在缺失 就以当前的为主
            pkl_bs = load_pickle(join(self.pkl_path, case_identifier+".pkl"))
            max_length = pkl_bs["tumor_max_length"] # 裁剪基线maxlength=d，矩形框2d*2d
        else:
            max_length = pkl["tumor_max_length"]
        center = pkl["tumor_center"]
        resizer = (slice(max(0,int(center[1]-max_length)), min(image.shape[0],int(center[1]+max_length))),
                   slice(max(0,int(center[2]-max_length)), min(image.shape[1],int(center[2]+max_length))))
        return image[resizer]

class myCT_Seq_Dataset(Dataset):
    def __init__(self, data, label, seq_id, year=None, transform=None):
        super().__init__()
        self.data = data
        self.label = label
        self.seq_id = seq_id
        self.year = year
        self.transform = transform

    def __len__(self):
        return len(self.label) # 数据集长度

    def __getitem__(self, index): # 根据索引返回对应的图像和标签
        image, label, seqId = self.data[index], self.label[index], self.seq_id[index]
        if(self.year != None):
            year = self.year[index]
            return image, label, year
        return image, label, seqId

# ******************************************** Feature tools ******************************************************** #
def eva_lesion_feature(model, data_loader, device):
    # 运行推理 收集特征与图片名 特征不需要标签是因为CT整合的时候可以直接获取CT的标签
    lesion_features, files_name = [], []
    progress_bar = tqdm(total=len(data_loader), desc='Extracting Progress', unit='batch')
    with torch.no_grad():
        for file_name, images, _ in data_loader:
            images = images.to(device)
            output = model(images)
            lesion_features.append(output.cpu())
            files_name.append(file_name)
            progress_bar.update(1)
        progress_bar.close()
        lesion_features = torch.cat(lesion_features, dim=0)
        files_name = [list(files) for files in files_name]
        slices_name = []
        for lst in files_name: slices_name += lst
    return lesion_features, slices_name

def process_radiomic_feature(radiomic_df, var_threshold, fea_dim):
    variance = radiomic_df.var()
    threshold_value = np.percentile(variance, var_threshold)
    selected_features = variance[variance >= threshold_value] # 可以查看选出来的特征名称
    selector = VarianceThreshold(threshold=threshold_value)
    X_filtered = selector.fit_transform(radiomic_df) #
    nmf_model = NMF(n_components=fea_dim)
    X_reduced = nmf_model.fit_transform(X_filtered)
    # 计算降维后特征之间的相关性矩阵
    correlation_matrix = np.corrcoef(X_reduced, rowvar=False)
    sns.heatmap(correlation_matrix, cmap='coolwarm')     # 绘制热力图
    plt.title('Heatmap of Feature Correlations')
    # plt.show()
    # 然后返回经由筛选和提取的特征
    df = pd.DataFrame(X_reduced)
    df = df.set_index(radiomic_df.index)
    return df

def get_ct_rad_feature(radiomic_df, data_list):
    rad_dict = collections.OrderedDict()
    lb_df = pd.read_csv('../Task_Pancreas/data_info/labels_CT.csv')
    lb_df.set_index(['id'],inplace=True)
    for case_identifier, feature in radiomic_df.iterrows():
        info = case_identifier.split("_")
        id, tp, fup = info[0], info[1], info[2]
        if fup != 'v': continue # 这里限定只使用一个时相 这里在单独训练组学特征的时候 也只采用了一种时相 依据是通常a时相是比较关键的时相
        label = lb_df.loc[int(id)][tp] # float
        if id in data_list:
            rad_dict[id+tp] = [1, np.array(feature), label]
    return rad_dict

def dict_fea_concat(train_ct_dict, train_rad_dict):
    """
    :param train_ct_dict: dict[id]=[ct_sl_num, ct_feature, ct_label, max_slice_area]
    :param train_rad_dict: dict[id]=[1, rad_feature, ct_label]
    :return: fea_dict[id]=[ct_sl_num, ct_feature+rad_feature, ct_label, max_slice_area]
    """
    id_list = sorted(train_ct_dict.keys()) # 有序的索引
    for key in id_list:
        ct_fea = torch.cat((train_ct_dict[key][1], torch.tensor(train_rad_dict[key][1])), 0)
        train_ct_dict[key][1] = ct_fea # 104796741
    return train_ct_dict

def get_diam_sum_of_target_lesion(diam_sum_path, id_list):
    """
    按照id_list的顺序 依次取出对应id的一行数据，筛选出来径向和
    :param: id_list
    :return: 获取靶病灶径向和，每一次CT都会有一个值 id为索引 0 1 2...为时间点数的df
    """
    df = pd.read_excel(diam_sum_path, sheet_name='Sheet1')
    if len(df.columns) > 16:
        df = df[['基线','sum0','sum1','sum2','sum3','sum4','sum5','sum6','sum7']]
    else:
        df = df[['基线','sum0','sum1','sum2','sum3','sum4']]
    df['id'] = df['基线'].str.split('+').apply(lambda x: x[0] if len(x) > 0 else None)
    df.set_index('id', inplace=True)
    df = df.drop('基线', axis=1)
    df.fillna(-1, inplace=True)
    common_ids = list(set(df.index) & set(id_list))
    diam_df = df.reindex(common_ids)
    diam_sum = []
    for index, row in diam_df.iterrows():
        diam_sum.append(row.tolist())
    return common_ids, diam_sum

def process_clinic_feature(stage_path, diam_sum_path):
    stage_df = pd.read_excel(stage_path, sheet_name='Sheet1')
    stage_df['id'] = stage_df['id'].apply(str)
    id_values = stage_df['id'].tolist()
    com_ids, diam_sum = get_diam_sum_of_target_lesion(diam_sum_path, id_values)
    stage_df.set_index('id', inplace=True)
    stage_sorted = stage_df.reindex(com_ids)
    stg_values = stage_sorted['stage'].tolist()
    clinic_df = DataFrame({'id':com_ids, 'stage':stg_values, 'diam':diam_sum})
    clinic_df.set_index('id', inplace=True)
    return clinic_df

def condition_mean(col_name, column):
    range_dict = {'CA199-baseline': (2, 20000), 'LDH': (120, 250), 'ALB': (40, 55), 'CPR': (0, 8.2)}
    if col_name in range_dict:
        condition = (column > range_dict[col_name][0]) & (column < range_dict[col_name][1])
        return column[condition].mean()
    else:
        return column.mean()

def process_clinic_baseline(clinic_baseline_path, norm=False, in_group='SD'):
    clinic_df = pd.read_excel(clinic_baseline_path, sheet_name='Sheet1')
    # clinic_df = clinic_df[clinic_df['group'] == in_group]
    clinic_df['ID'] = clinic_df['ID'].apply(str)
    clinic_df.set_index('ID', inplace=True)
    clinic_df = clinic_df.drop(['CA199-CT1', '化疗方案：1-AG，2-FFX', 'group'], axis=1)

    for i, column in enumerate(clinic_df.columns):
        if column == 'sex':
            clinic_df['sex'] = clinic_df['sex'].replace({'男': 1, '女': 2})
        if i >= 2:
            if clinic_df[column].isna().any(): # 填充NA
                mean = condition_mean(column, clinic_df[column])
                clinic_df[column] = clinic_df[column].fillna(mean)
    if norm:
        scaler = MinMaxScaler() # StandardScaler()
        numeric_cols = clinic_df.select_dtypes(include=['float64']).columns
        clinic_df[numeric_cols] = scaler.fit_transform(clinic_df[numeric_cols])
    return clinic_df


# ********************************************** my Tools ************************************************************ #
def seed_torch(seed=1286):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def draw_tSNE(features, labels, title):
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=25)  # 降维到 2 维
    features = torch.stack(features)
    features_tsne = tsne.fit_transform(features.to('cpu'))
    # 绘制散点图
    plt.figure(figsize=(8, 6))
    cmap = plt.cm.colors.ListedColormap(['blue', 'red'])  # 定义两种颜色
    scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=labels, cmap=cmap, alpha=0.6)
    plt.colorbar(scatter, label='Class Label')
    plt.title('t-SNE Visualization of Feature Distinction of fold_' + title)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(True)
    plt.show()

def draw_pca(features, labels, title):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    features = torch.stack(features)
    features = features.numpy()  # 转换为 NumPy 数组，因为 PCA 需要 NumPy 数组
    pca_result = pca.fit_transform(features)
    # 绘制散点图
    plt.figure(figsize=(8, 6))
    cmap = plt.cm.colors.ListedColormap(['blue', 'red'])  # 定义两种颜色
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=labels, cmap=cmap, alpha=0.6)
    plt.colorbar(label='Class Label')
    plt.title('PCA Visualization of Feature Distinction of fold_' + title)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.show()


def draw_tr_val_fig(train_losses, train_dice_scores, val_losses, val_dice_scores):
    plt.subplot(121)
    # 绘制损失曲线
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    # 绘制acc均值曲线
    plt.subplot(122)
    plt.plot(train_dice_scores, label='Train Auc')
    plt.plot(val_dice_scores, label='Validation Auc')
    plt.xlabel('Epoch')
    plt.ylabel('Auc Score')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    process_clinic_feature('../Task_Pancreas/data_info/stage.xlsx', '../Task_Pancreas/data_info/SD_diameter_sum.xlsx')

