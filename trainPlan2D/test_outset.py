"""
基于最佳权重，完成外部验证集测试
"""
import math
import os
from os.path import join
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from balanced_loss import Loss
from batchgenerators.utilities.file_and_folder_operations import subfiles
from sklearn.metrics import roc_auc_score, roc_curve, auc, f1_score
from sklearn.model_selection import KFold
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import monai.transforms as mt

from trainPlan2D.draw_pic import draw_topk_fea_bar
from trainPlan2D.models import cnnResNet, LSTM
from trainPlan2D.utils_cls import seed_torch, draw_tr_val_fig, get_dataset_slice_file_list, my2DLesionDataset, \
    get_file_paths, eva_lesion_feature, myCT_Seq_Dataset, process_radiomic_feature, get_ct_rad_feature, dict_fea_concat, \
    process_clinic_feature, process_clinic_baseline, draw_tSNE, draw_pca
from trainPlan2D.utils_seg import seg_slice2ct_feature, seg_ct2seq_feature, collate_fn_sequence
from dataProcess.utils import split_dataset_list
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from captum.attr import IntegratedGradients
import torch.nn.functional as F
from collections import Counter


def keep_top_k_and_zero_others(vector, k=10):
    top_k_indices = np.argsort(vector)[-k:] # 找到前k大的值的索引
    result = np.zeros_like(vector)
    result[top_k_indices] = vector[top_k_indices] # 将前k大的值保留
    return result

# *************** #
# 设备参数
# *************** #
seed_torch(seed=240129) # 固定一个种子
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
prediction_lstm = True
# *************** #
# 文件参数
# *************** #
divided_patient_sets_path = "/data/hsx/project_pc_multimodality_addsize/Task_Pancreas/data_info/divided_patient_sets.xlsx"
lesion_model_path = "./results/weights/cls_0224/ep022-val_acc0.881-val_auc0.954.pth"
cls_model_path =  "./results/weights/fold5_ablation/1" # "./results/weights/fold5_ablation/A" #
lesion_data_dir = "/data/hsx/project_pc_multimodality_addsize/Task_Pancreas_sec/lesion_data_2D"
resampled_data_2D = "/data/hsx/project_pc_multimodality_addsize/Task_Pancreas_sec/preprocessed/preprocess_plans_2D/stage_resample_xy"
# *************** #
# 模型参数
# *************** #
def load_fea_model(lesion_model_path):
    model = cnnResNet(isClassify=False).to(device) # 设置classify=false表示提取特征
    if lesion_model_path != "":
        model.load_state_dict(torch.load(lesion_model_path))
    model.eval()
    return model
def load_cls_model(cls_model_path):
    model = LSTM(CNN_embed_dim=1).to(device) # 512+ [stage+max_len+diameter_sum] = 512+1+1+1 = 515 + base_clinic
    model.load_state_dict(torch.load(cls_model_path))
    # model.eval()
    return model

def reset(fold):
    cls_model = load_cls_model(join(cls_model_path, 'fold'+str(fold), 'ep100.pth'))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cls_model.parameters(), 1e-3, betas = (0.9, 0.999), weight_decay = 2e-4) #
    # optimizer = optim.SGD(cls_model.parameters(), lr=5e-3, momentum=0.9, weight_decay=1e-2)
    return fea_model, cls_model, criterion, optimizer

fea_model = load_fea_model(lesion_model_path)
# *************** #
# 数据划分与处理与特征提取与特征整合
# *************** #
data_transforms = {
    "train": transforms.Compose([
        mt.ToTensor()
    ]),
    "val": transforms.Compose([
        mt.ToTensor()
    ])
}

# 导入基线clinic指标数据
clinic_baseline_df = process_clinic_baseline('../Task_Pancreas/data_info/pancreas_baseline_clinical_info.xlsx', in_group='PR')
# 导入stage分期+靶病灶径向和序列的 id-feature 的df用于ct序列特征的拼接
clinic_df = process_clinic_feature('../Task_Pancreas/data_info/stage.xlsx', '../Task_Pancreas/data_info/SEC_diameter_sum.xlsx')
# 划分训练集和验证集 按照文件名全部划分一通 提取特征的时候只需要lesion data且不用接收标签
data_file_list = np.array(os.listdir(join(lesion_data_dir, "images"))) # all病人slice，需要根据train_list val_list挑选slice
sets = pd.read_excel(divided_patient_sets_path, sheet_name="SEC") # dataset dataframe
testList = []
for i in range(len(sets['test'].dropna(axis=0, how='any'))):
    # print(i, sets.loc[i]) # 10619760+20210817+CT
    sets.loc[i] = sets.loc[i].str.split('+')
    testList.append(sets.loc[i]['test'][0]) # test组是原有设置组 基线组是仅只有基线的组


test_data_file = get_dataset_slice_file_list(data_file_list, testList) # , fup='a'
te_path_file_list = get_file_paths(join(lesion_data_dir, "images"), test_data_file)
# 创建数据加载器
test_dataset = my2DLesionDataset(te_path_file_list, join(resampled_data_2D, "images"), transform=data_transforms["train"])
test_loader = DataLoader(test_dataset, batch_size=50, shuffle=True)
# 特征提取和整理 整理中包括有组学特征的整理 以及函数的修改重写
test_seg_features, test_files_name = eva_lesion_feature(fea_model, test_loader, device) # [n, 960]
train_ct_dict = seg_slice2ct_feature(test_seg_features, test_files_name) # dict[id]=[ct_sl_num, feature960, ct_label, max_slice_area]
test_seqs_data, test_seqs_label, test_seqs_id = seg_ct2seq_feature(train_ct_dict, clinic_df, clinic_baseline_df, add_one=False) # [222233334445]的时间点序列
# 装载序列数据
test_seqDataset = myCT_Seq_Dataset(test_seqs_data, test_seqs_label, test_seqs_id)
test_seq_loader = DataLoader(test_seqDataset, batch_size=15, collate_fn=collate_fn_sequence) # 15


# *************** #
# 五模型的外部验证集基础测试
# *************** #
final_pred_y, final_pred_y_p, final_y = [], [], [] # 这里记录全部折的
folds_acc, folds_auc, folds_sen, folds_spe = [], [], [], []
folds_topk_indices = []
for fold in range(5):
    fea_model, cls_model, criterion, optimizer = reset(fold+1)
    progress_bar = tqdm(total=len(test_seq_loader), desc='Valing LSTM Progress', unit='batch')
    pred_y, pred_y_p, y = [], [], []
    attribution_fold = []
    fea_embeds = []
    loss_val = 0.0
    seqs_id = []
    cls_model.eval()
    with torch.no_grad():
        for i, (seq_len, inputs, labels, seq_id) in enumerate(test_seq_loader):
            labels = labels.type(torch.LongTensor)
            inputs = inputs.to(torch.float)
            inputs, labels = inputs.to(device), labels.to(device)
            seqs_id += seq_id
            outputs = cls_model(inputs)
            # outputs, embeds = cls_model(inputs)
            # fea_embeds.extend(embeds.cpu())

            cls_model.train()
            # applying integrated gradients on the SoftmaxModel and input data point
            # ig = IntegratedGradients(cls_model)
            # attributions, approximation_error = ig.attribute(inputs, target=1, n_steps=100, return_convergence_delta=True) #
            # attributions = attributions.cpu().numpy()
            # attributions_merged = np.apply_along_axis(keep_top_k_and_zero_others, 2, attributions[:, :, :512])
            # attributions_merged = np.concatenate((attributions_merged, attributions[:, :, 512:]), axis=-1)
            # attributions_merged = torch.from_numpy(attributions_merged)
            # # attributions_sum = attributions[:, :, :512].sum(dim=-1, keepdim=True)
            # # attributions_merged = torch.cat((attributions_merged, attributions[:, :, 512:]), dim=-1)
            # attributions_padded = F.pad(attributions_merged, (0, 0, 0, 5 - attributions_merged.shape[1]), "constant", 0)
            # attribution_fold.extend(attributions_padded)
            # cls_model.eval()

            loss = criterion(outputs, labels)
            loss_val += loss.item()
            # 计算0.5阈值可以 但约登指数不行 记得注释
            _, pred = torch.max(outputs.data, 1)
            # pred_y.extend(pred.cpu().detach().numpy())
            pred_y_p.extend(torch.sigmoid(outputs[:, 1]).cpu().detach().numpy())
            y.extend(labels.cpu().detach().numpy())
            progress_bar.update(1)
    auc_val = roc_auc_score(np.array(y), np.array(pred_y_p))
    # tn, fp, fn, tp = confusion_matrix(y, pred_y).ravel()

    # folds内单独使用约登指数
    fpr, tpr, thresholds = roc_curve(y, pred_y_p)
    youden_index = tpr - fpr
    optimal_threshold = thresholds[np.argmax(youden_index)]
    pred_y.extend((pred_y_p >= optimal_threshold).astype(int))
    tn, fp, fn, tp = confusion_matrix(y, pred_y).ravel()

    acc_val = np.sum(np.array(pred_y)==(np.array(y))) / len(y)
    sen_val = tp / (tp + fn)
    spe_val = tn / (tn + fp)

    f1_mi = f1_score(y, pred_y, average='micro')
    f1_ma = f1_score(y, pred_y, average='macro')

    # 一折测试的t-SNE图
    # draw_pca(fea_embeds, y, str(fold+1))

    progress_bar.close()

    # 保存预测结果和标准结果
    final_y += y
    # final_pred_y += pred_y
    final_pred_y_p += pred_y_p
    seqs_id = np.array(seqs_id)
    df = pd.DataFrame({'seq_id': seqs_id, 'label': y, 'pred': pred_y, 'pred_p': pred_y_p})
    df.to_csv('./results/ans_pic/1_sec/fold_'+str(fold+1)+'.csv', index=False) # 拼接label-pred_p用于绘制ROC
    # 打印fold折的预测指标
    loss_val /= len(test_seq_loader)
    print(f"Fold {fold+1}/5, "  f"Test Loss: {loss_val:.4f}",
          f"Test acc: {acc_val:.4f}, sen: {sen_val:.4f}, spe: {spe_val:.4f}, auc: {auc_val:.4f}, f1-score: {f1_mi:.4f} {f1_ma:.4f}"
          )
    folds_acc.append(acc_val)
    folds_auc.append(auc_val)
    folds_sen.append(sen_val)
    folds_spe.append(spe_val)


    # 查看贡献度 现在是18维 事实上只有基线多出来11维 所以选择决定性的前3 前5就差不多了？
    # topk_indices = []
    # attribution_fold = torch.stack(attribution_fold, dim=0)
    # for t in range(5):
    #     current_time_attributions = attribution_fold[:, t, :]
    #     # 这里找到了每个序列在t时间点贡献程度最高的三个特征
    #     _, indices = torch.topk(current_time_attributions, k=3, dim=1, largest=True)
    #     # 把索引放到对应序列的贡献度上判断是否为0 是则设置999作为无效标识 不纳入统计
    #     for i, index_group in enumerate(indices):
    #         for j, idx in enumerate(index_group):
    #             if current_time_attributions[i, idx] == 0: # 如果为0，则将indices中对应的索引设置为999
    #                 indices[i, j] = 999
    #     counter = Counter(indices.view(-1).tolist())
    #     topk_indices.append(counter.most_common(5))
    # folds_topk_indices.append(topk_indices)

# folds_counts = []
# for time_point in range(5):
#     feature_counts = {}
#     for model_indices in folds_topk_indices:
#         for index, count in model_indices[time_point]:
#             if index in feature_counts:
#                 feature_counts[index] += count
#             else:
#                 feature_counts[index] = count
#     for index in feature_counts:
#         feature_counts[index] //= 5
#     folds_counts.append(feature_counts)


# 计算完整的 所有测试数据的 acc auc spe sen
final_auc = roc_auc_score(np.array(final_y), np.array(final_pred_y_p))
fpr, tpr, thresholds = roc_curve(final_y, final_pred_y_p)
youden_index = tpr - fpr
optimal_threshold = thresholds[np.argmax(youden_index)]
final_pred_y.extend((final_pred_y_p >= optimal_threshold).astype(int))
tn, fp, fn, tp = confusion_matrix(final_y, final_pred_y).ravel()
final_acc = np.sum(np.array(final_pred_y)==(np.array(final_y))) / len(final_y)
final_sen = tp / (tp + fn)
final_spe = tn / (tn + fp)

