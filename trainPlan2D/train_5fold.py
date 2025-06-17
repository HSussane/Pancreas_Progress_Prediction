"""
这里基于最佳结果的设置，这里
"""
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
from captum.attr import IntegratedGradients
from sklearn.metrics import roc_auc_score, roc_curve, auc, f1_score
from sklearn.model_selection import KFold
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import monai.transforms as mt
from trainPlan2D.models import cnnResNet, LSTM
from trainPlan2D.utils_cls import seed_torch, draw_tr_val_fig, get_dataset_slice_file_list, my2DLesionDataset, \
    get_file_paths, eva_lesion_feature, myCT_Seq_Dataset, process_radiomic_feature, get_ct_rad_feature, dict_fea_concat, \
    process_clinic_feature, process_clinic_baseline, draw_tSNE
from trainPlan2D.utils_seg import seg_slice2ct_feature, seg_ct2seq_feature, collate_fn_sequence
from dataProcess.path import lesion_data_dir, resampled_data_2D
from dataProcess.utils import split_dataset_list
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
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
# *************** #
# 文件参数
# *************** #
save_model_dir = "./results/weights/fold5_ablation"
divided_patient_sets_path = "/data/hsx/project_pc_multimodality_addsize/Task_Pancreas/data_info/divided_patient_sets.xlsx"
lesion_model_path = "./results/weights/cls_0224/ep022-val_acc0.881-val_auc0.954.pth"
# cls_model_path =  "./results/weights/cls_seq_0704/5/ep100-val_acc0.765-val_auc0.740.pth" # 之前的最佳 但并不是五折实验

# *************** #
# 模型参数
# *************** #
def load_fea_model(lesion_model_path):
    model = cnnResNet(isClassify=False).to(device) # 设置seg=false表示提取特征
    if lesion_model_path != "":
        model.load_state_dict(torch.load(lesion_model_path))
    model.eval()
    return model
def load_cls_model():
    # 512+ [stage+max_len+diameter_sum] = 512+1+1+1 = 515 + 14 baseline = 529
    model = LSTM(CNN_embed_dim=529).to(device) # 529 + 组学特征
    return model

def reset():
    fea_model = load_fea_model(lesion_model_path)
    cls_model = load_cls_model()
    criterion = nn.CrossEntropyLoss()
    # criterion = Loss(
    #     loss_type = 'cross_entropy',
    #     samples_per_class = [168,81], # 249
    #     beta = 0.9999,
    #     class_balanced = True
    # )
    optimizer = optim.Adam(cls_model.parameters(), 1e-3, betas = (0.9, 0.999), weight_decay = 2e-4) #
    # optimizer = optim.SGD(cls_model.parameters(), lr=5e-3, momentum=0.9, weight_decay=1e-2)
    return fea_model, cls_model, criterion, optimizer

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

# 导入组学特征
# radiomic_df = pd.read_excel('./Pancreas-Radiomics-features.xlsx', sheet_name='Sheet1')
# radiomic_df.set_index(radiomic_df.columns[0], inplace=True)
# radiomic_df = process_radiomic_feature(radiomic_df, var_threshold=84, fea_dim=110) # 60

# 导入基线clinic指标数据
clinic_baseline_df = process_clinic_baseline('../Task_Pancreas/data_info/pancreas_baseline_clinical_info.xlsx')
# 导入stage分期+靶病灶径向和序列的 id-feature 的df用于ct序列特征的拼接
clinic_df = process_clinic_feature('../Task_Pancreas/data_info/stage.xlsx', '../Task_Pancreas/data_info/SD_diameter_sum.xlsx')
# 划分训练集和验证集 按照文件名全部划分一通 提取特征的时候只需要lesion data且不用接收标签
data_file_list = np.array(os.listdir(join(lesion_data_dir, "images"))) # all病人slice，需要根据train_list val_list挑选slice
trainList, valList, testList = split_dataset_list(divided_patient_sets_path, "622")
patientArr = np.array(trainList + valList + testList) # 按病人来划分成新的五折

# 这里读入不同的List后，将他们重新组织，随机打乱，化成五折
folds_topk_indices = [] # 贡献度
final_pred_y, final_pred_y_p, final_y = [], [], [] # 这里记录全部折的
folds_acc, folds_auc, folds_sen, folds_spe, folds_f1s = [], [], [], [], []
k1fold = KFold(n_splits=5, shuffle=True, random_state=24)
for fold, (indices, test_indices) in enumerate(k1fold.split(patientArr)):

    fea_model, cls_model, criterion, optimizer = reset()

    k2fold = KFold(n_splits=4, shuffle=True, random_state=95)
    train_indices, val_indices = next(k2fold.split(indices)) # 每次只选择随即划分后的第一折 作为一次622

    train_data_file = get_dataset_slice_file_list(data_file_list, patientArr[train_indices]) # , fup='d'
    val_data_file = get_dataset_slice_file_list(data_file_list, patientArr[val_indices]) # , fup='d'
    # 整理lesion类图片的文件路径列表
    tr_path_file_list = get_file_paths(join(lesion_data_dir, "images"), train_data_file)
    val_path_file_list = get_file_paths(join(lesion_data_dir, "images"), val_data_file)

    # 创建数据加载器
    train_dataset = my2DLesionDataset(tr_path_file_list, join(resampled_data_2D, "images"), transform=data_transforms["train"])
    val_dataset = my2DLesionDataset(val_path_file_list, join(resampled_data_2D, "images"), transform=data_transforms["val"])
    train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=50, shuffle=False)
    # 特征提取和整理 整理中包括有组学特征的整理 以及函数的修改重写
    train_seg_features, train_files_name = eva_lesion_feature(fea_model, train_loader, device) # [n, 960]
    val_seg_features, val_files_name = eva_lesion_feature(fea_model, val_loader, device)
    train_ct_dict = seg_slice2ct_feature(train_seg_features, train_files_name) # dict[id]=[ct_sl_num, feature960, ct_label, max_slice_area]
    val_ct_dict = seg_slice2ct_feature(val_seg_features, val_files_name)

    # 组学特征的组装
    # train_rad_dict = get_ct_rad_feature(radiomic_df, patientArr[train_indices]) # dict[id]=[1, rad_feature, ct_label]
    # val_rad_dict = get_ct_rad_feature(radiomic_df, patientArr[val_indices])
    # 重新整合深度特征和组学特征 concat形式 就是对应id 将rad_dict的特征拼到ct_dict的特征上
    # train_fea_dict = dict_fea_concat(train_ct_dict, train_rad_dict)
    # val_fea_dict = dict_fea_concat(val_ct_dict, val_rad_dict)

    train_seqs_data, train_seqs_label, train_seqs_id = seg_ct2seq_feature(train_ct_dict, clinic_df, clinic_baseline_df) # [222233334445]的时间点序列
    val_seqs_data, val_seqs_label, val_seqs_id = seg_ct2seq_feature(val_ct_dict, clinic_df, clinic_baseline_df)
    # 装载序列数据
    train_seqDataset = myCT_Seq_Dataset(train_seqs_data, train_seqs_label, train_seqs_id)
    val_seqDateset = myCT_Seq_Dataset(val_seqs_data, val_seqs_label, val_seqs_id)
    train_seq_loader = DataLoader(train_seqDataset, batch_size=15, collate_fn=collate_fn_sequence) # 10
    val_seq_loader = DataLoader(val_seqDateset, batch_size=15, collate_fn=collate_fn_sequence)

    # *************** #
    # 时序训练
    # *************** #
    num_epochs = 100
    save_period = 2
    save_flag = False
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    train_losses, train_acc, train_auc, val_losses, val_acc, val_auc = [], [], [], [], [], [] # 绘图用

    for epoch in range(num_epochs):
        print("--------------------------Epoch ", epoch, "-------------------------------")
        loss_tr, loss_val = 0.0, 0.0 # 损失监督
        acc_tr, acc_val, auc_tr, auc_val = 0.0, 0.0, 0.0, 0.0 # 评价指标
        sen_tr, sen_val, spe_tr, spe_val = 0.0, 0.0, 0.0, 0.0
        pred_y, pred_y_p, y = [], [], []
        progress_bar = tqdm(total=len(train_seq_loader), desc='Training LSTM Progress', unit='batch')
        cls_model.train()
        for i, (seq_len, inputs, labels, _) in enumerate(train_seq_loader):
            labels = labels.type(torch.LongTensor)
            inputs = inputs.to(torch.float)
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = cls_model(inputs)
            # outputs, _ = cls_model(inputs)
            loss = criterion(outputs, labels) #
            loss.backward()
            optimizer.step()
            loss_tr += loss.item()
            _, pred = torch.max(outputs.data, 1)
            pred_y.extend(pred.cpu().detach().numpy())
            pred_y_p.extend(torch.sigmoid(outputs[:, 1]).cpu().detach().numpy())
            y.extend(labels.cpu().detach().numpy())
            progress_bar.update(1)
        auc_tr = roc_auc_score(np.array(y), np.array(pred_y_p))
        tn, fp, fn, tp = confusion_matrix(y, pred_y).ravel()

        # 使用约登指数
        # fpr, tpr, thresholds = roc_curve(y, pred_y_p)
        # youden_index = tpr - fpr
        # optimal_threshold = thresholds[np.argmax(youden_index)]
        # pred_y = (pred_y_p >= optimal_threshold).astype(int)
        # tn, fp, fn, tp = confusion_matrix(y, pred_y).ravel()

        acc_tr = np.sum(np.array(pred_y)==(np.array(y))) / len(y)
        sen_tr = tp / (tp + fn)
        spe_tr = tn / (tn + fp)

        scheduler.step() # 每个epoch调整学习率
        progress_bar.close()

        # 在验证集上评估模型
        progress_bar = tqdm(total=len(val_seq_loader), desc='Valing LSTM Progress', unit='batch')
        pred_y, pred_y_p, y = [], [], []
        cls_model.eval()
        with torch.no_grad():
            for i, (seq_len, inputs, labels, _) in enumerate(val_seq_loader):
                labels = labels.type(torch.LongTensor)
                inputs = inputs.to(torch.float)
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = cls_model(inputs)
                # outputs, _ = cls_model(inputs)
                loss = criterion(outputs, labels)
                loss_val += loss.item()
                _, pred = torch.max(outputs.data, 1)
                pred_y.extend(pred.cpu().detach().numpy())
                pred_y_p.extend(torch.sigmoid(outputs[:, 1]).cpu().detach().numpy())
                y.extend(labels.cpu().detach().numpy())
                progress_bar.update(1)
            auc_val = roc_auc_score(np.array(y), np.array(pred_y_p))
            tn, fp, fn, tp = confusion_matrix(y, pred_y).ravel()

            # 使用约登指数
            # fpr, tpr, thresholds = roc_curve(y, pred_y_p)
            # youden_index = tpr - fpr
            # optimal_threshold = thresholds[np.argmax(youden_index)]
            # pred_y = (pred_y_p >= optimal_threshold).astype(int)
            # tn, fp, fn, tp = confusion_matrix(y, pred_y).ravel()

            acc_val = np.sum(np.array(pred_y)==(np.array(y))) / len(y)
            sen_val = tp / (tp + fn)
            spe_val = tn / (tn + fp)
            progress_bar.close()

        # 打印每个 epoch 的训练损失和验证损失均值 以及dice评估分数
        if train_seq_loader is not None: loss_tr /= len(train_seq_loader)
        loss_val /= len(val_seq_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, "  f"Train Loss: {loss_tr:.4f}, Val Loss: {loss_val:.4f}",
              f"Train acc: {acc_tr:.4f}, sen: {sen_tr:.4f}, spe: {spe_tr:.4f}",
              f"Val acc: {acc_val:.4f}, sen: {sen_val:.4f}, spe: {spe_val:.4f}",
              f"Train auc: {auc_tr:.4f}, Val auc: {auc_val:.4f}",
              )

        # 加入绘图列表
        train_losses.append(loss_tr)
        train_acc.append(acc_tr)
        train_auc.append(auc_tr)
        val_losses.append(loss_val)
        val_acc.append(acc_val)
        val_auc.append(auc_val)
        # 保存模型权重
        if save_flag and ((epoch + 1) % save_period == 0 or epoch + 1 == num_epochs):
            torch.save(cls_model.state_dict(), os.path.join(save_model_dir, '1', 'fold'+str(fold+1), 'ep%03d-val_acc%.3f-val_auc%.3f.pth'
                                                            %((epoch + 1), acc_val, auc_val)))
    # 训练一轮后绘图
    draw_tr_val_fig(train_losses, train_auc, val_losses, val_auc)
    # 测试 选取最后一个epoch
    test_data_file = get_dataset_slice_file_list(data_file_list, patientArr[test_indices]) # , fup='d'
    test_path_file_list = get_file_paths(join(lesion_data_dir, "images"), test_data_file)
    test_dataset = my2DLesionDataset(test_path_file_list, join(resampled_data_2D, "images"), transform=data_transforms["val"])
    test_loader = DataLoader(test_dataset, batch_size=50, shuffle=False)
    test_seg_features, test_files_name = eva_lesion_feature(fea_model, test_loader, device)
    test_ct_dict = seg_slice2ct_feature(test_seg_features, test_files_name)

    # test_rad_dict = get_ct_rad_feature(radiomic_df, patientArr[test_indices])
    # test_fea_dict = dict_fea_concat(test_ct_dict, test_rad_dict)

    test_seqs_data, test_seqs_label, test_seqs_id = seg_ct2seq_feature(test_ct_dict, clinic_df, clinic_baseline_df)
    # 装载序列数据 test_loader 但是命名 val_seq_loader
    test_seqDataset = myCT_Seq_Dataset(test_seqs_data, test_seqs_label, test_seqs_id)
    test_seq_loader = DataLoader(test_seqDataset, batch_size=15, collate_fn=collate_fn_sequence)

    # 在测试集上评估模型
    progress_bar = tqdm(total=len(test_seq_loader), desc='Test LSTM Progress', unit='batch')
    acc_te, auc_te, sen_te, spe_te = 0.0, 0.0, 0.0, 0.0 # 评价指标
    pred_y, pred_y_p, y = [], [], []
    loss_te = 0.0
    seqs_id = []
    attribution_fold = []
    fea_embeds = []
    cls_model.load_state_dict(torch.load("./results/weights/fold5_ablation/529/fold" + str(fold+1) +"/ep100.pth"))
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
            # 计算特征贡献度
            # cls_model.train()
            # # applying integrated gradients on the SoftmaxModel and input data point
            # ig = IntegratedGradients(cls_model)
            # attributions, approximation_error = ig.attribute(inputs, target=1, n_steps=100, return_convergence_delta=True) #
            # attributions = attributions.cpu().numpy()
            # attributions_merged = np.apply_along_axis(keep_top_k_and_zero_others, 2, attributions[:, :, :512])
            # attributions_merged = np.concatenate((attributions_merged, attributions[:, :, 512:]), axis=-1)
            # attributions_merged = torch.from_numpy(attributions_merged)
            # # attributions_sum = attributions[:, :, :512].sum(dim=-1, keepdim=True)
            # # attributions_merged = torch.cat((attributions_merged, attributions[:, :, 512:]), dim=-1)
            # attributions_padded = F.pad(attributions_merged, (0, 0, 0, 8 - attributions_merged.shape[1]), "constant", 0)
            # attribution_fold.extend(attributions_padded)
            # cls_model.eval()

            loss = criterion(outputs, labels)
            loss_te += loss.item()
            _, pred = torch.max(outputs.data, 1)
            # pred_y.extend(pred.cpu().detach().numpy()) # 0.5阈值预测
            pred_y_p.extend(torch.sigmoid(outputs[:, 1]).cpu().detach().numpy())
            y.extend(labels.cpu().detach().numpy())
            progress_bar.update(1)
        auc_te = roc_auc_score(np.array(y), np.array(pred_y_p))
        # tn, fp, fn, tp = confusion_matrix(y, pred_y).ravel()

        # 使用约登指数
        fpr, tpr, thresholds = roc_curve(y, pred_y_p)
        youden_index = tpr - fpr
        optimal_threshold = thresholds[np.argmax(youden_index)]
        print(fold+1, "的分类阈值是：", optimal_threshold)
        pred_y.extend((pred_y_p >= optimal_threshold).astype(int))
        tn, fp, fn, tp = confusion_matrix(y, pred_y).ravel()

        acc_te = np.sum(np.array(pred_y)==(np.array(y))) / len(y)
        sen_te = tp / (tp + fn)
        spe_te = tn / (tn + fp)

        f1_mi = f1_score(y, pred_y, average='micro')
        f1_ma = f1_score(y, pred_y, average='macro')

        # 一折测试的t-SNE图
        # draw_tSNE(fea_embeds, y, str(fold+1))

        progress_bar.close()

    # 保存预测结果和标准结果
    final_y += y
    # final_pred_y += pred_y
    final_pred_y_p += pred_y_p
    seqs_id = np.array(seqs_id)
    df = pd.DataFrame({'seq_id': seqs_id, 'label': y, 'pred': pred_y, 'pred_p': pred_y_p})
    df.to_csv('./results/529_sd_5folds/fold_'+str(fold+1)+'.csv', index=False)
    # 打印fold折的预测指标
    loss_te /= len(test_seq_loader)
    print(f"Fold {fold+1}/5, "  f"Test Loss: {loss_te:.4f}",
          f"Test acc: {acc_te:.4f}, sen: {sen_te:.4f}, spe: {spe_te:.4f}, auc: {auc_te:.4f}, f1-score: {f1_mi:.4f} {f1_ma:.4f}"
          )
    folds_acc.append(acc_te)
    folds_auc.append(auc_te)
    folds_sen.append(sen_te)
    folds_spe.append(spe_te)
    folds_f1s.append(f1_ma)


    # 查看贡献度 现在是18维 事实上只有基线多出来11维 所以选择决定性的前3 前5就差不多了？
    # topk_indices = []
    # attribution_fold = torch.stack(attribution_fold, dim=0)
    # for t in range(8):
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

# 计算五折模型平均的前3 4贡献度特征
folds_counts = []
for time_point in range(8):
    feature_counts = {}
    for model_indices in folds_topk_indices:
        for index, count in model_indices[time_point]:
            if index in feature_counts:
                feature_counts[index] += count
            else:
                feature_counts[index] = count
    for index in feature_counts:
        feature_counts[index] //= 5
    folds_counts.append(feature_counts)

# 计算完整的 所有测试数据取约登指数的 acc auc spe sen
final_auc = roc_auc_score(np.array(final_y), np.array(final_pred_y_p))
fpr, tpr, thresholds = roc_curve(final_y, final_pred_y_p)
youden_index = tpr - fpr
optimal_threshold = thresholds[np.argmax(youden_index)]
final_pred_y.extend((final_pred_y_p >= optimal_threshold).astype(int))
tn, fp, fn, tp = confusion_matrix(final_y, final_pred_y).ravel()
final_acc = np.sum(np.array(final_pred_y)==(np.array(final_y))) / len(final_y)
final_sen = tp / (tp + fn)
final_spe = tn / (tn + fp)


