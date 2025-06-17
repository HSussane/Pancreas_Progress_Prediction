# ******************************************************************************************************************** #
# 1.在dataset中裁剪的时候 读取对应slice基线CT的pkl文件和对应CT的pkl文件，分别获取2d的大小以及该CT的肿瘤中心
# 2.slice的标签是顺带获取的，在dataset中就可以直接给出
# 3.对于同一张slice会裁剪出一张同样大小的肿瘤外区域 预想的方式是在 肿瘤中心外围 2.53d的圆周上随机选择野哥为中心点
# ******************************************************************************************************************** #
import os
from os.path import join
import random
from dataProcess.path import resampled_data_2D
import torch
from batchgenerators.utilities.file_and_folder_operations import subfiles
from sklearn.metrics import roc_auc_score
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import monai.transforms as mt
from tqdm import tqdm

from bagnet import BagNet33
from Plan_2D_Cls.models import cnnResNet
from Plan_2D_Cls.utils import seed_torch, get_dataset_slice_file_list, get_file_paths, my2DLesionDataset, \
    draw_tr_val_fig
import numpy as np

from dataProcess.path import lesion_data_dir
from dataProcess.utils import split_dataset_list

seed_torch(seed=240129) # 固定一个种子
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# *************** #
# 文件参数
# *************** #
save_model_dir = "./results/weights/cls_bag_0620" # 0227/2
divided_patient_sets_path = "/data/hsx/project_pc_multimodality_addsize/Task_Pancreas/data_info/divided_patient_sets.xlsx"

# *************** #
# 模型参数
# *************** #
# model = cnnResNet(isClassify=True).to(device)
model = BagNet33(pretrain=True).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), 5e-3, betas = (0.9, 0.999), weight_decay = 5e-4)
# optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-2)

# *************** #
# 数据划分与处理
# *************** #
data_transforms = {
    "train": transforms.Compose([
        mt.RandRotate(range_x=90, range_y=90, prob=0.4),
        mt.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.2),  # 水平镜像
    ]),
    "val": transforms.Compose([
        mt.ToTensor()
    ])
}

# 划分训练集和验证集 按照文件名全部划分一通
# 挑选case_identifier
tum_data_file_list = np.array(os.listdir(join(lesion_data_dir, "images"))) # all病人slice，需要根据train_list val_list挑选slice
tis_data_file_list = np.array(os.listdir(join(lesion_data_dir, "non_lesion_images_2")))
trainList, valList, _ = split_dataset_list(divided_patient_sets_path, "622") # id list
tum_train_data_file = get_dataset_slice_file_list(tum_data_file_list, trainList)
tum_val_data_file = get_dataset_slice_file_list(tum_data_file_list, valList)
tis_train_data_file = get_dataset_slice_file_list(tis_data_file_list, trainList)
tis_val_data_file = get_dataset_slice_file_list(tis_data_file_list, valList)
# 整理两类图片的文件路径列表
tr_lesion_npz_file_list = get_file_paths(join(lesion_data_dir, "images"), tum_train_data_file)
tr_non_lesion_npz_file_list = get_file_paths(join(lesion_data_dir, "non_lesion_images_2"), tis_train_data_file)
tr_path_file_list = tr_lesion_npz_file_list + tr_non_lesion_npz_file_list
random.shuffle(tr_path_file_list)
val_lesion_npz_file_list = get_file_paths(join(lesion_data_dir, "images"), tum_val_data_file)
val_non_lesion_npz_file_list = get_file_paths(join(lesion_data_dir, "non_lesion_images_2"), tis_val_data_file)
val_path_file_list = val_lesion_npz_file_list + val_non_lesion_npz_file_list
random.shuffle(val_path_file_list)
# 装载数据集
train_dataset = my2DLesionDataset(tr_path_file_list, join(resampled_data_2D, "images"), transform=data_transforms["train"])
val_dataset = my2DLesionDataset(val_path_file_list, join(resampled_data_2D, "images"), transform=data_transforms["val"])
train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=50, shuffle=False)

# *************** #
# 训练模型
# *************** #
num_epochs = 120
save_period = 3
save_flag = True
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
train_losses, train_acc_scores, val_losses, val_acc_scores = [], [], [], [] # 绘图用

for epoch in range(num_epochs):
    print("--------------------------Epoch ", epoch, "-------------------------------")
    loss_tr, loss_val = 0.0, 0.0 # 损失监督
    acc_tr, acc_val, auc_tr, auc_val = 0.0, 0.0, 0.0, 0.0 # 评价指标
    pred_y, pred_y_p, y = [], [], []

    progress_bar = tqdm(total=len(train_loader), desc='Training Progress', unit='batch')
    model.train()
    for file_name, images, labels in train_loader:
        labels = labels.type(torch.LongTensor)
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        loss_tr += loss.item()
        _, pred = torch.max(outputs.data, 1)
        pred_y.extend(pred.cpu().detach().numpy())
        pred_y_p.extend(torch.sigmoid(outputs[:, 1]).cpu().detach().numpy())
        y.extend(labels.cpu().detach().numpy())
        progress_bar.update(1)
    scheduler.step() # 每个epoch调整学习率
    progress_bar.close()
    acc_tr = np.sum(np.array(pred_y)==(np.array(y))) / len(y)  # 完整epoch预测正确的个数
    auc_tr = roc_auc_score(np.array(y), np.array(pred_y_p))

    # 在验证集上评估模型
    progress_bar = tqdm(total=len(val_loader), desc='Valing Progress', unit='batch')
    model.eval()
    with torch.no_grad():
        for file_name, images, labels in val_loader:
            labels = labels.type(torch.LongTensor)
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_val += loss.item()
            _, pred = torch.max(outputs.data, 1)
            pred_y.extend(pred.cpu().detach().numpy())
            pred_y_p.extend(torch.sigmoid(outputs[:, 1]).cpu().detach().numpy())
            y.extend(labels.cpu().detach().numpy())
            progress_bar.update(1)
        progress_bar.close()
        acc_val = np.sum(np.array(pred_y)==(np.array(y))) / len(y)  # 完整epoch预测正确的个数
        auc_val = roc_auc_score(np.array(y), np.array(pred_y_p))

    # 打印每个 epoch 的训练损失和验证损失均值 以及dice评估分数
    loss_tr /= len(train_loader)
    loss_val /= len(val_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, "  f"Train Loss: {loss_tr:.4f}, Val Loss: {loss_val:.4f}",
          f"Train acc: {acc_tr:.4f}, Val acc: {acc_val:.4f}",
          f"Train auc: {auc_tr:.4f}, Val auc: {auc_val:.4f}")

    # 加入绘图列表
    train_losses.append(loss_tr)
    train_acc_scores.append(acc_tr)
    val_losses.append(loss_val)
    val_acc_scores.append(acc_val)
    # 保存模型权重
    if save_flag and ((epoch + 1) % save_period == 0 or epoch + 1 == num_epochs):
        torch.save(model.state_dict(), os.path.join(save_model_dir, 'ep%03d-val_acc%.3f-val_auc%.3f.pth'
                                                   %((epoch + 1), acc_val, auc_val)))

# 绘图
draw_tr_val_fig(train_losses, train_acc_scores, val_losses, val_acc_scores)