import collections
import math
import pandas as pd
from functools import partial
from os.path import join
import random
import os
import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm
from batchgenerators.utilities.file_and_folder_operations import load_pickle
import torch.nn.utils.rnn as rnn_utils
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

from trainPlan2D.utils_cls import process_clinic_feature, process_clinic_baseline
from dataProcess.path import seg_data_2D
import copy

# ********************************************** Dataset ************************************************************* #
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


class my2DSegDataset(Dataset):
    def __init__(self, seg_data_file, seg_data_dir, full_img=True, transform=None):
        self.image_dir = join(seg_data_dir, "imagesTr")
        self.mask_dir = join(seg_data_dir, "masksTr" )
        self.full_img = full_img
        self.transform = transform
        self.image_files = seg_data_file
        print("Full Img is", full_img)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        file_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, file_name)
        mask_path = os.path.join(self.mask_dir, file_name)
        image = np.load(img_path)['data']
        mask = np.load(mask_path)['data']

        if self.full_img is False: # 就是说每张图要经过一点裁剪
            # print(file_name)
            image, mask = self.do_cut(image, mask, file_name[0:12])
            image, mask = self.do_resize(image, mask, target_length=224)
        else: # 那就是按纵横比缩放 然后补黑边
            image, mask = self.do_resize(image, mask, padding=True)

        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)
        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)
        return file_name, image, mask

    def do_cut(self, image, mask, case_identifier, cut_length=90): # 框选定为180 * 180
        # 如果想以CT肿瘤中心去裁剪矩形框也是可以做到的，可以在stage_seg阶段完善CT肿瘤边界框信息，读入pkl文件即可
        pkl = load_pickle(join(seg_data_2D, case_identifier+".pkl")) # 10474855_0_a_13.npz
        center = pkl["tumor_center"]
        resizer = (slice(max(0,int(center[1]-cut_length)), min(image.shape[0],int(center[1]+cut_length))),
                   slice(max(0,int(center[2]-cut_length)), min(image.shape[1],int(center[2]+cut_length))))
        return image[resizer], mask[resizer]

    def do_resize(self, image, mask, target_length=448, padding=False): # 锁定y高度320 或者锁定x宽度448
        height, width = image.shape # 获取原始图像的高度和宽度
        aspect_ratio = width / height # 计算纵横比 并保证长的边resize到224 短的边补黑边
        if width >= height:
            target_height = int(target_length / aspect_ratio) # 根据目标高度计算宽度
            resized_image = cv2.resize(image, (target_length, target_height)) # 使用cv2.resize函数进行缩放
            resized_mask = cv2.resize(mask, (target_length, target_height))
        else:
            target_width = int(target_length * aspect_ratio) # 根据目标高度计算宽度
            resized_image = cv2.resize(image, (target_width, target_length)) # 使用cv2.resize函数进行缩放
            resized_mask = cv2.resize(mask, (target_width, target_length))
        if aspect_ratio != 1 or padding: # 正方裁剪下无需padding但是如果纵横比不为1则需要
            # 创建新的空白图像
            padded_image = np.full((target_length, target_length), min(image[0][0], image[-1][-1]), dtype=np.float32)
            padded_mask = np.full((target_length, target_length), mask[0][0], dtype=np.float32)
            # 将调整大小后的图像粘贴到新的空白图像中心
            target_height, target_width = resized_image.shape
            x = (target_length - target_width) // 2 # 这里是不用补的
            y = (target_length - target_height) // 2
            padded_image[y:y+target_height, x:x+target_width] = resized_image
            padded_mask[y:y+target_height, x:x+target_width] = resized_mask

            padded_mask[padded_mask != 1] = 0 #False
            return padded_image, padded_mask
        else:
            resized_mask[resized_mask != 1] = 0
            return resized_image, resized_mask

class myCT_Seq_Dataset(Dataset):
    def __init__(self, data, label, transform=None):
        super().__init__()
        self.data = data
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.label) # 数据集长度

    def __getitem__(self, index): # 根据索引返回对应的图像和标签
        image, label = self.data[index], self.label[index]
        return image, label

def collate_fn_src_mask(data_tuple):
    data_tuple.sort(key=lambda x: len(x[0]), reverse=True) # 按照序列的长度倒序排列
    data = [sq[0] for sq in data_tuple]
    label = [sq[1] for sq in data_tuple]
    year = [sq[2] for sq in data_tuple]
    length = [len(sq) for sq in data]
    max_len = max([len(sq) for sq in data])  # sq 确定batch中的最大序列长度

    for i in range(len(data)):
        data[i] = torch.tensor([item.numpy() for item in data[i]])
    # 将label扩展成序列的label
    sq_labels = []
    for i in range(len(label)):
        lb = torch.zeros(length[i], dtype=torch.long) # 创建一个长度为 n+1 的零向量 因为输入序列
        lb = torch.cat([lb, torch.full((max_len + 1 - lb.size(0),), -100, dtype=torch.long)])
        lb[-1] = label[i] # 将最后的序列应位置的值设置为原标签 用作最后计算预测值的gold
        sq_labels.append(lb)
    # 对所有序列、序列标签进行填充至最大长度
    padded_sqs = [torch.cat([sq, torch.zeros(max_len - sq.size(0), sq.size(1))]) for sq in data]
    padded_sqs = torch.stack(padded_sqs)  # 转换为一个batch的tensor
    padded_clinic = padded_sqs[:,:,-18:-1] # 企图只取最后几位特征位 -18:
    padded_diam = np.array(padded_sqs[:,:,2])
    padded_sqs = padded_sqs[:,:,:512] # 如果是结合前529维特征一块呢？
    # padded_lbs = [torch.cat([sq, torch.full((max_len + 1 - sq.size(0),), -100, dtype=torch.long)]) for sq in sq_labels]
    padded_lbs = torch.stack(sq_labels)
    padded_year = [sublist + [0]*(max_len - len(sublist)) for sublist in year]
    padded_year = np.array(padded_year)     # 转换为NumPy数组，然后转换为张量
    # 创建src_mask，以屏蔽填充的部分 可以在之后 使用label来创建掩码
    # src_mask = (padded_sqs != 0).unsqueeze(-2)  # 对于填充的0位置为False，非填充位置为True .unsqueeze(1).
    # 创建diam_matrix
    diam_matrix = []
    padded_diam = np.array(padded_sqs[:,:,-16])
    # 创建time_matrix
    time_matrix = []
    for index, li in enumerate(padded_year): # padded_year
        m = np.zeros((max_len, max_len), np.float32)
        if len(year[index]) > 1: # 如果序列长度短 2或3 则均等注意
            for i in range(m.shape[0]):
                for j in range(m.shape[1]):
                    if i>=j:
                        m[i][j] = max(li[i] - li[j], 0) / 60
                    else:
                        m[i][j] = max(li[j] - li[i], 0) / 60
        time_matrix.append(m)
        diam_matrix.append(m)
    time_matrix = torch.tensor(time_matrix)
    diam_matrix = torch.tensor(diam_matrix)
    padded_year = torch.from_numpy(padded_year)
    # 检查结果sq_
    # print("Padded Features:", padded_sqs)
    return length, padded_sqs, padded_lbs, padded_clinic, padded_year, time_matrix # diam_matrix #


def collate_fn_sequence(data_tuple):   # data_tuple是一个列表，列表中包含batchsize个元组，每个元组中包含数据和标签
    data_tuple.sort(key=lambda x: len(x[0]), reverse=True) # 按照序列的长度倒序排列
    data = [sq[0] for sq in data_tuple]
    label = [sq[1] for sq in data_tuple]
    seq_id = [sq[2] for sq in data_tuple]
    data_length = [len(sq) for sq in data]  # sq
    # [5, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2] 这是不经过提前排序的结果
    # print(data[0]) # 五个tensor
    # print(data_length)
    for i in range(len(data)):
        data[i] = torch.tensor([item.numpy() for item in data[i]])
    padded_data = rnn_utils.pad_sequence(data, batch_first=True, padding_value=0.0)     # 用零补充，使长度对齐
    # print("完成填充")
    # label = rnn_utils.pad_sequence(label, batch_first=True, padding_value=0.0)   # 这行代码只是为了把列表变为tensor
    label = torch.tensor(label)
    return data_length, padded_data, label, seq_id  # .unsqueeze(-1) padded_

# ******************************************** Feature tools ******************************************************** #
def eva_seg_feature(model, data_loader, device):
    # 运行推理 收集特征与图片名
    seg_features, files_name = [], []
    progress_bar = tqdm(total=len(data_loader), desc='Extracting Progress', unit='batch')
    with torch.no_grad():
        for file_name, images, labels in data_loader:
            labels = labels.type(torch.LongTensor)
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            seg_features.append(output.cpu())
            files_name.append(file_name)
            progress_bar.update(1)
        progress_bar.close()
        seg_features = torch.cat(seg_features, dim=0)
        files_name = [list(files) for files in files_name]
        slices_name = []
        for lst in files_name: slices_name += lst
    return seg_features, slices_name

def seg_slice2ct_feature(seg_features, slices_name):
    """
    这里可以知道slice name 那么就在这里完成文件读取 文件名匹配CT的检查日期的字典！！！ 6.4
    :param seg_features files_name: 【n, 960】类似的构成 file_name=id_tp_fup_slice.npz
    :param pkl 文件读取 stage_seg中得到case_identifier+pkl
    :return:
    """
    id_dict = collections.OrderedDict() # 按照插入顺序的字典
    lb_df = pd.read_csv('../Task_Pancreas/data_info/labels_CT.csv')
    lb_df.set_index(['id'],inplace=True)
    n = 0
    for i, slice_name in enumerate(slices_name):
        info = slice_name.split("_")
        id, tp, fup, sl, case_identifier = info[0], info[1], info[2], info[3][:-4], info[0]+'_'+info[1]+'_'+info[2]
        pkl = load_pickle(join(seg_data_2D, case_identifier+'.pkl'))
        if np.where(pkl['tumor_index']==int(sl))[0].size == 0: # 更改了一些勾画 有些老旧的slice遗留在文件夹中 跳过
            continue
        index = np.where(pkl['tumor_index']==int(sl))[0][0]
        sqw = [area / sum(pkl['tumor_area']) for area in pkl['tumor_area']][index]
        feature = seg_features[i] * sqw # 这里对特征按照他们的面积权重占比已经做出了处理， 方便后续直接相加
        # feature = seg_features[i] # 这里不进行加权 后续相加后直接会求均值
        # print("本条slice特征为%s"%slice_name)
        if id+tp not in id_dict: # 这是一条新数据 [slice数，CT特征加权和，标签，最大面积 or 体积]
            label = lb_df.loc[int(id)][tp] # float
            id_dict[id+tp] = [1, feature, label, max(pkl['tumor_area'])] # 记录插入的是第k行（因为是k条新数据），是id人tp时间点的第1条向量
            n += 1
        else:  # 这是一条已经存在的数据 是某个病人某个时间点的数据 故直接相加 更新切片数量加1
            id_dict[id+tp][1] += feature
            id_dict[id+tp][0] += 1
        # print("总记录数量id+tp有%d条"% n)
    print("成功整合单模态CT特征")
    return id_dict

def seg_ct2seq_feature(id_dict, clinic_df=None, baseline_info_df=None, add_one=False):
    X, y, seq = [], [], []
    seq_id = [] # 每一条序列用最后这个时间点标识即可
    id_list = sorted(id_dict.keys()) # 有序的索引
    for key in id_list:
        # id_dict[key][1] /= id_dict[key][0]
        # print(f"{key}:{id_dict[key][0]} slices")
        # 构造不定长序列 在id+tp的每一条特征中，一行将变成 01 012 0123 01234
        id, tp = key[:-1], key[-1]
        if clinic_df is not None and not clinic_df.empty:
            baseline_info = np.zeros(14)
            if int(tp) is 0: # 0时间点 基线特征增加临床df
                baseline_info = baseline_info_df.loc[id].values
            # baseline_info = baseline_info[2:3] # 只取CA199
            fea = torch.cat((id_dict[key][1], torch.tensor(clinic_df.loc[id]['stage']).unsqueeze(0),
                             torch.tensor(clinic_df.loc[id]['diam'][int(tp)]).unsqueeze(0),
                             torch.tensor(id_dict[key][-1]).unsqueeze(0),
                             torch.tensor(baseline_info)
                             ), 0) # 529深度特征的构建
        else:
            # fea = torch.cat((id_dict[key][1], torch.tensor(id_dict[key][-1]).unsqueeze(0)), 0) # 513的构建
            fea = id_dict[key][1]
        # fea = torch.cat((torch.tensor(id_dict[key][1]), torch.tensor(id_dict[key][-1]).unsqueeze(0)), 0) # 无用 这里是组学特征先搞成tensor
        # fea = torch.from_numpy(fea) # 要转成tensor
        # fea = fea.to(torch.float32) # 以上 这三句略显多余的话是解决单独的组学特征
        if not int(tp): # 其实如果时间点是0 那么一定是一个新人
            seq = [] # 置空，迎接新人
        seq.append(fea)
        if int(tp) >= 1:
            # seq = np.array(seq)
            # print(seq.shape)
            # if seq.shape[0] >= 2:
            X.append(copy.deepcopy(seq))
            y.append(id_dict[key][2]) # 当前时间点的lb
            seq_id.append(id+"_"+tp)
            # seq = list(seq)
        if int(tp) == 0 and add_one:
            X.append(copy.deepcopy(seq))
            if id_dict[key][2] is 'Nan':
                y.append(0) # tp=0单时间点也加入
            else:
                y.append(id_dict[key][2])
            seq_id.append(id+"_"+tp)
    # print(X[1].shape) # 2
    # print(X[2])
    # 这里重新需要将【序列，标签】变成一个字典，然后根据剧烈长度进行排序 X[i]-y[i]
    # len_dict = {}
    # data, label = [], []
    # for i in range(len(X)):
    #     len_dict[i] = (X[i].shape[0], X[i], y[i])
    # len_list = sorted(len_dict.items(), key=lambda x:x[1][0], reverse=False) # 有序序列数据的索引
    # # print("接下来是打印 按照序列顺序排列的 那些序列的长度")
    # for i in range(len(len_list)):
    #     # print(len_list[i][0]) # 这个应该是有序list的记录的key值
    #     # print(f"{i}:{len_dict[len_list[i][0]][0]}") # 根据key值可以访问原dict的value中len元素
    #     data.append(len_dict[len_list[i][0]][1])
    #     label.append(len_dict[len_list[i][0]][2])
    label = torch.tensor(y, device = 'cpu')
    print("总序列条数：", len(label))
    return X, label, seq_id

def seg_ct2seq_feature_op(id_dict, clinic_df):
    """
    仅构造单时间点序列 one point 实现构建长度为1的序列，只单纯使用机器学习的分类器
    但是基线不参与 对于入组时那些病人本来就是为0才入组的
    但不管怎样 这里整理成一个个的时间点的X和y和seq_id即可 但是后面投入什么 可以都尝试？
    """
    X, y, seq = [], [], []
    seq_id = [] # 每一条序列用最后这个时间点标识即可
    id_list = sorted(id_dict.keys()) # 有序的索引
    for key in id_list:
        id, tp = key[:-1], key[-1]
        if int(tp) == 0: continue # 这里不采纳基线
        fea = torch.cat((id_dict[key][1],
                         # torch.tensor(clinic_df.loc[id]['stage']).unsqueeze(0),
                         # torch.tensor(clinic_df.loc[id]['diam'][int(tp)]).unsqueeze(0),
                         # torch.tensor(id_dict[key][-1]).unsqueeze(0)
                         ), 0) # 515深度特征的构建 或者更少
        X.append(fea.numpy())
        y.append(id_dict[key][2]) # 当前时间点的lb
        seq_id.append(id+"_"+tp)
    # label = torch.tensor(y, device = 'cpu')
    print("总特征时间点数：", len(y))
    return X, y, seq_id

def seg_ct2seq_delta_feature(id_dict, clinic_df=None):
    clinic_baseline_df = process_clinic_baseline('../Task_Pancreas/data_info/pancreas_baseline_clinical_info.xlsx')
    if clinic_df is None:
        clinic_df = process_clinic_feature('../Task_Pancreas/data_info/stage.xlsx', '../Task_Pancreas/data_info/SD_diameter_sum.xlsx')
    delta_year_df = pd.read_csv('../Task_Pancreas/data_info/delta_year_PR.csv', index_col=0)
    X, y, delta_year, seq = [], [], [], []
    id_list = sorted(id_dict.keys()) # 有序的索引
    for key in id_list:
        # id_dict[key][1] /= id_dict[key][0]
        print(f"{key}:{id_dict[key][0]} slices")
        # 构造不定长序列 在id+tp的每一条特征中，一行将变成 01 012 0123 01234
        id, tp = key[:-1], key[-1]
        baseline_info = -np.ones(14)
        if int(tp) is 0: # 0时间点 基线特征增加临床df
            baseline_info = clinic_baseline_df.loc[id].values
        fea = torch.cat((id_dict[key][1], torch.tensor(clinic_df.loc[id]['stage']).unsqueeze(0),
                         torch.tensor(id_dict[key][-1]).unsqueeze(0),
                         torch.tensor(clinic_df.loc[id]['diam'][int(tp)]).unsqueeze(0), # -16
                         torch.tensor(baseline_info).float(),
                         torch.tensor(-1).unsqueeze(0)), 0) # 深度特征的构建 513
        # 为了后续方便这里去掉了首位特征
        if not int(tp): # 其实如果时间点是0 那么一定是一个新人
            seq = [] # 置空，迎接新人
        seq.append(fea)
        if int(tp) >= 1:
            seq = np.array(seq)
            print(seq.shape)
            if seq.shape[0] >= 2:
                X.append(seq) # 这里是开始加入序列了
                y.append(id_dict[key][2]) # 当前时间点的lb
                delta_year.append(delta_year_df.loc[int(id)][:int(tp)+1].tolist())
            seq = list(seq)
    # 这里重新需要将【序列，标签】变成一个字典，然后根据剧烈长度进行排序 X[i]-y[i]-delta_year[i] 分别表示的是一条序列的数据标签和时间差
    # len_dict = {}
    # data, label, year = [], [], []
    # for i in range(len(X)):
    #     len_dict[i] = (X[i].shape[0], X[i], y[i], delta_year[i])
    # len_list = sorted(len_dict.items(), key=lambda x:x[1][0], reverse=False) # 有序序列数据的索引
    # print("接下来是打印 按照序列顺序排列的 那些序列的长度")
    # for i in range(len(len_list)):
    #     print(len_list[i][0]) # 这个应该是有序list的记录的key值
    #     print(f"{i}:{len_dict[len_list[i][0]][0]}") # 根据key值可以访问原dict的value中len元素
    #     data.append(len_dict[len_list[i][0]][1])
    #     label.append(len_dict[len_list[i][0]][2])
    #     year.append(len_dict[len_list[i][0]][3])
    # label = torch.tensor(label, device = 'cpu')
    print("总序列条数：", len(y))
    return X, y, delta_year # data, label, year #

# ******************************************** training tools ******************************************************** #
def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.05, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.05, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                    1.0 + math.cos(math.pi* (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)
    return func

def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def CE_Loss(inputs, target, num_classes=0):
    # n, c, h, w = inputs.size()
    # nt, ct, ht, wt = target.size()
    # temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    # temp_target = target.view(-1)
    # 忽略背景类的分割损失 设置忽略0
    target = target.squeeze(dim=1)
    CE_loss  = nn.CrossEntropyLoss(ignore_index=num_classes)(inputs, target)

    return CE_loss

def Dice_loss(inputs, target, smooth = 1e-8):
    n, c, h, w = inputs.size()
    #--------------------------------------------#
    #   计算dice loss
    #--------------------------------------------#
    preds = inputs[:,1,:,:].view(n, c, -1) # 只取肿瘤类
    labels = target.view(n, c, -1)
    intersection = torch.sum(preds * labels)
    dice = (2. * intersection + smooth) / (torch.sum(preds) + torch.sum(labels) + smooth)

    dice_loss = 1 - torch.mean(dice)
    return dice_loss

def Focal_Loss(inputs, target, num_classes=0, alpha=0.25, gamma=2):
    # n, c, h, w = inputs.size()
    target = target.squeeze(dim=1) # 这一行在进行分割的时候要解开注释 ！！！ IMPORTANT ignore_index=num_classes,
    # temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    # temp_target = target.view(-1)
    # F.binary_cross_entropy_with_logits(inputs[:, 1], target_mask.float())
    logpt  = -nn.CrossEntropyLoss(ignore_index=num_classes, reduction='none')(inputs, target) #  reduction='none'
    pt = torch.exp(logpt)
    if alpha is not None:
        logpt *= alpha
    loss = -((1 - pt) ** gamma) * logpt
    loss = loss.mean() * 1e3
    return loss

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

def draw_tr_val_fig(train_losses, train_dice_scores, val_losses, val_dice_scores):
    # 绘制损失曲线
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # 绘制Dice均值曲线
    plt.plot(train_dice_scores, label='Train Auc')
    plt.plot(val_dice_scores, label='Validation Auc')
    plt.xlabel('Epoch')
    plt.ylabel('Auc Score')
    plt.legend()
    plt.show()