import torch
import torchvision.models as models
from torch import nn
import torch.nn.functional as F
from torch.autograd import grad


class cnnResNet(nn.Module):
    def __init__(self, fc_hidden1=1024, fc_hidden2=768, drop_p=0.3, CNN_embed_dim=512, isClassify=True):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(cnnResNet, self).__init__()
        self.fc_hidden1, self.fc_hidden2, self.CNN_embed_dim = fc_hidden1, fc_hidden2, CNN_embed_dim
        self.drop_p = drop_p
        self.isClassify = isClassify

        resnet = models.resnet50(pretrained=True if self.isClassify else False) # 152    .fc.in_features
        resnet.conv1= nn.Conv2d(1, 64, kernel_size = (7, 7), stride = (2, 2), padding=3, bias=False) # 改变通道数
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(resnet.fc.in_features, fc_hidden1) # 获取原来fc层的输入通道数
        self.bn1 = nn.BatchNorm1d(fc_hidden1, momentum=0.01)
        self.fc2 = nn.Linear(fc_hidden1, fc_hidden2)
        self.bn2 = nn.BatchNorm1d(fc_hidden2, momentum=0.01)
        self.fc3 = nn.Linear(fc_hidden2, CNN_embed_dim) # 特征提取层
        self.classify = nn.Linear(CNN_embed_dim, 2) # class num 最后的分类层

    def forward(self, x_2d): # 现在就是正常的 batch_size * 1 * 224 * 224
        x = self.resnet(x_2d)  # ResNet
        x = x.view(x.size(0), -1)             # flatten output of conv
        # FC layers
        x = self.bn1(self.fc1(x))
        x = F.relu(x)
        x = self.bn2(self.fc2(x))
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc3(x)
        # 分类
        if self.isClassify:
            x = self.classify(x)
        return x

class LSTM(nn.Module):
    def __init__(self, CNN_embed_dim=512, h_RNN_layers=2, h_RNN=256, h_FC_dim=128, drop_p=0.0, num_classes=2): # 2类
        super(LSTM, self).__init__()
        self.RNN_input_size = CNN_embed_dim
        self.h_RNN_layers = h_RNN_layers   # RNN hidden layers
        self.h_RNN = h_RNN                 # RNN hidden nodes
        self.h_FC_dim = h_FC_dim
        self.drop_p = drop_p
        self.num_classes = num_classes

        self.LSTM = nn.LSTM(
            input_size=self.RNN_input_size,
            hidden_size=self.h_RNN,
            num_layers=h_RNN_layers,
            batch_first=True  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.fc1 = nn.Linear(self.h_RNN, self.h_FC_dim)
        self.fc2 = nn.Linear(self.h_FC_dim, self.num_classes)

    def forward(self, x_RNN):
        # print("序列形状为%s"%(x_RNN.size(),)) # batch * timepoint * 512
        self.LSTM.flatten_parameters()
        RNN_out, _ = self.LSTM(x_RNN, None) # (h_n, h_c)
        """ h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size) """
        """ None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) """
        # 从这里拿到一条序列的256维特征 拿去t-SNE进行降维可视化 标签为0的一个颜色 标签为1的另一个颜色 返回去模型做堆叠收集
        # FC layers
        x = self.fc1(RNN_out[:, -1, :])   # choose RNN_out at the last time step 取最后时间点的输出
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc2(x) # 分类层
        return x #, RNN_out[:, -1, :]

# 定义一个函数来计算集成梯度
def integrated_gradients(model, inputs, baseline, steps=100):
    attribution = torch.zeros_like(inputs)
    delta = (inputs - baseline) / steps

    for i in range(steps):
        inputs_step = baseline + i * delta
        inputs_step.requires_grad = True
        pred = model(inputs_step)
        loss = torch.sum(pred)  # 这里可以根据需要选择不同的损失函数
        gradients = grad(loss, inputs_step, retain_graph=True)[0]
        attribution += gradients

    attribution = attribution / steps
    return attribution * delta