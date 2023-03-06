#   梁丹阳
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from torch.nn import init
class HorizontalMaxPool2d(nn.Module):
    def __init__(self):
        super(HorizontalMaxPool2d, self).__init__()
    def forward(self, x):
        inp_size = x.size()
        x=F.avg_pool2d(x,kernel_size=(9,1),stride=(1,1))
        return nn.functional.max_pool2d(input=x,kernel_size= (1, inp_size[3]))
class align_pcb(nn.Module):
    def __init__(self, num_classes=702, loss={'softmax'}, aligned=False, droupout=0,FCN=True,**kwargs):
        super(align_pcb, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)

        for mo in resnet50.layer4[0].modules():
            if isinstance(mo, nn.Conv2d):
                mo.stride = (1, 1)
        self.dropout=droupout
        self.FCN=FCN
        self.has_embedding=True
        self.num_classes = num_classes
        self.num_features = 2048
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        out_planes = 2048  # self.base.fc.in_features
        #self.local_conv = nn.Conv2d(out_planes, 128, kernel_size=1, padding=0, bias=False)  # 1*1卷积 可选  降维
        #init.kaiming_normal(self.local_conv.weight, mode='fan_out')
        self.classifier = nn.Linear(2048, num_classes)
        init.normal(self.classifier.weight, std=0.001)
        init.constant(self.classifier.bias, 0)

        self.feat_dim = 2048  # feature dimension
        self.aligned = aligned

        self.horizon_pool = HorizontalMaxPool2d()

        self.feat_bn2d = nn.BatchNorm2d(self.num_features)  # may not be used, not working on caffe
        init.constant(self.feat_bn2d.weight, 1)  # initialize BN, may not be used
        init.constant(self.feat_bn2d.bias, 0)  # iniitialize BN, may not be used

        ##---------------------------stripe1----------------------------------------------#
        self.instance0 = nn.Linear(self.num_features, self.num_classes)
        init.normal(self.instance0.weight, std=0.001)  # 对weight初始化参数值符合正态分布
        init.constant(self.instance0.bias, 0)  # 对bias 初始化值为常值
        ##---------------------------stripe1----------------------------------------------#
        ##---------------------------stripe1----------------------------------------------#
        self.instance1 = nn.Linear(self.num_features, self.num_classes)
        init.normal(self.instance1.weight, std=0.001)
        init.constant(self.instance1.bias, 0)
        ##---------------------------stripe1----------------------------------------------#
        ##---------------------------stripe1----------------------------------------------#
        self.instance2 = nn.Linear(self.num_features, self.num_classes)
        init.normal(self.instance2.weight, std=0.001)
        init.constant(self.instance2.bias, 0)
        ##---------------------------stripe1----------------------------------------------#
        ##---------------------------stripe1----------------------------------------------#
        self.instance3 = nn.Linear(self.num_features, self.num_classes)
        init.normal(self.instance3.weight, std=0.001)
        init.constant(self.instance3.bias, 0)
        ##---------------------------stripe1----------------------------------------------#
        ##---------------------------stripe1----------------------------------------------#
        self.instance4 = nn.Linear(self.num_features, self.num_classes)
        init.normal(self.instance4.weight, std=0.001)
        init.constant(self.instance4.bias, 0)
        ##---------------------------stripe1----------------------------------------------#
        ##---------------------------stripe1----------------------------------------------#
        self.instance5 = nn.Linear(self.num_features, self.num_classes)
        init.normal(self.instance5.weight, std=0.001)
        init.constant(self.instance5.bias, 0)
        ##---------------------------stripe1----------------------------------------------#
        self.instance6 = nn.Linear(self.num_features, self.num_classes)
        init.normal(self.instance6.weight, std=0.001)
        init.constant(self.instance6.bias, 0)
        ##---------------------------stripe1----------------------------------------------#
        ##---------------------------stripe1----------------------------------------------#
        self.instance7 = nn.Linear(self.num_features, self.num_classes)
        init.normal(self.instance7.weight, std=0.001)
        init.constant(self.instance7.bias, 0)
        ##---------------------------stripe1----------------------------------------------#
        self.drop = nn.Dropout(self.dropout)  # 有多少的神经元不被激活
        if self.aligned:
            self.bn = nn.BatchNorm2d(2048)
            self.relu = nn.ReLU(inplace=True)
            self.conv1 = nn.Conv2d(2048, 128, kernel_size=1, stride=1, padding=0, bias=True)
        # Append new layers
        # if self.has_embedding:
        #     self.feat = nn.Linear(out_planes, self.num_features, bias=False)
        #     self.feat_bn = nn.BatchNorm1d(self.num_features)
        #     init.kaiming_normal(self.feat.weight, mode='fan_out')
        # else:
        #     # Change the num_features to CNN output channels
        #     self.num_features = out_planes
        if self.dropout > 0:
            self.drop = nn.Dropout(self.dropout)


    def forward(self, x):
        x = self.base(x)  # torch.Size([32, 2048, 24, 8])        16*8
        xc = x

        if self.FCN:
            xc = F.avg_pool2d(xc, kernel_size=(11, 8), stride=(1, 8))  # torch.Size([32, 2048, 6, 1])
        # out0 = F.avg_pool2d(x, x.size()[2:])
        # out0 = out0.view(out0.size(0), -1)
        xc = self.drop(xc)

        xc = self.feat_bn2d(xc)  # BN2d
        xc = F.relu(xc)  # relu for local_conv feature

        xc = xc.chunk(6, 2)
        x0 = xc[0].contiguous().view(xc[0].size(0), -1)  # torch.Size([32, 128])
        x1 = xc[1].contiguous().view(xc[1].size(0), -1)
        x2 = xc[2].contiguous().view(xc[2].size(0), -1)
        x3 = xc[3].contiguous().view(xc[3].size(0), -1)
        x4 = xc[4].contiguous().view(xc[4].size(0), -1)
        x5 = xc[5].contiguous().view(xc[5].size(0), -1)
        # x6 = xc[6].contiguous().view(xc[6].size(0), -1)
        # x7 = xc[7].contiguous().view(xc[7].size(0), -1)

        c0 = self.instance0(x0)  # torch.Size([32, 751])
        c1 = self.instance1(x1)
        c2 = self.instance2(x2)
        c3 = self.instance3(x3)
        c4 = self.instance4(x4)
        c5 = self.instance5(x5)
        # c6 = self.instance6(x6)
        # c7 = self.instance7(x7)

        if not self.training:
            lf =self.horizon_pool(x) #F.avg_pool2d(x, kernel_size=(3, 8), stride=(3, 8)) #
        if self.aligned and self.training:
            lf = self.bn(x)
            lf = self.relu(lf)
            lf =  self.horizon_pool(lf) #F.avg_pool2d(x, kernel_size=(3, 8), stride=(3, 8)
            lf = self.conv1(lf)  # lf.shape=32*128*8*1
        if self.aligned or not self.training:
            lf = lf.view(lf.size()[0:3])
            lf = lf / torch.pow(lf, 2).sum(dim=1, keepdim=True).clamp(min=1e-12).sqrt()  # lf.shape=32*128*12
        x = self.feat_bn2d(x)
        x = F.avg_pool2d(x, x.size()[2:])
        f = x.view(x.size(0), -1)

        if not self.training:
            return f, lf
        y = self.classifier(f)

        if self.loss == {'softmax'}:
            return y, c0, c1, c2, c3, c4, c5,
        elif self.loss == {'metric'}:
            if self.aligned: return f, lf
            return f
        elif self.loss == {'softmax', 'metric'}:
            if self.aligned: return y, f, lf, c0, c1, c2, c3, c4, c5
            return y, f, c0, c1, c2, c3, c4, c5
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))