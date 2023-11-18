import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18
from vit_single import ViTSingle
from tokenlearner import TokenLearner


class Model_sim(nn.Module):
    def __init__(self, cf,  feature_dim=128):
        super(Model_sim, self).__init__()

        # self.f = []
        # for name, module in resnet50().named_children():
        #     if name == 'conv1':
        #         module = nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1, bias=False)
        #     if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
        #         self.f.append(module)
        # encoder
        self.f = resnet18()
        self.f.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        if cf['flow_net']["tokenlearner_transforme"]:
            token_seq = 8
            # self.tklr = TokenLearner(S=token_seq)
            self.vitsingle = ViTSingle(dim=128, is_first=True, depth=1, heads=8, mlp_dim=128, dropout=0.1)
            self.tklr = TokenLearner(S=token_seq)
            self.vit = ViTSingle(dim=128, is_first=False, depth=12, heads=8, mlp_dim=128)
            # self.transformer = TransformerEncoder(seq_len=token_seq, d_model=4, n_layers=6, n_heads=2)
            # self._fc_input_size = 128
            # self.fc = nn.Linear(self._fc_input_size, 64, bias=True)
            # self.f.append(self.fc)


        # self.f = ViTSingle()
        # projection head
        self.g_linear1 = nn.Linear(1024, 512, bias=False)
        self.batchnorm = nn.BatchNorm1d(512)
        self.g_relu = nn.ReLU(inplace=True)
        self.g_linear2 = nn.Linear(512, feature_dim, bias=True)

    def forward(self, x):
        x = self.vitsingle(x)
        b, hh, c = x.shape
        h = 9
        w = 16
        x = x.view(b, h, w, c)
        x = self.tklr(x)
        x = self.vit(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g_linear1(feature)
        out = self.batchnorm(out)
        out = self.g_relu(out)
        out = self.g_linear2(out)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)


class Model_resnet50(nn.Module):
    def __init__(self, feature_dim=128):
        super(Model_resnet50, self).__init__()

        self.f = []
        for name, module in resnet18().named_children():
            if name == 'conv1':
                module = nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        self.f.append(nn.Conv2d(512, 512, kernel_size=7, stride=7, padding=(0, 2), bias=False))
        self.f.append(nn.Conv2d(512, 512, kernel_size=(3, 7), stride=1, bias=False))
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)
