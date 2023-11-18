import math

from collections import OrderedDict

import torch.nn as nn
import numpy as np
import util
import yaml
import os
from loss import C2_Smooth_loss, C1_Smooth_loss, Optical_loss, Undefine_loss, Angle_loss, Follow_loss, Stay_loss
from gyro import torch_norm_quat, torch_QuaternionProduct
import torch.nn.functional as F
import torch
from torch.nn import Parameter
from enum import IntEnum
from typing import *

from transformer import TransformerEncoder
from vit import ViT
from vit_single import ViTSingle
# from tokenlearner_pytorch import TokenLearner
from tokenlearner import TokenLearner
from simclr import SimCLR


Activates = {"sigmoid": nn.Sigmoid, "relu": nn.ReLU, "tanh": nn.Tanh}


class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2


class GRUCell(nn.Module):
    """
    An implementation of GRUCell.

    """

    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def init_hidden(self, batch_size):
        self.ht = torch.zeros((batch_size, self.hidden_size)).cuda()

    def forward(self, x):
        x = x.view(-1, x.size(1))

        gate_x = self.x2h(x)
        gate_h = self.h2h(self.ht)

        gate_x = gate_x.squeeze()
        gate_h = gate_h.squeeze()

        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)

        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + (resetgate * h_n))

        self.ht = newgate + inputgate * (self.ht - newgate)

        return self.ht

class LayerMogGRU(nn.Module):
    """
    An implementation of GRUCell.

    """

    def __init__(self, input_size, hidden_size, mog_iterations, bias=True):
        super(LayerMogGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.mog_iterations = mog_iterations
        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        # Define/initialize all tensors
        # self.Wih = Parameter(torch.Tensor(input_size, hidden_size * 3))
        # self.Whh = Parameter(torch.Tensor(hidden_size, hidden_size * 3))
        # self.bih = Parameter(torch.Tensor(hidden_size * 3))
        # self.bhh = Parameter(torch.Tensor(hidden_size * 3))
        # Mogrifiers
        self.Q = Parameter(torch.Tensor(hidden_size, input_size))
        self.R = Parameter(torch.Tensor(input_size, hidden_size))

        self.reset_parameters()

    def mogrify(self, xt):
        for i in range(1, self.mog_iterations + 1):
            if (i % 2 == 0):
                self.ht = (2 * torch.sigmoid(xt @ self.R)) * self.ht
            else:
                xt = (2 * torch.sigmoid(self.ht @ self.Q)) * xt
        return xt, self.ht

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def init_hidden(self, batch_size):
        self.ht = torch.zeros((batch_size, self.hidden_size)).cuda()
        # self.Ct = torch.zeros((batch_size, self.hidden_size)).cuda()


    def forward(self, xt):
        # xt = x.view(-1, x.size(1))
        xt, self.ht = self.mogrify(xt)  # mogrification

        # gates = (xt @ self.Wih + self.bih) + (self.ht @ self.Whh + self.bhh)
        # ingate, cellgate, outgate = gates.chunk(3, 1)
        gate_x = self.x2h(xt)
        gate_h = self.h2h(self.ht)
        # gate = (gate_x + gate_h).squeeze()
        # gate_x = gate_x.squeeze()
        # gate_h = gate_h.squeeze()

        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)

        #cell
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + (resetgate * h_n))

        self.ht = newgate + inputgate * (self.ht - newgate)

        return self.ht

class LayerMogLSTM(nn.Module):
    def __init__(self, input_sz: int, hidden_sz: int, mog_iterations: int):
        super().__init__()
        self.input_size = input_sz
        self.hidden_size = hidden_sz
        self.mog_iterations = mog_iterations
        # Define/initialize all tensors
        self.Wih = Parameter(torch.Tensor(input_sz, hidden_sz * 4))
        self.Whh = Parameter(torch.Tensor(hidden_sz, hidden_sz * 4))
        self.bih = Parameter(torch.Tensor(hidden_sz * 4))
        self.bhh = Parameter(torch.Tensor(hidden_sz * 4))
        # Mogrifiers
        self.Q = Parameter(torch.Tensor(hidden_sz, input_sz))
        self.R = Parameter(torch.Tensor(input_sz, hidden_sz))

        self.init_weights()

    def init_hidden(self, batch_size):
        self.ht = torch.zeros((batch_size, self.hidden_size)).cuda()
        self.Ct = torch.zeros((batch_size, self.hidden_size)).cuda()
        # self.ht = torch.zeros((batch_sz, self.hidden_size)).to(x.device)
        # # self.Ct = torch.zeros((batch_sz, self.hidden_size)).to(x.device)


    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def mogrify(self, xt):
        for i in range(1, self.mog_iterations + 1):
            if (i % 2 == 0):
                self.ht = (2 * torch.sigmoid(xt @ self.R)) * self.ht
            else:
                xt = (2 * torch.sigmoid(self.ht @ self.Q)) * xt
        return xt, self.ht

    # Define forward pass through all LSTM cells across all timesteps.
    # By using PyTorch functions, we get backpropagation for free.
    def forward(self, x: torch.Tensor,
                # init_states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor]:
                           #Tuple[torch.Tensor, torch.Tensor]]:
        """Assumes x is of shape (batch, sequence, feature)"""
        # is_moglstmCell = False
        try:
            batch_sz, seq_sz, _ = x.size()
            is_moglstmCell = False
        except:
            batch_sz, _ = x.size()
            seq_sz = 1
            is_moglstmCell = True

        # hidden_seq = []
        # ht and Ct start as the previous states and end as the output states in each loop below
        # if init_states is None:
        #     self.init_hidden(batch_sz)
            # self.ht = torch.zeros((batch_sz, self.hidden_size)).to(x.device)
            # self.Ct = torch.zeros((batch_sz, self.hidden_size)).to(x.device)
        # else:
        #     self.ht, self.Ct = init_states
        for t in range(seq_sz):  # iterate over the time steps
            if is_moglstmCell:
                xt = x
            else:
                xt = x[:, t, :]
            xt, self.ht = self.mogrify(xt)  # mogrification
            gates = (xt @ self.Wih + self.bih) + (self.ht @ self.Whh + self.bhh)
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ### The LSTM Cell!
            ft = torch.sigmoid(forgetgate)
            it = torch.sigmoid(ingate)
            Ct_candidate = torch.tanh(cellgate)
            ot = torch.sigmoid(outgate)
            # outputs
            self.Ct = (ft * self.Ct) + (it * Ct_candidate)
            self.ht = ot * torch.tanh(self.Ct)
            ###

            # hidden_seq.append(self.ht.unsqueeze(Dim.batch))
        # hidden_seq = torch.cat(hidden_seq, dim=Dim.batch)
        # # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        # hidden_seq = hidden_seq.transpose(Dim.batch, Dim.seq).contiguous()
        return self.ht






class LayerLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, bias):
        super(LayerLSTM, self).__init__()
        self.LSTM = nn.LSTMCell(input_size, hidden_size, bias)
        self.hidden_size = hidden_size
    
    def init_hidden(self, batch_size):
        self.hx = torch.zeros((batch_size, self.hidden_size)).cuda()
        self.cx = torch.zeros((batch_size, self.hidden_size)).cuda()

    def forward(self, x):
        self.hx, self.cx = self.LSTM(x, (self.hx, self.cx))
        return self.hx
        

class LayerCNN(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, pooling_size=None, 
                        activation_function=nn.ReLU, batch_norm=True):
        super(LayerCNN, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm2d(out_channel) if batch_norm else None
        self.activation = activation_function(inplace=True)
        if pooling_size is not None:
            self.pooling = nn.MaxPool2d(pooling_size)
        else:
            self.pooling = None
        
    def forward(self, x):
        x = self.conv(x)     #x->[batch,channel,height,width]
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x = self.activation(x)
        if self.pooling is not None:
            x = self.pooling(x)
        return x

class LayerFC(nn.Module):
    def __init__(self, in_features, out_features, bias, drop_out=0, activation_function=nn.ReLU, batch_norm = False):
        super(LayerFC, self).__init__()
        self.fc = nn.Linear(in_features, out_features, bias=bias)
        # self.activation = activation_function(inplace=True) if activation_function is not None else None
        self.activation = activation_function() if activation_function is not None else None
        self.dropout = nn.Dropout(p=drop_out,inplace=False) if drop_out else None
        self.batch_norm = nn.BatchNorm1d(out_features) if batch_norm else None
        
    def forward(self, x):
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.fc(x)
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

class Net(nn.Module):
    def __init__(self, cf):
        super(Net, self).__init__()
        self.cnn_param = cf["model"]["cnn"]
        self.rnn_param = cf["model"]["rnn"]
        self.fc_param = cf["model"]["fc"]
        self.unit_size = 4
        self.no_flo = cf['train']["no_flow"]

        if self.no_flo is False:
            self._rnn_input_size = (2*cf["data"]["number_real"]+1+cf["data"]["number_virtual"]) * 4 + 64
        else:
            self._rnn_input_size = (2*cf["data"]["number_real"]+1+cf["data"]["number_virtual"]) * self.unit_size

        #CNN Layers
        cnns = []
        cnn_activation = Activates[self.cnn_param["activate_function"]]
        cnn_batch_norm = self.cnn_param["batch_norm"]
        cnn_layer_param = self.cnn_param["layers"]
        if cnn_layer_param is not None:
            cnn_layers = len(cnn_layer_param)
            for layer in range(cnn_layers):
                in_channel = eval(cnn_layer_param[layer][0])[0]
                out_channel = eval(cnn_layer_param[layer][0])[1]
                kernel_size = eval(cnn_layer_param[layer][1])
                stride = eval(cnn_layer_param[layer][2])
                padding = eval(cnn_layer_param[layer][3])
                pooling_size = eval(cnn_layer_param[layer][4])

                cnn = None
                cnn = LayerCNN(in_channel, out_channel, kernel_size, stride, padding, pooling_size, 
                            activation_function=cnn_activation, batch_norm=cnn_batch_norm)
                cnns.append(('%d' % layer, cnn))
        
                self._rnn_input_size = int(math.floor((self._rnn_input_size+2*padding[1]-kernel_size[1])/stride[1])+1)
                if pooling_size is not None:
                    self._rnn_input_size = int(math.floor((self._rnn_input_size-pooling_size[1])/pooling_size[1])+1)
            self.convs = nn.Sequential(OrderedDict(cnns))

        else:
            self.convs = None
            out_channel = cf["data"]["channel_size"]
            
        self.gap = nn.AvgPool2d(self._rnn_input_size) if self.cnn_param["gap"] else None
        self._rnn_input_size = out_channel if self.cnn_param["gap"] else out_channel*(self._rnn_input_size)
        #mo lstm
        MogLSTM = cf['lstm']["moglstm"]
        MogGRU = cf['lstm']["mogGRU"]
        self.is_transformer = cf['transformer_or_rnn']["transformer"]
        if self.is_transformer:
             # self.transformer = TransformerEncoder(seq_len=23) #TODO
            self.transformers = nn.Sequential(TransformerEncoder(seq_len=23))
            # self._fc_input_size = 92
            #RNN Layers
        else:
            rnns = []
            rnn_layer_param = self.rnn_param["layers"]
            rnn_layers = len(rnn_layer_param)

            for layer in range(rnn_layers):
                if layer:
                    if MogLSTM:
                        rnn = LayerMogLSTM(rnn_layer_param[layer - 1][0], rnn_layer_param[layer][0], 2)
                    elif MogGRU:
                        rnn = LayerMogGRU(rnn_layer_param[layer - 1][0], rnn_layer_param[layer][0], 2)
                    else:
                        rnn = LayerLSTM(rnn_layer_param[layer-1][0], rnn_layer_param[layer][0], rnn_layer_param[layer][1])
                else:
                    if MogLSTM:
                        rnn = LayerMogLSTM(self._rnn_input_size, rnn_layer_param[layer][0], 2)
                    elif MogGRU:
                        rnn = LayerMogGRU(self._rnn_input_size, rnn_layer_param[layer][0], 2)
                    else:
                        rnn = LayerLSTM(self._rnn_input_size, rnn_layer_param[layer][0], rnn_layer_param[layer][1])
                rnns.append(('%d'%layer, rnn))
            self.rnns = nn.Sequential(OrderedDict(rnns))

            self._fc_input_size = rnn_layer_param[rnn_layers-1][0] #* 2 # ois

        #transformer
        # self.transformers = Transformer()


        #FC Layers
        fcs = []
        fc_activation = Activates[self.fc_param["activate_function"]]
        fc_batch_norm = self.fc_param["batch_norm"]
        fc_layer_param = self.fc_param["layers"]
        fc_drop_out = self.fc_param["drop_out"]
        fc_layers = len(fc_layer_param)
        
        if fc_layers == 1:
            fc = LayerFC(self._fc_input_size,fc_layer_param[0][0],fc_layer_param[0][1],
                    fc_drop_out, None, fc_batch_norm)
            fcs.append(('%d'%(fc_layers-1), fc))
        else:
            for layer in range(fc_layers-1):
                if layer:
                    fc = LayerFC(fc_layer_param[layer-1][0],fc_layer_param[layer][0],fc_layer_param[layer][1],
                        fc_drop_out, fc_activation, fc_batch_norm)
                else:
                    fc = LayerFC(self._fc_input_size,fc_layer_param[layer][0],fc_layer_param[layer][1],
                        fc_drop_out,fc_activation, fc_batch_norm)
                fcs.append(('%d'%layer, fc))
            fc = LayerFC(fc_layer_param[fc_layers-2][0],fc_layer_param[fc_layers-1][0],fc_layer_param[fc_layers-1][1],
                        fc_drop_out,None, fc_batch_norm) # Modified
            fcs.append(('%d'%(fc_layers-1), fc))

        self.class_num = fc_layer_param[fc_layers-1][0]
        self.fcs = nn.Sequential(OrderedDict(fcs))

    def init_hidden(self, batch_size):
        for i in range(len(self.rnns)):
            self.rnns[i].init_hidden(batch_size)

    def forward(self, x, flo, ois):
        b,c = x.size()   #x->[batch,channel,height,width]
        if self.convs is not None:
            x = self.convs(x)
        if self.gap is not None:
            x = self.gap(x)
        x = x.view(b,-1)

        if self.is_transformer:
            x = x.view(b, -1, 4)
            x = self.transformers(x)
            x = x.view(b, -1)
        if self.no_flo is False:
            x = torch.cat((x, flo), dim=1)
        x = self.rnns(x)
        x = self.fcs(x) # [b, 4]
        x = torch_norm_quat(x)
        return x

class Model():
    def __init__(self, cf, model_simCLR):
        super().__init__()
        self.net = Net(cf)
        self.unet = UNet(cf, model_simCLR)
        self.init_weights(cf)
        
        self.loss_smooth = C1_Smooth_loss()
        self.loss_follow = Follow_loss()
        self.loss_c2_smooth = C2_Smooth_loss()
        self.loss_optical = Optical_loss()
        self.loss_undefine = Undefine_loss(ratio = 0.08)
        self.loss_angle = Angle_loss()
        self.loss_stay = Stay_loss()

        self.loss_smooth_w = cf["loss"]["smooth"]
        self.loss_angle_w = cf["loss"]["angle"]
        self.loss_follow_w = cf["loss"]["follow"]
        self.loss_c2_smooth_w = cf["loss"]["c2_smooth"]
        self.loss_undefine_w = cf["loss"]["undefine"]
        self.loss_opt_w = cf["loss"]["opt"]
        self.loss_stay_w = cf["loss"]["stay"]

        self.gaussian_weight = np.array([0.072254, 0.071257, 0.068349, 0.063764, 0.057856, 0.051058, 0.043824, 0.036585, 0.029705, 0.023457, 0.01801])

    def loss(
        self, out, vt_1, virtual_inputs, real_inputs, flo, flo_back, 
        real_projections_t, real_projections_t_1, real_postion_anchor, 
        follow = True, undefine = True, optical = True, stay = False
        ):
        unit_size = self.net.unit_size
        mid = real_inputs.size()[1]//(2*unit_size) 

        Rt = real_inputs[:,unit_size*(mid):unit_size*(mid)+4] 
        v_pos = torch_QuaternionProduct(out, virtual_inputs[:, -4:])
        r_pos = torch_QuaternionProduct(v_pos, real_postion_anchor)

        loss = torch.zeros(7).cuda()
        if self.loss_follow_w > 0 and follow:
            for i in range(-2,3):
                loss[0] += self.loss_follow_w * self.loss_follow(v_pos, real_inputs[:,unit_size*(i+mid):unit_size*(i+mid)+4], None)
        if self.loss_angle_w > 0 and follow:
            threshold = 6 / 180 * 3.1415926
            loss_angle, theta = self.loss_angle(v_pos, Rt, threshold = threshold)
            loss[1] = self.loss_angle_w * loss_angle
        if self.loss_smooth_w > 0:
            loss_smooth = self.loss_smooth(out)
            loss[2] = self.loss_smooth_w * loss_smooth
        if self.loss_c2_smooth_w > 0: 
            loss[3] = self.loss_c2_smooth_w * self.loss_c2_smooth(out, virtual_inputs[:, -4:], virtual_inputs[:, -8:-4])
        if self.loss_undefine_w > 0 and undefine:
            Vt_undefine = v_pos.clone() 
            for i in range(0, 10, 2):
                Rt_undefine = real_inputs[:,unit_size*(mid+i):unit_size*(mid+i)+4]
                loss_undefine_w = self.loss_undefine_w * self.gaussian_weight[i]
                loss[4] +=  loss_undefine_w * self.loss_undefine(Vt_undefine, Rt_undefine)
                Vt_undefine = torch_QuaternionProduct(out, Vt_undefine)
                Vt_undefine = torch_QuaternionProduct(out, Vt_undefine)
        if self.loss_opt_w > 0 and optical:
            loss[5] = self.loss_opt_w * self.loss_optical(r_pos, vt_1, flo, flo_back, real_projections_t, real_projections_t_1) 
        if self.loss_stay_w > 0 and stay:
            loss[6] = self.loss_stay_w * self.loss_stay(out) 
        return loss


    def init_weights(self, cf):
        for m in self.net.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d) or  isinstance(m, nn.Linear):
                if cf["train"]["init"] == "xavier_uniform":
                    nn.init.xavier_uniform_(m.weight.data)
                elif cf["train"]["init"] == "xavier_normal":
                    nn.init.xavier_normal_(m.weight.data)

        for m in self.unet.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d) or  isinstance(m, nn.Linear):
                if cf["train"]["init"] == "xavier_uniform":
                    nn.init.xavier_uniform_(m.weight.data)
                elif cf["train"]["init"] == "xavier_normal":
                    nn.init.xavier_normal_(m.weight.data)

    def save_checkpoint(self, epoch = 0, optimizer=None):
        package = {
                'cnn': self.net.cnn_param,
                'fc': self.net.fc_param,
                'state_dict': self.net.state_dict(),
                }
        if optimizer is not None:
            package['optim_dict'] = optimizer.state_dict()
        if self.unet is not None:
            package['unet'] = self.unet.state_dict()
        package["epoch"] = epoch
        return package


class UNet(nn.Module):
    def __init__(self, cf, simCLR_model=None, n_channels = 4, n_classes = 16, tokenlearn_num=8, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.is_cnn = cf['flow_net']["cnn"]
        self.is_cnn_vit = cf['flow_net']["cnn_vit"]
        self.is_vit = cf['flow_net']["vit"]
        self.tokenlearner_transforme = cf['flow_net']["tokenlearner_transforme"]
        if simCLR_model is not None:
            self.simclR_model = simCLR_model
            for name, param in self.simclR_model.named_parameters():
                param.requires_grad = False

            self.out_1 = nn.Linear(2048, 1024, bias=True)
            self.out_2 = nn.Linear(1024, 512, bias=True)
            self._fc_input_size = 512
        elif self.is_cnn:
            self.inc = DoubleConv(n_channels, 8)
            self.down1 = Down(8, 16)
            self.down2 = Down(16, 32)
            self.down3 = Down(32, 64)
            # factor = 2 if bilinear else 1
            self.down4 = Down(64, 128)
            self._fc_input_size = 128 * 1 * 1
        elif cf['flow_net']["cnn_vit"]:
            self.inc = DoubleConv(n_channels, 8)
            self.down1 = Down(8, 16)
            self.down2 = Down(16, 32)
            # self.down3 = Down(32, 64)
            self.vit = ViT(dim=128, image_size_h=16, image_size_w=30, patch_size=2, depth=3, heads=2, mlp_dim=128, channels=32)
            self._fc_input_size = 128
        elif cf['flow_net']["vit"]:
            self.vit = ViT(dim=128, image_size_h=270, image_size_w=480, patch_size=30, depth=3, heads=2, mlp_dim=128, channels=4)
            self._fc_input_size = 128
        elif cf['flow_net']["tokenlearner_transforme"]:
            token_seq = 8
            # self.tklr = TokenLearner(S=token_seq)
            self.vitsingle = ViTSingle(dim=128, is_first=True, depth=1, heads=8, mlp_dim=128, dropout=0.1)
            self.tklr = TokenLearner(S=token_seq)
            self.vit = ViTSingle(dim=128, is_first=False, depth=12, heads=8, mlp_dim=128)
            # self.transformer = TransformerEncoder(seq_len=token_seq, d_model=4, n_layers=6, n_heads=2)
            self._fc_input_size = 128
        self.fc = LayerFC(self._fc_input_size, 64, bias = True)

    def forward(self, x):
        # if x_back is not None:
        #     x = torch.cat((x,x_back), dim =3)

        b,c,h,w = x.size()
        if c != 4:
            x = x.permute(0, 3, 1, 2)
        if self.simclR_model is not None:
            feature, out = self.simclR_model(x)
            x1 = self.out_1(feature)
            x2 = nn.ReLU()(x1)
            x3 = self.out_2(x2)
            x5 = nn.ReLU()(x3)
        elif self.is_cnn:
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
        elif self.is_cnn_vit:
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            # x4 = self.down3(x3)
            x5 = self.vit(x3)
        elif self.is_vit:
            x5 = self.vit(x)
        elif self.tokenlearner_transforme:
            x = self.vitsingle(x)
            b, hh, c = x.shape
            h = 9
            w = 16
            x = x.view(b, h, w, c)
            x = self.tklr(x)
            x5 = self.vit(x)
            # x = x.view(-1, 18, 32)
            # h = int(math.sqrt(hh))
            # x = x.view(-1, 18, 32, c)
            # x = x.permute(0, 2, 3, 1)
            # device = x.device
            # for layer in self.tklr_layers:
            #     x1 = layer(x)
            # x5 = self.transformer(x1)
        x = torch.reshape(x5, (b, -1))
        x = self.fc(x)
        return x


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(4),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)