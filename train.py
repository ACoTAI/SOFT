import os
import sys
import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import time
import yaml
import argparse
import numpy as np
from printer import Printer
from dataset import get_data_loader
from model import Model
import datetime
import copy
from util import make_dir, get_optimizer, AverageMeter, save_train_info, norm_flow
from gyro import torch_QuaternionProduct, torch_QuaternionReciprocal, torch_norm_quat
from tensorboardX import SummaryWriter
from simCLR import Model_sim, Model_resnet50
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


write = SummaryWriter()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

train_simCLR_transform = transforms.Compose([
    # transforms.ToPILImage(),
    # transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

def simCLR_run_epoch(model, loader, cf, epoch, epochs, optimizer=None,temperature=0.5, USE_CUDA=True):
    no_flo = cf['train']["no_flow"]
    total_num = 0
    total_loss = 0
    print('SimCLR Train Epoch: [{}/{}]'.format(epoch, epochs))
    for i, data in enumerate(loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        real_inputs, times, flo, flo_back, real_projections, real_postion, ois, real_queue_idx, number_train = data
        print("Fininsh Load data")
        real_inputs = real_inputs.type(torch.float)  # [b,60,84=21*4]

        batch_size, step, dim = real_inputs.size()

        for j in range(step):
            flo_step = flo[:, j]
            flo_back_step = flo_back[:, j]
            b, h, w, _ = flo_step.size()
            flo_step = norm_flow(flo_step, h, w)
            flo_back_step = norm_flow(flo_back_step, h, w)
            # flow = np.concatenate((flo_step, flo_back_step), dim=3)

            flow = torch.cat((flo_step, flo_back_step), dim=3)
            # flow = flow.numpy()
            while True:
                flow_1 = train_simCLR_transform(Image.fromarray(np.uint8(flow.numpy()[0]))).unsqueeze(0)
                if flow_1.shape[1] == 4:
                    break
                else:
                    continue
            while True:
                flow_2 = train_simCLR_transform(Image.fromarray(np.uint8(flow.numpy()[0]))).unsqueeze(0)
                if flow_2.shape[1] == 4:
                    break
                else:
                    continue
            # flow_1 = train_simCLR_transform(Image.fromarray(np.uint8(flow.numpy()[0]))).unsqueeze(0)
            # flow_2 = train_simCLR_transform(Image.fromarray(np.uint8(flow.numpy()[0]))).unsqueeze(0)
            for batch in range(1, batch_size):
                while True:
                    temp = train_simCLR_transform(Image.fromarray(np.uint8(flow.numpy()[batch])))
                    if temp.shape[0] == 4:
                        flow_1 = torch.cat((flow_1, temp.unsqueeze(0)), dim=0)
                        break
                    else:
                        continue
                while True:
                    temp = train_simCLR_transform(Image.fromarray(np.uint8(flow.numpy()[batch])))
                    if temp.shape[0] == 4:
                        flow_2 = torch.cat((flow_2, temp.unsqueeze(0)), dim=0)
                        break
                    else:
                        continue

            # flow_1 = train_simCLR_transform(flow)
            # flow_2 = train_simCLR_transform(flow)
            # print(flow_1)
            feature_1, out_1 = model(flow_1.cuda())
            feature_2, out_2 = model(flow_2.cuda())
            # [2*B, D]
            out = torch.cat([out_1, out_2], dim=0)
            # [2*B, 2*B]
            sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
            mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device))  #.type(torch.bool)#bool()
            # [2*B, 2*B-1]
            # sim_matrix = torch.masked_select(sim_matrix, mask).view(2 * batch_size, -1)
            sim_matrix = sim_matrix.masked_select(mask.type(torch.uint8)).view(2 * batch_size, -1)
            # sum_sim_matrix = torch.zeros_like(sim_matrix[0])
            # for column in range(sim_matrix.shape[1]):
            #     for row in range(sim_matrix.shape[0]):
            #         if column != row:
            #             sum_sim_matrix[row] += sim_matrix[row][column]

            # compute loss
            pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
            # [2*B]
            pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
            loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_num += batch_size
            total_loss += loss.item() * batch_size
            # train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))
            if (j + 1) % 10 == 0:
                print("Step: " + str(j + 1) + "/" + str(step))
                print(total_loss / total_num)
                print(loss)
    # print('Loss: {:.4f}'.format(total_loss / total_num))
    # return total_loss / total_num
    return model


def run_epoch(model, model_simclr, loader, cf, epoch, lr, optimizer=None, is_training=True, USE_CUDA=True, clip_norm=0):
    no_flo = cf['train']["no_flow"]
    number_virtual, number_real = cf['data']["number_virtual"], cf['data']["number_real"]
    # is_only_train_simCLR = cf['flow_net']["simCLR"]
    # batch_simCLR = cf['simCLR']["batch_simCLR"]
    avg_loss = AverageMeter()
    if is_training:
        model.net.train()
        model.unet.train()
    else:
        model.net.eval()
        model.unet.eval()
    if epoch <= 30:
        follow = True
    else:
        follow = False

    if epoch > 30:
        undefine = True
    else:
        undefine = False

    if epoch > 40:
        optical = True
    else:
        optical = False

    for i, data in enumerate(loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        # print(i)
        real_inputs, times, flo, flo_back, real_projections, real_postion, ois, real_queue_idx, number_train = data
        print("Fininsh Load data")

        real_inputs = real_inputs.type(torch.float)  # [b,60,84=21*4]
        real_projections = real_projections.type(torch.float)
        flo = flo.type(torch.float)
        flo_back = flo_back.type(torch.float)
        ois = ois.type(torch.float)
        batch_size, step, dim = real_inputs.size()
        times = times.numpy()
        real_queue_idx = real_queue_idx.numpy()
        virtual_queue = loader.dataset.random_init_virtual_queue(batch_size, real_postion[:, 0, :].numpy(),
                                                                 times[:, 1])  # TODO
        # virtual_queue = [None] * batch_size
        loss = 0
        if not cf['transformer_or_rnn']['transformer']:
            model.net.init_hidden(batch_size)

        for j in range(step):
            virtual_inputs, vt_1 = loader.dataset.get_virtual_data(
                virtual_queue, real_queue_idx, times[:, j], times[:, j + 1], times[:, 0], batch_size, number_virtual,
                real_postion[:, j])

            real_inputs_step = real_inputs[:, j, :]
            inputs = torch.cat((real_inputs_step, virtual_inputs), dim=1)


            if no_flo is False:
                flo_step = flo[:, j]
                flo_back_step = flo_back[:, j]
                b, h, w, _ = flo_step.size()
                flo_step = norm_flow(flo_step, h, w)
                flo_back_step = norm_flow(flo_back_step, h, w)
                flow = torch.cat((flo_step, flo_back_step), dim=3)
                if cf['simCLR']["is_use"]:
                    while True:
                        flow_1 = train_simCLR_transform(Image.fromarray(np.uint8(flow.numpy()[0]))).unsqueeze(0)
                        if flow_1.shape[1] == 4:
                            break
                        else:
                            continue
                    for batch in range(1, batch_size):
                        while True:
                            temp = train_simCLR_transform(Image.fromarray(np.uint8(flow.numpy()[batch])))
                            if temp.shape[0] == 4:
                                flow_1 = torch.cat((flow_1, temp.unsqueeze(0)), dim=0)
                                break
                            else:
                                continue

            # inputs = Variable(real_inputs_step)
            if USE_CUDA:
                real_inputs_step = real_inputs_step.cuda()
                virtual_inputs = virtual_inputs.cuda()
                inputs = inputs.cuda()
                if no_flo is False:
                    flow_1 = flow_1.cuda()
                    flo_step = flo[:, j].cuda()
                    flo_back_step = flo_back[:, j].cuda()
                else:
                    flo_step = None
                    flo_back_step = None
                vt_1 = vt_1.cuda()
                real_projections_t = real_projections[:, j + 1].cuda()
                real_projections_t_1 = real_projections[:, j].cuda()
                real_postion_anchor = real_postion[:, j].cuda()
                ois_step = ois[:, j].cuda()



            if is_training:
                if no_flo is False:
                    flo_out = model.unet(flow_1)
                else:
                    flo_out = None

                if j < 1:
                    for i in range(2):
                        out = model.net(inputs, flo_out, ois_step)
                else:
                    out = model.net(inputs, flo_out, ois_step)
            else:
                with torch.no_grad():
                    if no_flo is False:
                        # flow_step = torch.cat((flo_step, flo_back_step), dim=3)
                        flo_out = model.unet(flow_1)
                    else:
                        flo_out = None

                    if j < 1:
                        for i in range(2):
                            out = model.net(inputs, flo_out, ois_step)
                    else:
                        out = model.net(inputs, flo_out, ois_step)

            loss_step = model.loss(out, vt_1, virtual_inputs, real_inputs_step, \
                                   flo_step, flo_back_step, real_projections_t, real_projections_t_1,
                                   real_postion_anchor, \
                                   follow=follow, undefine=undefine, optical=optical, stay=optical)

            loss = loss_step

            virtual_position = virtual_inputs[:, -4:]
            pos = torch_QuaternionProduct(virtual_position, real_postion_anchor)
            out = torch_QuaternionProduct(out, pos)

            if USE_CUDA:
                out = out.cpu().detach().numpy()

            virtual_queue = loader.dataset.update_virtual_queue(batch_size, virtual_queue, out, times[:, j + 1])

            if (j + 1) % 10 == 0:
                print("Step: " + str(j + 1) + "/" + str(step))
                print(loss)
            loss = torch.sum(loss)
            if is_training:
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                # write.add_scalar('loss', loss, )
                if clip_norm:
                    nn.utils.clip_grad_norm_(model.net.parameters(), max_norm=clip_norm)
                    nn.utils.clip_grad_norm_(model.unet.parameters(), max_norm=clip_norm)
                optimizer.step()

            avg_loss.update(loss.item(), batch_size)


    return avg_loss.avg


def simclr(cf, model_simCLR, optimizer_simCLR, train_loader, start_epoch, end_epoch, model_dict, simclr_model_pth):
    for sim_epoch in range(start_epoch, end_epoch + 1):
        model_simCLR = simCLR_run_epoch(model_simCLR, train_loader, cf, sim_epoch, end_epoch,
                                        optimizer_simCLR)
        if sim_epoch % 20 == 0:
            model_dict['epoch'] = sim_epoch
            model_dict['model'] = model_simCLR.state_dict()
            model_dict['optimizer'] = optimizer_simCLR.state_dict()

            torch.save(model_dict, simclr_model_pth)
        # for name, param in model_simCLR.named_parameters():
        #     param.requires_grad = False

    print("--------start model train, simclr is Lock----------")
    return model_simCLR.eval()


def train(args=None):
    torch.autograd.set_detect_anomaly(True)
    config_file = args.config
    cf = yaml.load(open(config_file, 'r'))

    USE_CUDA = cf['data']["use_cuda"]
    seed = cf['train']["seed"]

    torch.manual_seed(seed)
    if USE_CUDA:
        torch.cuda.manual_seed(seed)

    checkpoints_dir = cf['data']['checkpoints_dir']
    epochs = cf["train"]["epoch"]
    snapshot = cf["train"]["snapshot"]
    decay_epoch = cf['train']['decay_epoch']
    init_lr = cf["train"]["init_lr"]
    lr_decay = cf["train"]["lr_decay"]
    lr_step = cf["train"]["lr_step"]
    clip_norm = cf["train"]["clip_norm"]
    load_model = cf["model"]["load_model"]

    checkpoints_dir = make_dir(checkpoints_dir, cf)

    if load_model is None:
        log_file = open(os.path.join(cf["data"]["log"], cf['data']['exp'] + '.log'), 'w+')
    else:
        log_file = open(os.path.join(cf["data"]["log"], cf['data']['exp'] + '.log'), 'a')
    printer = Printer(sys.stdout, log_file).open()

    print('----Print Arguments Setting------')
    for key in cf:
        print('{}:'.format(key))
        for para in cf[key]:
            print('{:50}:{}'.format(para, cf[key][para]))
        print('\n')


    is_use_simCLR = cf['simCLR']["is_use"]

    # if is_use_simCLR:




    # Define the model

    if is_use_simCLR:
        print("-----------Load Dataset----------")
        size = cf['simCLR']["batch_simCLR"]
        num_workers = cf['simCLR']["num_workers"]
        train_loader, test_loader = get_data_loader(cf, size, num_workers, no_flo=False)
        feature_dim = cf["simCLR"]["feature_dim"]
        start_epoch_simCLR = 1
        # model_simCLR = Model_sim(cf, feature_dim)
        model_simCLR = Model_resnet50(feature_dim)
        model_simCLR.cuda()
        optimizer_simCLR = optim.Adam(model_simCLR.parameters(), lr=1e-4, weight_decay=1e-6)

        epoch_train_simCLR = cf['simCLR']["epoch_train_simCLR"]
        model_simCLR.train()
        model_dict = {'epoch': 0, 'model': model_simCLR.state_dict(),  'optimizer': optimizer_simCLR.state_dict()}
        dir_simclr_model = './checkpoint_simCLR'
        simclr_model_pth = os.path.join(dir_simclr_model, 'checkpoint.pth')
        if not os.path.exists(dir_simclr_model):
            os.makedirs(dir_simclr_model)
            model_simCLR = simclr(cf, model_simCLR, optimizer_simCLR, train_loader, start_epoch_simCLR, epoch_train_simCLR, model_dict, simclr_model_pth)

        elif len(os.listdir(dir_simclr_model)) != 0:
            checkpoints = torch.load(simclr_model_pth)
            model_simCLR.load_state_dict(checkpoints['model'])
            start_epoch_simCLR = checkpoints['epoch']
            optimizer_simCLR = optimizer_simCLR.load_state_dict(checkpoints['optimizer'])
            if start_epoch_simCLR != epoch_train_simCLR:
                model_simCLR = simclr(cf, model_simCLR, optimizer_simCLR, train_loader, start_epoch_simCLR,
                                      epoch_train_simCLR, model_dict, simclr_model_pth)
        elif len(os.listdir(dir_simclr_model)) == 0:
            model_simCLR = simclr(cf, model_simCLR, optimizer_simCLR, train_loader, start_epoch_simCLR,
                                  epoch_train_simCLR, model_dict, simclr_model_pth)

    else:
        model_simCLR = None

    model = Model(cf, model_simCLR=model_simCLR)
    optimizer = get_optimizer(cf["train"]["optimizer"], model, init_lr, cf)


    for idx, m in enumerate(model.net.children()):
        print('{}:{}'.format(idx, m))
    for idx, m in enumerate(model.unet.children()):
        print('{}:{}'.format(idx, m))

    if load_model is not None:
        print("------Load Pretrined Model--------")
        checkpoint = torch.load(load_model)
        model.net.load_state_dict(checkpoint['state_dict'])
        model.unet.load_state_dict(checkpoint['unet'])
        print("------Resume Training Process-----")
        optimizer.load_state_dict(checkpoint['optim_dict'])
        epoch_load = checkpoint['epoch']
        print("Epoch load: ", epoch_load)
    else:
        epoch_load = 0


    if USE_CUDA:
        model.net.cuda()
        model.unet.cuda()
        if load_model is not None:
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
            for param in optimizer.param_groups:
                init_lr = param['lr']

    size = cf["data"]["batch_size"]
    print("-----------Load Dataset----------")
    # size = cf['simCLR']["batch_simCLR"]
    num_workers = cf["data"]["num_workers"]
    train_loader, test_loader = get_data_loader(cf, size, num_workers,  no_flo=False)

    print("----------Start Training----------")
    currentDT = datetime.datetime.now()
    print(currentDT.strftime(" %Y-%m-%d %H:%M:%S"))

    start_time = time.time()

    if lr_step:
        decay_epoch = list(range(1 + lr_step, epochs + 1, lr_step))

    lr = init_lr

    for count in range(epoch_load + 1, epochs + 1):
        if decay_epoch != None and count in decay_epoch:
            lr *= lr_decay
            for param in optimizer.param_groups:
                param['lr'] *= lr_decay


             # pass

        print("Epoch: %d, learning_rate: %.5f" % (count, lr))
        train_loss = run_epoch(model, model_simCLR, train_loader, cf, count, lr, optimizer=optimizer, clip_norm=clip_norm,
                               is_training=True, USE_CUDA=USE_CUDA)

        test_loss = run_epoch(model, model_simCLR,  test_loader, cf, count, lr, is_training=False, USE_CUDA=USE_CUDA)

        time_used = (time.time() - start_time) / 60
        print("Epoch %d done | TrLoss: %.4f | TestLoss: %.4f | Time_used: %.4f minutes" % (
            count, train_loss, test_loss, time_used))

        if count % snapshot == 0:
            save_train_info("epoch", checkpoints_dir, cf, model, count, optimizer)
            save_train_info("last", checkpoints_dir, cf, model, count, optimizer)
            print("Model stored at epoch %d" % count)

    currentDT = datetime.datetime.now()
    print(currentDT.strftime(" %Y-%m-%d %H:%M:%S"))
    print("------------End Training----------")
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Training model")
    parser.add_argument("--config", default="./conf/stabilzation_train.yaml", help="Config file.")
    args = parser.parse_args()
    train(args=args)
