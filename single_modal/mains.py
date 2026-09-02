import argparse
import os
import sys
import random
import time
from pathlib import Path
import torch
import cv2
import math
import numpy as np
import torch.backends.cudnn as cudnn
from torch.optim import Adam
from torch.utils.data import DataLoader
from utils import *
import json
import scipy.io as scio
from data import HSTrainingData
from data import HSTestData
from SDL_N_ms import SDL
# loss
# from loss import HyLapLoss
from metrics import quality_assessment
from torch.autograd import Variable
import logging  # 导入 logging 模块
# global settings
resume = False
log_interval = 50
model_name = ''
test_data_dir = ''
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "single_modal"
# 配置日志记录
logging.basicConfig(filename='training_log_ca2_.txt', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def save_checkpoint(args, model, epoch):
    device = torch.device("cuda" if args.cuda else "cpu")
    # model.eval().cpu()
    checkpoint_model_dir = './checkpoints/'
    if not os.path.exists(checkpoint_model_dir):
        os.makedirs(checkpoint_model_dir)
    ckpt_model_filename = args.dataset_name + "_" + args.model_title + "_ca_" + str(epoch) + ".pth"
    ckpt_model_path = os.path.join(checkpoint_model_dir, ckpt_model_filename)
    # state = {"epoch": epoch, "model": model}
    # torch.save(state, ckpt_model_path)
    # model.cuda().train()
    torch.save(model, ckpt_model_path)  # os.path.join(model_path, 'model_%03d.pth' % (epoch + 1))
    print("Checkpoint saved to {}".format(ckpt_model_path))

def model_structure(model):
    blank = ' '
    print('-' * 90)
    print('|' + ' ' * 11 + 'weight name' + ' ' * 10 + '|' \
          + ' ' * 15 + 'weight shape' + ' ' * 15 + '|' \
          + ' ' * 3 + 'number' + ' ' * 3 + '|')
    print('-' * 90)
    num_para = 0
    type_size = 1  # 如果是浮点数就是4

    for index, (key, w_variable) in enumerate(model.named_parameters()):
        if len(key) <= 30:
            key = key + (30 - len(key)) * blank
        shape = str(w_variable.shape)
        if len(shape) <= 40:
            shape = shape + (40 - len(shape)) * blank
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para
        str_num = str(each_para)
        if len(str_num) <= 10:
            str_num = str_num + (10 - len(str_num)) * blank

        print('| {} | {} | {} |'.format(key, shape, str_num))
    print('-' * 90)
    print('The total number of parameters: ' + str(num_para))
    print('The parameters of Model {}: {:4f}M'.format(model._get_name(), num_para * type_size / 1000 / 1000))
    print('-' * 90)


def main():
    # parsers
    # main_parser = argparse.ArgumentParser(description="parser for SR network")
    # subparsers = main_parser.add_subparsers(title="subcommands", dest="subcommand")
    # train_parser = subparsers.add_parser("train", help="parser for training arguments")
    train_parser = argparse.ArgumentParser(description="parser for training arguments")
    train_parser.add_argument("--cuda", type=int, required=False, default=1,
                              help="set it to 1 for running on GPU, 0 for CPU")
    train_parser.add_argument("--subcommand", type=str, default="train", help="batch size, default set to 64")
    train_parser.add_argument("--batch_size", type=int, default=12, help="batch size, default set to 64")  # p:4,P:2, # 4
    train_parser.add_argument("--epochs", type=int, default=120, help="epochs, default set to 20")  # 150
    train_parser.add_argument("--n_feats", type=int, default=256, help="feature dimension, default: 256")
    train_parser.add_argument("--n_blocks", type=int, default=3, help="n_blocks, default set to 6")
    train_parser.add_argument("--n_subs", type=int, default=8, help="n_subs, default set to 8")
    train_parser.add_argument("--n_ovls", type=int, default=2, help="n_ovls, default set to 1")
    train_parser.add_argument("--n_scale", type=int, default=2, help="super-resolution scale")
    train_parser.add_argument("--use_share", type=bool, default=True, help="f_share, default set to 1")
    train_parser.add_argument("--dataset_name", type=str, default="pavia",
                              help="dataset name: pavia, cave, or Chikusei")
    train_parser.add_argument("--model_title", type=str, default="SDL",
                              help="model_title, default set to model_title")
    train_parser.add_argument("--seed", type=int, default=1, help="start seed for model")
    train_parser.add_argument("--learning_rate", type=float, default=0.0001,  # p 0.0003/0.5, c 0.0003/0.5, 0.0005/0.5
                              help="learning rate, default set to 1e-4")
    train_parser.add_argument("--weight_decay", type=float, default=0, help="weight decay, default set to 0")
    train_parser.add_argument("--save_dir", type=str, default="./trained_model/",
                              help="directory for saving trained models, default is trained_model folder")
    train_parser.add_argument("--gpus", type=str, default="2", help="gpu ids (default: 7)")
    train_parser.add_argument("--data_root", dest="root_path", type=str,
                              default=str(DEFAULT_DATA_ROOT),
                              help="dataset root, default: <repo>/data/single_modal")
    train_parser.add_argument("--checkpoint", type=str, default=None,
                              help="checkpoint path used for testing")
    train_parser.add_argument("--scheduler", type=str, default='MultiStepLR', help='MultiStepLR or CosineAnnealingLR')

    train_parser.add_argument("--milestones", type=int,
                              default=[30,60,90],  #5, 15, 30, 45, 75, chikusei:100,cave,2:4,0.0001:30,60,90,120, cave,4:10,0.0001,
                              help='milestones for MultiStepLR')
    train_parser.add_argument("--gamma", type=float, default=0.5,
                              help='learning rate decay for MultiStepLR')  # 0.6/0.0007

    # test_parser = subparsers.add_parser("test", help="parser for testing arguments")
    # test_parser.add_argument("--cuda", type=int, required=False, default=1,
    #                          help="set it to 1 for running on GPU, 0 for CPU")
    # test_parser.add_argument("--gpus", type=str, default="0", help="gpu ids (default: 7)")

    # test_parser.add_argument("--test_dir", type=str, required=True, help="directory of testset")
    # test_parser.add_argument("--model_dir", type=str, required=True, help="directory of trained model")

    args = train_parser.parse_args()
    # print(args.gpus)
     # ... existing code ...
    logging.info("Starting the main function.")  # 记录主函数开始
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    if args.subcommand is None:
        print("ERROR: specify either train or test")
        sys.exit(1)
    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)
    if args.subcommand == "train":
        logging.info("Starting training process.")  # 记录训练开始
        train(args)
        logging.info("Training process completed.")  # 记录训练完成
        # train(args)
    else:
        logging.info("Starting testing process.")  # 记录测试开始
        test(args)
        logging.info("Testing process completed.")  # 记录测试完成
    pass


def train(args, img='c'):
    from tensorboardX import SummaryWriter
    from torchnet import meter
    from thop import profile

    device = torch.device("cuda" if args.cuda else "cpu")
    # args.seed = random.randint(1, 10000)
    print("Start seed: ", args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True
    # ... existing code ...
    logging.info("Start seed: %d", args.seed)  # 记录随机种子
    print('===> Loading datasets')
    data_root = Path(args.root_path).expanduser()
    if not data_root.is_absolute():
        data_root = REPO_ROOT / data_root

    dataset_dir = (
        data_root
        / f"{args.dataset_name.lower()}_2"
        / f"{args.dataset_name}_x{args.n_scale}"
    )

    train_path = str(dataset_dir / "trains")
    eval_path = str(dataset_dir / "evals")
    test_data_dir = str(dataset_dir / f"{args.dataset_name}_test.mat")

    train_set = HSTrainingData(image_dir=train_path, augment=True) #cave:false
    eval_set = HSTrainingData(image_dir=eval_path, augment=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=8, shuffle=True)
    eval_loader = DataLoader(eval_set, batch_size=args.batch_size, num_workers=4, shuffle=False)

    if args.dataset_name == 'cave':
        colors = 28
        input_tensor = torch.randn(1, 28, 64, 64).cuda()  # batch_size=1，输入尺寸224×224×3
        lms = torch.randn(1, 28, 128, 128).cuda()  # batch_size=1，输入尺寸224×224×3
    elif args.dataset_name == 'pavia':
        colors = 102
        input_tensor = torch.randn(1, 102, 56, 56).cuda()  # batch_size=1，输入尺寸224×224×3
        lms = torch.randn(1, 102, 112, 112).cuda()  # batch_size=1，输入尺寸224×224×3
    else:
        colors = 128
        input_tensor = torch.randn(1, 128, 64, 64).cuda()  # batch_size=1，输入尺寸224×224×3
        lms = torch.randn(1, 128, 128, 128).cuda()  # batch_size=1，输入尺寸224×224×3

    print('===> Building model')
    net = SDL(dim=args.n_feats, band=colors, scale=args.n_scale, num_blocks=[1, 1, 1]).cuda()  #
    model_structure(net)
    flops, params = profile(net, inputs=(input_tensor,))
    print(f"FLOPs: {flops/1e9} GFLOPs")  # 转换为GFLOPs（1e9次）
    args.model_title = "SDL"
    model_title = args.dataset_name + "_" + args.model_title + '_Blocks=' + str(args.n_blocks) + '_Subs' + str(
        args.n_subs) + '_Ovls' + str(args.n_ovls) + '_Feats=' + str(args.n_feats) + '_scale=' + str(args.n_scale)
    model_name = './checkpoints/' + model_title + "_ckpt_epoch_" + str(40) + ".pth"
    args.model_title = model_title

    if torch.cuda.device_count() > 1:
        print("===> Let's use", torch.cuda.device_count(), "GPUs.")
        net = torch.nn.DataParallel(net)
    start_epoch = 0
    if resume:
        if os.path.isfile(model_name):
            print("=> loading checkpoint '{}'".format(model_name))
            checkpoint = torch.load(model_name)
            start_epoch = checkpoint["epoch"]
            net.load_state_dict(checkpoint["model"].state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(model_name))
    net.cuda().train()

    # h_loss = nn.L1Loss(reduction="mean").cuda()
    # h_loss = torch.nn.L1Loss()
    h_loss = HLoss(0.5, 0.1)
    L1_loss = torch.nn.L1Loss()

    print("===> Setting optimizer and logger")
    # add L2 regularization
    optimizer = Adam(net.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
    # optimizer = torch.optim.SGD(net.parameters(), lr=args.learning_rate)
    epoch_meter = meter.AverageValueMeter()
    writer = SummaryWriter('runs/' + model_title + '_' + str(time.time()))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    print('===> Start training')
    for e in range(start_epoch, args.epochs):
        # ... existing code ...
        logging.info("Start epoch %d, learning rate = %f", e + 1, optimizer.param_groups[0]["lr"])  # 记录每个epoch的开始
        
        epoch_meter.reset()
        print("Start epoch {}, learning rate = {}".format(e + 1, optimizer.param_groups[0]["lr"]))
        for iteration, (x, lms, gt) in enumerate(train_loader):
            x, lms, gt = x.cuda(), lms.cuda(), gt.cuda()
            optimizer.zero_grad()
            y = net(x)
            loss = h_loss(y, gt)
            epoch_meter.add(loss.item())
            loss.backward()
            # torch.nn.utils.clip_grad_norm(net.parameters(), clip_para)
            optimizer.step()
            # tensorboard visualization
            if (iteration + log_interval) % log_interval == 0:
                print(
                    "===> {} B{} Sub{} Fea{} GPU{}\tEpoch[{}]({}/{}): Loss: {:.6f}".format(time.ctime(), args.n_blocks,
                                                                                           args.n_subs, args.n_feats,
                                                                                           args.gpus, e + 1,
                                                                                           iteration + 1,
                                                                                           len(train_loader),
                                                                                           loss.item()))
                n_iter = e * len(train_loader) + iteration + 1
                writer.add_scalar('scalar/train_loss', loss, n_iter)
        scheduler.step()
        print("===> {}\tEpoch {} Training Complete: Avg. Loss: {:.6f}".format(time.ctime(), e + 1,
                                                                              epoch_meter.value()[0]))
        # run validation set every epoch
        eval_loss = validate(args, eval_loader, net, L1_loss)
        # tensorboard visualization
        writer.add_scalar('scalar/avg_epoch_loss', epoch_meter.value()[0], e + 1)
        writer.add_scalar('scalar/avg_validation_loss', eval_loss, e + 1)
        # save model weights at checkpoints every 10 epochs
        if (e + 1) % 1 == 0:
            save_checkpoint(args, net, e + 1)
            print('===>testset' + str(e + 1) + "=============================")
            ttest(args, net, test_data_dir, args.n_scale)
        # ... existing code ...
        logging.info("Epoch %d Training Complete: Avg. Loss: %.6f", e + 1, epoch_meter.value()[0])  # 记录每个epoch的平均损失

    # save model after training
    net.eval().cpu()
    save_model_filename = model_title + "_epoch_all" + str(args.epochs) + "_" + \
                          str(time.time()).replace(' ', '_') + ".pth"
    save_model_path = os.path.join(args.save_dir, save_model_filename)
    if torch.cuda.device_count() > 1:
        torch.save(net.module.state_dict(), save_model_path)
    else:
        torch.save(net.state_dict(), save_model_path)
    print("\nDone, trained model saved at", save_model_path)

    ## Save the testing results
    print("Running testset")
    print('===> Loading testset')
    test_set = HSTestData(test_data_dir)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    print('===> Start testing')
    net.eval().cuda()
    scale = args.n_scale
    with torch.no_grad():
        output = []
        test_number = 0
        for k, (ms, lms, gt) in enumerate(test_loader):
            ms, lms, gt = ms.cuda(), lms.cuda(), gt.cuda()
            # y = net(ms)
            # 定义裁剪和拼接的参数
            crop_size = gt.shape[-1]
            num_channels = gt.shape[1]

            # 存储处理后的子张量
            y = torch.zeros_like(gt)
            # 裁剪和处理
            for i in range(0, gt.shape[2], crop_size):
                for j in range(0, gt.shape[3], crop_size):
                    # 裁剪出 [64, 64, 8] 的子张量
                    img_lr = ms[:, :, i // scale:i // scale + crop_size // scale,
                             j // scale:j // scale + crop_size // scale]
                    # 处理子张量（这里以简单的平方操作为例）
                    output1 = net(img_lr)

                    # 将处理后的子张量添加到列表中
                    y[:, :, i:i + crop_size, j:j + crop_size] = output1
            y, gt = y.squeeze().cpu().numpy().transpose(1, 2, 0), gt.squeeze().cpu().numpy().transpose(1, 2, 0)
            y = y[:gt.shape[0], :gt.shape[1], :]
            if k == 0:
                indices = quality_assessment(gt, y, data_range=1., ratio=4)
            else:
                indices = sum_dict(indices, quality_assessment(gt, y, data_range=1., ratio=4))
            output.append(y)
            test_number += 1
        for index in indices:
            indices[index] = indices[index] / test_number

    # save_dir = "/data/test.npy"
    save_dir = model_title + '.npy'
    np.save(save_dir, output)
    print("Test finished, test results saved to .npy file at ", save_dir)
    print(indices)

    name = args.dataset_name + '_x_all_' + str(args.n_scale) + '.mat'
    out = np.array(output)
    scio.savemat(name, {'pred': out})

    QIstr = model_title + '_all' + str(time.time()) + ".txt"
    json.dump(indices, open(QIstr, 'w'))


def sum_dict(a, b):
    temp = dict()
    for key in a.keys() | b.keys():
        temp[key] = sum([d.get(key, 0) for d in (a, b)])
    return temp


def adjust_learning_rate(start_lr, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 50 epochs"""
    lr = start_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def ttest(args, net, ttest_data_dir, scale):
    ## Save the testing results
    print("Running testset")
    print('===> Loading testset')
    test_set = HSTestData(ttest_data_dir)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    print('===> Start testing')
    net.eval()
    with torch.no_grad():
        output = []
        test_number = 0
        for k, (ms, lms, gt) in enumerate(test_loader):
            ms, lms, gt = ms.cuda(), lms.cuda(), gt.cuda()
            # y = net(ms)
            # 定义裁剪和拼接的参数
            crop_size = gt.shape[-1]
            num_channels = gt.shape[1]

            # 存储处理后的子张量
            y = torch.zeros_like(gt)
            # 裁剪和处理
            for i in range(0, gt.shape[2], crop_size):
                for j in range(0, gt.shape[3], crop_size):
                    # 裁剪出 [64, 64, 8] 的子张量
                    img_lr = ms[:, :, i // scale:i // scale + crop_size // scale,
                             j // scale:j // scale + crop_size // scale]
                    # 处理子张量（这里以简单的平方操作为例）
                    output1 = net(img_lr)

                    # 将处理后的子张量添加到列表中
                    y[:, :, i:i + crop_size, j:j + crop_size] = output1
            # y = chop_forward(ms, net, args.n_scale)
            y, gt = y.squeeze().cpu().numpy().transpose(1, 2, 0), gt.squeeze().cpu().numpy().transpose(1, 2, 0)
            y = y[:gt.shape[0], :gt.shape[1], :]
            if k == 0:
                indices = quality_assessment(gt, y, data_range=1., ratio=4)
            else:
                indices = sum_dict(indices, quality_assessment(gt, y, data_range=1., ratio=4))
            output.append(y)
            test_number += 1
        for index in indices:
            indices[index] = indices[index] / test_number

    name = args.dataset_name + '_x' + str(args.n_scale) + str(1111) + '.mat'
    out = np.array(output)
    scio.savemat(name, {'pred': out})
    print("Test finished")
    print(indices)
    print("=====================================")
    # ... existing code ...
    logging.info(indices)  # 记录每个epoch的平均损失
    net.train()


def validate(args, loader, model, criterion):
    from torchnet import meter

    device = torch.device("cuda" if args.cuda else "cpu")
    # switch to evaluate mode
    model.eval()
    epoch_meter = meter.AverageValueMeter()
    epoch_meter.reset()
    with torch.no_grad():
        for i, (ms, lms, gt) in enumerate(loader):
            ms, lms, gt = ms.cuda(), lms.cuda(), gt.cuda()
            # y = model(ms)            
            y = model(ms)
            loss = criterion(y, gt)
            epoch_meter.add(loss.item())
        mesg = "===> {}\tEpoch evaluation Complete: Avg. Loss: {:.6f}".format(time.ctime(), epoch_meter.value()[0])
        print(mesg)
    # back to training mode
    model.train()
    return epoch_meter.value()[0]


def test(args):
    device = torch.device("cuda" if args.cuda else "cpu")
    print('===> Loading testset')
    data_root = Path(args.root_path).expanduser()
    if not data_root.is_absolute():
        data_root = REPO_ROOT / data_root

    dataset_dir = (
        data_root
        / f"{args.dataset_name.lower()}_2"
        / f"{args.dataset_name}_x{args.n_scale}"
    )
    test_data_dir = str(dataset_dir / f"{args.dataset_name}_test.mat")
    test_set = HSTestData(test_data_dir)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    model_title = args.dataset_name + "_" + args.dataset_name + "_" + args.model_title + '_Blocks=' + str(
        args.n_blocks) + '_Subs' + str(
        args.n_subs) + '_Ovls' + str(args.n_ovls) + '_Feats=' + str(args.n_feats) + '_scale=' + str(args.n_scale)
    if not args.checkpoint:
        raise ValueError("Testing requires --checkpoint <path-to-model.pth>")

    checkpoint_path = Path(args.checkpoint).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = REPO_ROOT / checkpoint_path

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print('===> Start testing')
    model = torch.load(str(checkpoint_path), map_location=device)
    # model = SDL(dim=args.n_feats, band=128, scale=args.n_scale, num_blocks=[1, 1, 1]).eval()  # .cuda()  #
    # ckpt = torch.load(model_name, map_location='cpu')
    # model_dict = model.state_dict()
    # for k, v in ckpt['model'].named_parameters():  # :ckpt['model'].items()
    #     kr = k.replace("module.", "")  # f" "  # f" "
    #     if kr in model_dict.keys():
    #         assert v.shape == model_dict[kr].shape, f"Shape mismatch: {v.shape} vs {model_dict[kr].shape}"
    #         model_dict[kr] = v
    #     else:
    #         print(f"Passing weights: {k}")
    #
    # model.load_state_dict(model_dict)
    # # model = torch.load(model_name)['model']
    model.eval().to(device)
    scale = args.n_scale
    with torch.no_grad():
        output = []
        test_number = 0
        for k, (ms, lms, gt) in enumerate(test_loader):
            ms, lms, gt = ms.cuda(), lms.cuda(), gt.cuda()
            # y = net(ms)
            # 定义裁剪和拼接的参数
            crop_size = gt.shape[-1]
            num_channels = gt.shape[1]
            # 存储处理后的子张量
            y = torch.zeros_like(gt)
            # 裁剪和处理
            for i in range(0, gt.shape[2], crop_size):
                for j in range(0, gt.shape[3], crop_size):
                    # 裁剪出 [64, 64, 8] 的子张量
                    img_lr = ms[:, :, i // scale:i // scale + crop_size // scale,
                             j // scale:j // scale + crop_size // scale]
                    # 处理子张量（这里以简单的平方操作为例）
                    output1 = model(img_lr)

                    # 将处理后的子张量添加到列表中
                    y[:, :, i:i + crop_size, j:j + crop_size] = output1
            # y = chop_forward(ms, net, args.n_scale)
            y, gt = y.squeeze().cpu().numpy().transpose(1, 2, 0), gt.squeeze().cpu().numpy().transpose(1, 2, 0)
            y = y[:gt.shape[0], :gt.shape[1], :]
            if k == 0:
                indices = quality_assessment(gt, y, data_range=1., ratio=4)
            else:
                indices = sum_dict(indices, quality_assessment(gt, y, data_range=1., ratio=4))
            output.append(y)
            test_number += 1
        for index in indices:
            indices[index] = indices[index] / test_number

        # save_dir = "/data/test.npy"
    save_dir = model_title + '.npy'
    np.save(save_dir, output)
    print("Test finished, test results saved to .npy file at ", save_dir)
    print(indices)

    name = args.dataset_name + '_x' + str(args.n_scale) + '.mat'
    out = np.array(output)
    scio.savemat(name, {'pred': out})

    QIstr = model_title + '_' + str(time.time()) + ".txt"
    json.dump(indices, open(QIstr, 'w'))


def chop_forward(x, model, scale, shave=16):
    b, c, h, w = x.size()
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half, w_half  # + shave + shave
    inputlist = [
        x[:, :, 0:h_size, 0:w_size],
        x[:, :, 0:h_size, (w - w_size):w],
        x[:, :, (h - h_size):h, 0:w_size],
        x[:, :, (h - h_size):h, (w - w_size):w]]
    outputlist = []
    for i in range(4):
        input_batch = inputlist[i]
        output_batch = model(input_batch)
        outputlist.append(output_batch)

    output = Variable(x.data.new(b, c, h * scale, w * scale))
    print(output.shape)
    output[:, :, 0:h_half * scale, 0:w_half * scale] = outputlist[0][:, :, 0:h_half * scale, 0:w_half * scale]
    output[:, :, 0:h_half * scale, w_half * scale:w * scale] = outputlist[1][:, :, 0:h_half * scale,
                                                               (w_size - w + w_half) * scale:w_size * scale]
    output[:, :, h_half * scale:h * scale, 0:w_half * scale] = outputlist[2][:, :,
                                                               (h_size - h + h_half) * scale:h_size * scale,
                                                               0:w_half * scale]
    output[:, :, h_half * scale:h * scale, w_half * scale:w * scale] = outputlist[3][:, :,
                                                                       (h_size - h + h_half) * scale:h_size * scale,
                                                                       (w_size - w + w_half) * scale:w_size * scale]

    return output


if __name__ == "__main__":
    main()
