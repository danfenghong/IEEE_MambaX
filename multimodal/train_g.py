from model_two import *
from utilities.utils import *
import torch.utils.data as tud
import datetime
from option_traing import opt
import sys
import torch.nn as nn
from utilities.visualizer import *
from utilities.load_train_data import Dataset_Pro1
import errno
from thop import profile
from metrics import ref_evaluate, no_ref_evaluate
from utilities.load_test_data import load_h5py1
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


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
if not torch.cuda.is_available():
    raise Exception('NO GPU!')

# saving path
date_time = str(datetime.datetime.now())
date_time = time2file_name(date_time)
result_path = os.path.join(opt.outf, date_time, "result")
txt_path = os.path.join(result_path, "output.txt")
model_path = os.path.join(opt.outf, date_time, "model")
if not os.path.exists(result_path):
    os.makedirs(result_path)
if not os.path.exists(model_path):
    os.makedirs(model_path)


# log_path = opt.outf + date_time + '/log.txt'
# model
def calculate_averages(lst, interval):
    averges = []
    for i in range(0, len(lst), interval):
        sublist = lst[i:i + interval]
        avg = sum(sublist) / len(sublist)
        averges.append(avg)
    return averges


model = model_generator(opt.method, opt).cuda()
# model = torch.nn.DataParallel(aaa)
model_structure(model)
input1 = torch.randn(1, 4, 16, 16).cuda()
input3 = torch.randn(1, 1, 64, 64).cuda()
flops, params = profile(model, inputs=(input1,input3))
print(f"FLOPs: {flops/1e9} GFLOPs")  # 转换为GFLOPs（1e9次）
# optimizing
optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, betas=(0.9, 0.999))  # Adam
# optimizer = torch.optim.SGD(model.parameters(), lr=opt.learning_rate)
if opt.scheduler == 'MultiStepLR':
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestones, gamma=opt.gamma)
elif opt.scheduler == 'CosineAnnealingLR':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.max_epoch, eta_min=1e-6)

# criterion = nn.MSELoss().cuda()
# criterion_mean = nn.L1Loss(reduction="mean").cuda()
criterion_mean = HLoss(0.5, 0.1).cuda()

def eval_metrics1(epoch, loader, mode):
    # input:epoch test_loader
    # output:None
    psnr_mean_list = []
    ssim_mean_list = []
    mse_mean_list = []
    sam_mean_list = []

    begin = time.time()

    duration = len(loader.dataset)

    for iteration, batch in enumerate(loader, 1):
        # print(f"begin {i}th iteration\n")
        psnr_list, ssim_list, mse_list, sam_list = [], [], [], []
        gt_test, label_rgb_test, input_meas = batch[0].cuda(), batch[3].cuda(), batch[4].cuda()
        model.eval()
        with torch.no_grad():
            pred = model(input_meas, label_rgb_test)

        """
                Quality Metrics Setting
                                            """
        for k in range(gt_test.shape[0]):
            # --------------- torch_psnr ---------------#
            psnr_val = torch_psnr(pred[k, :, :, :], gt_test[k, :, :, :])
            psnr_mean_list.append(psnr_val.detach().cpu().numpy())
            # --------------- torch_ssim ---------------#
            ssim_val = torch_ssim(pred[k, :, :, :], gt_test[k, :, :, :])
            ssim_mean_list.append(ssim_val.detach().cpu().numpy())
            # --------------- compare_mse ---------------#
            mse_val = compare_mse(pred[k, :, :, :], gt_test[k, :, :, :])
            mse_mean_list.append(mse_val.detach().cpu().numpy())
            # --------------- SAM_GPU ---------------#
            sam_val = SAM_GPU(pred[k, :, :, :], gt_test[k, :, :, :])
            sam_mean_list.append(sam_val.detach().cpu().numpy())
    # ------------------------- Mean_list -------------------------#
    psnr_mean = np.mean(np.asarray(psnr_mean_list))
    ssim_mean = np.mean(np.asarray(ssim_mean_list))
    mse_mean = np.mean(np.asarray(mse_mean_list))
    sam_mean = np.mean(np.asarray(sam_mean_list))

    end = time.time()
    ###### =========================================================================================================
    ### ================================================================================================================
    # ======================================================================================================================

    if mode == "test":
        psnr_list = calculate_averages(psnr_mean_list, duration)
        ssim_list = calculate_averages(ssim_mean_list, duration)
        mse_list = calculate_averages(mse_mean_list, duration)
        sam_list = calculate_averages(sam_mean_list, duration)
        print(f"===> Epoch {epoch + 1}: psnr list: {psnr_list}")
        print(f"===> Epoch {epoch + 1}: ssim list: {ssim_list}")
        print(f"===> Epoch {epoch + 1}: mse list: {mse_list}")
        print(f"===> Epoch {epoch + 1}: sam list: {sam_list}")
    print(
        '===> Epoch {}: {} mode  psnr = {:.2f}, ssim = {:.3f}, mse = {:.10f},sam = {:.3f}, time: {:.2f}'
        .format(epoch + 1, mode, psnr_mean, ssim_mean, mse_mean, sam_mean,
                (end - begin)))
    model.train()
    return psnr_mean, ssim_mean, mse_mean, sam_mean
def eval_metrics(epoch, loader, mode):
    file_path = opt.reduced_test_path
    img_lr, img_lr_u, img_pan, gt = load_h5py1(file_path)

    # get size
    image_num, C, h, w = img_lr.shape
    _, _, H, W = img_pan.shape

    # # reduce
    psnr_all = np.zeros(image_num)
    Q_all = np.zeros(image_num)
    sam_all = np.zeros(image_num)
    ergas_all = np.zeros(image_num)

    for k in range(image_num):
        print('Processing the {}th sample...'.format(k + 1))
        model.eval()
        with torch.no_grad():
            # output = model(img_lr[k:k + 1, :, :, :], img_pan[k:k + 1, :, :, :])
            img_lr_u1 = img_lr_u[k:k + 1, :, :, :].to('cuda')
            img_lr1 = img_lr[k:k + 1, :, :, :].to('cuda')
            img_pan1 = img_pan[k:k + 1, :, :, :].to('cuda')
            # output = model(img_lr1, img_lr_u1, img_pan1)  # 网络输出

            # 定义裁剪和拼接的参数
            crop_size = img_lr_u1.shape[-1]
            num_channels = img_lr_u1.shape[1]

            # 存储处理后的子张量
            output = torch.zeros_like(img_lr_u1)
            # 裁剪和处理
            for i in range(0, img_lr_u.shape[2], crop_size):
                for j in range(0, img_lr_u.shape[3], crop_size):
                    # 裁剪出 [64, 64, 8] 的子张量
                    img_pan11 = img_pan1[:, :, i:i + crop_size, j:j + crop_size]
                    img_lr11 = img_lr1[:, :, i // 4:i // 4 + crop_size // 4, j // 4:j // 4 + crop_size // 4]
                    # 处理子张量（这里以简单的平方操作为例）
                    output1 = model(img_lr11, img_pan11)

                    # 将处理后的子张量添加到列表中
                    output[:, :, i:i + crop_size, j:j + crop_size] = output1

            target = gt[k:k + 1, :, :, :].to('cuda')
            output = torch.squeeze(output).permute(1, 2, 0).cpu().detach().numpy()  # HxWxC
            target = torch.squeeze(target).permute(1, 2, 0).cpu().detach().numpy()  # HxWxC
            psnr, ssim, sam, ergas, scc, q = ref_evaluate(output, target)

            print(f"psnr:{psnr},ssim:{ssim},sam:{sam},ergas:{ergas}")
            psnr_all[k] = psnr
            Q_all[k] = ssim
            sam_all[k] = sam
            ergas_all[k] = ergas
            if (k + 1) % image_num == 0:
                print(f'++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
                print(f"psnr:{psnr_all.mean()},ssim:{Q_all.mean()},sam:{sam_all.mean()},ergas:{ergas_all.mean()}")
                print(f'++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    model.train()


# iteration {1: 3d}, '
# 'PSNR {2:2.2f} dB.'.format(args.denoiser.upper(), k + 1, psnr_all),
#  'SSIM:{:2.3f}.'.format(ssim_all))

class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


if __name__ == "__main__":
    # sys.stdout = Logger(log_path)
    sys.stdout = Logger(fpath=txt_path)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    print(opt)
    data_init_begin = time.time()

    # ======================================================================================================================
    """
            Dataset Load Setting
                                        """

    train_set = Dataset_Pro1(opt.data_path_train) #gf
    validate_set = Dataset_Pro1(opt.data_path_test)
    loader_train = tud.DataLoader(dataset=train_set, num_workers=0, batch_size=opt.batch_size,
                                  shuffle=True, pin_memory=True, drop_last=True)
    loader_test = tud.DataLoader(dataset=validate_set, num_workers=0, batch_size=opt.batch_size,
                                 shuffle=True, pin_memory=True, drop_last=True)

    # ======================================================================================================================
    # ======================================================================================================================
    data_init_end = time.time()
    print(f"dataset loading costs {data_init_end - data_init_begin} s\n")

    # pipline of training
    for epoch in range(0, opt.max_epoch):
        print(f"begin the {epoch + 1}th epoch train:")
        model.train()
        epoch_loss = 0
        # epoch_loss_gt = 0
        # epoch_loss_kl = 0
        psnr_mean_list = []
        ssim_mean_list = []
        mse_mean_list = []
        sam_mean_list = []

        start_time = time.time()

        # ---------------- Iteration ----------------#
        for iteration, batch in enumerate(loader_train, 1):
            # print(f"begin {i}th iteration\n")
            iteration_time_begin = time.time()
            gt, label_rgb, input = batch[0].cuda(), batch[3].cuda(), batch[4].cuda()

            psnr_list, ssim_list, mse_list, sam_list = [], [], [], []
            model_out = model(input, label_rgb)

            # ---------------- Loss Setting ----------------#
            loss = criterion_mean(model_out, gt)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iteration_time_end = time.time()
        elapsed_time = time.time() - start_time

        # ======================================================================================================================
        print(
            'epcoh = %4d , loss = %.10f, time = %4.2f s' % (
                epoch + 1, epoch_loss / len(train_set), elapsed_time))
        if epoch in list(range(0, opt.max_epoch, 10)):
            print(f"begin to calculate the {epoch + 1}th epoch train metrics ")
            psnr_train, ssim_train, mse_train, sam_train = eval_metrics1(epoch=epoch, loader=loader_train, mode="train")
        print(f"begin to calculate the {epoch + 1}th epoch test metrics ")
        eval_metrics(epoch=epoch, loader=loader_test, mode="test")

        scheduler.step()

        torch.save(model, os.path.join(model_path, 'model_%03d.pth' % (epoch + 1)))
