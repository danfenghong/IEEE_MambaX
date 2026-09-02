import scipy.io as sio
import os
import glob
import re
import random
import logging
from .ssim_torch import *


# ======================================================================================================================
# Usage Metri
# ======================================================================================================================
def _as_floats(im1, im2):
    im1 = im1.cpu().detach()
    im1 = im1.numpy()
    im2 = im2.cpu().detach()
    im2 = im2.numpy()
    float_type = np.result_type(im1.dtype, im2.dtype, np.float64)  # trans to float64/32
    im1 = np.asarray(im1, dtype=float_type)
    im2 = np.asarray(im2, dtype=float_type)
    return im1, im2


# def torch_psnr(img, ref):  # input [28,256,256]
#     # img = img.numpy().astype(np.float64)
#     # ref = ref.numpy().astype(np.float64)
#     img, ref = _as_floats(img, ref)
#
#     nC = img.shape[0]
#     psnr = 0
#     for i in range(nC):
#         # mse = torch.mean((img[i, :, :] - ref[i, :, :]) ** 2)
#         # mse = torch.tensor(np.mean(np.square(img[i, :, :] - ref[i, :, :]), dtype=np.float64))
#         psnr += 10 * torch.log10((255 * 255) / mse)
#     return psnr / nC

def torch_psnr(img, ref):  # input [28,256,256]
    img = (img * 256).round()
    ref = (ref * 256).round()
    nC = img.shape[0]
    psnr = 0
    for i in range(nC):
        mse = torch.mean((img[i, :, :] - ref[i, :, :]) ** 2)
        psnr += 10 * torch.log10((255 * 255) / mse)
    return psnr / nC


def compute_psnr(img2, img1):
    assert img1.ndim == 3 and img2.ndim == 3
    img_c, img_w, img_h = img1.shape
    ref = img1.reshape(img_c, -1)
    tar = img2.reshape(img_c, -1)
    msr = np.mean((ref - tar) ** 2, 1)
    max1 = np.max(ref, 1)
    psnrall = 10 * np.log10(max1 ** 2 / msr)
    out_mean = np.mean(psnrall)
    return out_mean  # , max1


def torch_ssim(img, ref):  # input [28,256,256]
    return ssim(torch.unsqueeze(img, 0), torch.unsqueeze(ref, 0))


def compare_mse(im1, im2):
    im1, im2 = _as_floats(im1, im2)
    return torch.tensor((np.mean(np.square(im1 - im2), dtype=np.float64)))


def SAM_GPU(img, ref):
    C = img.size()[0]
    H = img.size()[1]
    W = img.size()[2]
    esp = 1e-12
    Itrue = img.clone()  # .resize_(C, H*W)
    Ifake = ref.clone()  # .resize_(C, H*W)
    nom = torch.mul(Itrue, Ifake).sum(dim=0)  # .resize_(H*W)
    denominator = Itrue.norm(p=2, dim=0, keepdim=True).clamp(min=esp) * \
                  Ifake.norm(p=2, dim=0, keepdim=True).clamp(min=esp)
    denominator = denominator.squeeze()
    sam = torch.div(nom, denominator).acos()
    sam[sam != sam] = 0
    sam_sum = torch.sum(sam) / (H * W) / np.pi * 180
    return sam_sum


# ======================================================================================================================
# ======================================================================================================================

def normalize(data):
    h, w, c = data.shape
    data = data.reshape((h * w, c))
    data -= np.min(data, axis=0)
    data /= np.max(data, axis=0)
    data = data.reshape((h, w, c))
    return data


def generate_masks(mask_path, batch_size):
    mask = sio.loadmat(mask_path + '/mask.mat')
    mask = mask['mask']
    mask3d = np.tile(mask[:, :, np.newaxis], (1, 1, 28))
    mask3d = np.transpose(mask3d, [2, 0, 1])
    mask3d = torch.from_numpy(mask3d)
    [nC, H, W] = mask3d.shape
    mask3d_batch = mask3d.expand([batch_size, nC, H, W]).cuda().float()
    return mask3d_batch


def generate_shift_masks(mask_path, batch_size):
    mask = sio.loadmat(mask_path + '/mask_3d_shift.mat')
    mask_3d_shift = mask['mask_3d_shift']
    mask_3d_shift = np.transpose(mask_3d_shift, [2, 0, 1])
    mask_3d_shift = torch.from_numpy(mask_3d_shift)
    [nC, H, W] = mask_3d_shift.shape
    Phi_batch = mask_3d_shift.expand([batch_size, nC, H, W]).cuda().float()
    Phi_s_batch = torch.sum(Phi_batch ** 2, 1)
    Phi_s_batch[Phi_s_batch == 0] = 1
    print(Phi_batch.shape, Phi_s_batch.shape)  # 256,310,28
    return Phi_batch, Phi_s_batch


def shift_mask(mask_3d, batch_size):
    # mask = sio.loadmat(mask_path + '/mask_3d_shift.mat')
    # mask_3d_shift = mask['mask_3d_shift']
    mask_3d_shift = np.transpose(mask_3d, [2, 0, 1])
    mask_3d_shift = torch.from_numpy(mask_3d_shift)
    [nC, H, W] = mask_3d_shift.shape
    Phi_batch = mask_3d_shift.expand([batch_size, nC, H, W]).cuda().float()
    Phi_s_batch = torch.sum(Phi_batch ** 2, 1)
    Phi_s_batch[Phi_s_batch == 0] = 1
    # print(Phi_batch.shape, Phi_s_batch.shape)  256,310,28
    return Phi_s_batch  # Phi_batch


def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, 'model_*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch


def loadpath(pathlistfile):
    fp = open(pathlistfile)
    pathlist = fp.read().splitlines()
    fp.close()
    random.shuffle(pathlist)
    return pathlist


def time2file_name(time):
    year = time[0:4]
    month = time[5:7]
    day = time[8:10]
    hour = time[11:13]
    minute = time[14:16]
    second = time[17:19]
    time_filename = year + '_' + month + '_' + day + '_' + hour + '_' + minute + '_' + second
    return time_filename


def shuffle_crop_all(train_hsi, train_rgb, batch_size, crop_size=256, argument=True):
    if argument:
        gt_batch = []
        rgb_batch = []
        index_hsi = np.random.choice(range(len(train_hsi)), batch_size)
        processed_data1 = np.zeros((batch_size, crop_size, crop_size, 28), dtype=np.float32)

        index_rgb = np.random.choice(range(len(train_rgb)), batch_size)
        processed_data2 = np.zeros((batch_size, crop_size, crop_size, 3), dtype=np.float32)

        for i in range(batch_size):
            img_hsi = train_hsi[index_hsi[i]]
            h, w, _ = img_hsi.shape
            x_index = np.random.randint(0, h - crop_size)
            y_index = np.random.randint(0, w - crop_size)
            processed_data1[i, :, :, :] = img_hsi[x_index:x_index + crop_size, y_index:y_index + crop_size, :].cpu()
            img_rgb = train_rgb[index_rgb[i]]
            processed_data2[i, :, :, :] = img_rgb[x_index:x_index + crop_size, y_index:y_index + crop_size, :]

        gt_batch = torch.from_numpy(np.transpose(processed_data1, (0, 3, 1, 2))).cuda().float()
        rgb_batch = torch.from_numpy(np.transpose(processed_data2, (0, 3, 1, 2))).cuda().float()

        return gt_batch, rgb_batch
    else:
        index_rgb = np.random.choice(range(len(train_rgb)), batch_size)
        processed_data2 = np.zeros((batch_size, crop_size, crop_size, 3), dtype=np.float32)

        index_hsi = np.random.choice(range(len(train_hsi)), batch_size)
        processed_data1 = np.zeros((batch_size, crop_size, crop_size, 28), dtype=np.float32)
        for i in range(batch_size):
            h, w, _ = train_rgb[index_rgb[i]].shape
            x_index = np.random.randint(0, h - crop_size)
            y_index = np.random.randint(0, w - crop_size)
            img_rgb = train_rgb[index_rgb[i]]
            processed_data2[i, :, :, :] = img_rgb[x_index:x_index + crop_size, y_index:y_index + crop_size,
                                          :].cpu()
            img = train_hsi[index_hsi[i]]
            processed_data1[i, :, :, :] = img[x_index:x_index + crop_size, y_index:y_index + crop_size, :]
        gt_batch = torch.from_numpy(np.transpose(processed_data1, (0, 3, 1, 2)))
        rgb_batch = torch.from_numpy(np.transpose(processed_data2, (0, 3, 1, 2)))

        return gt_batch, rgb_batch


def gen_log(model_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")

    log_file = model_path + '/log.txt'
    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    #
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    #
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def gen_meas_torch(data_batch, mask3d_batch, Y2H=True, mul_mask=False):
    nC = data_batch.shape[1]
    temp = shift(mask3d_batch * data_batch, 2)
    meas = torch.sum(temp, 1)
    if Y2H:
        meas = meas / nC * 2
        H = shift_back(meas)
        if mul_mask:
            HM = torch.mul(H, mask3d_batch)
            return HM
        return H
    return meas


def init_mask(mask, Phi, Phi_s, mask_type):
    if mask_type == 'Phi':
        input_mask = Phi
    elif mask_type == 'Phi_PhiPhiT':
        input_mask = (Phi, Phi_s)
    elif mask_type == 'Mask':
        input_mask = mask
    elif mask_type == None:
        input_mask = None
    return input_mask


def shift(inputs, step=2):
    [bs, nC, row, col] = inputs.shape
    output = torch.zeros(bs, nC, row, col + (nC - 1) * step).cuda().float()
    for i in range(nC):
        output[:, i, :, step * i:step * i + col] = inputs[:, i, :, :]
    return output


def shift_back(inputs, step=2):  # input [bs,256,310]  output [bs, 28, 256, 256]
    [bs, row, col] = inputs.shape
    nC = 28
    output = torch.zeros(bs, nC, row, col - (nC - 1) * step).cuda().float()
    for i in range(nC):
        output[:, i, :, :] = inputs[:, :, step * i:step * i + col - (nC - 1) * step]
    return output


def init_meas(gt, mask, input_setting):
    if input_setting == 'H':
        input_meas = gen_meas_torch(gt, mask, Y2H=True, mul_mask=False)
    elif input_setting == 'HM':
        input_meas = gen_meas_torch(gt, mask, Y2H=True, mul_mask=True)
    elif input_setting == 'Y':
        input_meas = gen_meas_torch(gt, mask, Y2H=False, mul_mask=True)
    return input_meas


def checkpoint(model, epoch, model_path, logger):
    model_out_path = model_path + "/model_epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
    logger.info("Checkpoint saved to {}".format(model_out_path))


#######################################################################################################
def init_mea(gt, mask, input_setting):
    if input_setting == 'H':
        input_mea = gen_mea(gt, mask, Y2H=True)
    return input_mea


def gen_mea(data_batch, mask3d_batch, Y2H=True):
    nC = data_batch.shape[0]
    temp = sift(mask3d_batch * data_batch, 2)
    mea = torch.sum(temp, 1)
    if Y2H:
        mea = mea / nC * 2
        H = sift_data(mea)
        return H
    return mea


def sift(input, step=2):
    [nC, row, col] = input.shape
    output = torch.zeros(nC, row, col + (nC - 1) * step)
    for i in range(nC):
        output[i, :, step * i:step * i + col] = input[i, :, :]  #
    return output


def sift_data(input, nC, step=2):  # input [256,310]  output [256, 256, 28]
    [row, col] = input.shape
    # nC = 31
    output = torch.zeros(nC, row, col - (nC - 1) * step)
    for i in range(nC):
        output[i, :, :] = input[:, step * i:step * i + col - (nC - 1) * step]
    return output


#######################################################################################################
def LoadTest(path_test, data_type):
    scene_list = os.listdir(path_test)
    scene_list.sort()
    if data_type == "kaist":
        test_data = np.zeros((len(scene_list), 512, 512, 28)).astype(np.float32)
        test_rgb = np.zeros((len(scene_list), 512, 512, 3)).astype(np.float32)
    elif data_type == "cave":
        test_data = np.zeros((len(scene_list), 256, 256, 28)).astype(np.float32)  # 512
        test_rgb = np.zeros((len(scene_list), 256, 256, 3)).astype(np.float32)
    elif data_type == "icvl":
        test_data = np.zeros((len(scene_list), 1300, 1300, 31)).astype(np.float32)
        test_rgb = np.zeros((len(scene_list), 1300, 1300, 3)).astype(np.float32)
    if data_type == "cave":
        for i in range(len(scene_list)):
            scene_path = path_test + scene_list[i]
            data = sio.loadmat(scene_path)
            test_data[i, :, :, :] = data['mat_list'][i, :, :, 0:28]
            test_rgb[i, :, :, :] = data['mat_list'][i, :, :, 28:31]
    else:
        for i in range(len(scene_list)):
            scene_path = path_test + scene_list[i]
            data = sio.loadmat(scene_path)
            if data_type == "kaist":
                img = data['kaist_data'] / 65536.
                rgb = data['kaist_rgb'] / 65536.
            if data_type == "cave":
                img = data['cave_data']
                rgb = data['cave_rgb']
            if data_type == "icvl":
                img = data['data']
                rgb = data['rgb']
            test_data[i, :, :, :] = img
            test_rgb[i, :, :, :] = rgb
    test_data = torch.from_numpy(np.transpose(test_data, (0, 3, 1, 2)))
    test_rgb = torch.from_numpy(np.transpose(test_rgb, (0, 3, 1, 2)))
    return test_data, test_rgb


#########################
# mask_3d to mask_3d shift

def sift_mask(inputs, step=2):
    [nC, row, col] = inputs.shape
    output = torch.zeros(nC, row, col + (nC - 1) * step)
    for i in range(nC):
        output[i, :, step * i:step * i + col] = inputs[i, :, :]  # by channel one by one move, so have some zero
    return output
#######################################################################################################

# def prepare_data_cave(path, file_num):
#     HR_HSI = np.zeros((((512,512,28,file_num))))
#     file_list = os.listdir(path)
#     # for idx in range(1):
#     for idx in range(file_num):
#         print(f'loading CAVE {idx}')
#         ####  read HrHSI
#         HR_code = file_list[idx]
#         path1 = os.path.join(path) + HR_code
#         data = sio.loadmat(path1)
#         HR_HSI[:,:,:,idx] = data['data_slice'] / 65535.0
#         HR_HSI[HR_HSI < 0] = 0
#         HR_HSI[HR_HSI > 1] = 1
#     return HR_HSI
#
# def prepare_data_KAIST(path, file_num):
#     HR_HSI = np.zeros((((2704,3376,28,file_num))))
#     file_list = os.listdir(path)
#     # for idx in range(1):
#     for idx in range(file_num):
#         print(f'loading KAIST {idx}')
#         ####  read HrHSI
#         HR_code = file_list[idx]
#         path1 = os.path.join(path) + HR_code
#         data = sio.loadmat(path1)
#         HR_HSI[:,:,:,idx] = data['HSI']
#         HR_HSI[HR_HSI < 0] = 0
#         HR_HSI[HR_HSI > 1] = 1
#     return HR_HSI


def cal_gradient_c(x):
    c_x = x.size(1)
    g = x[:, 1:, 1:, 1:] - x[:, :c_x - 1, 1:, 1:]
    return g


def cal_gradient_x(x):
    c_x = x.size(2)
    g = x[:, 1:, 1:, 1:] - x[:, 1:, :c_x - 1, 1:]
    return g


def cal_gradient_y(x):
    c_x = x.size(3)
    g = x[:, 1:, 1:, 1:] - x[:, 1:, 1:, :c_x - 1]
    return g


def cal_gradient(inp):
    x = cal_gradient_x(inp)
    y = cal_gradient_y(inp)
    c = cal_gradient_c(inp)
    g = torch.sqrt(torch.pow(x, 2) + torch.pow(y, 2) + torch.pow(c, 2) + 1e-6)
    return g


def cal_sam(Itrue, Ifake):
    esp = 1e-6
    InnerPro = torch.sum(Itrue * Ifake, 1, keepdim=True)
    len1 = torch.norm(Itrue, p=2, dim=1, keepdim=True)
    len2 = torch.norm(Ifake, p=2, dim=1, keepdim=True)
    divisor = len1 * len2
    mask = torch.eq(divisor, 0)
    divisor = divisor + (mask.float()) * esp
    cosA = torch.sum(InnerPro / divisor, 1).clamp(-1 + esp, 1 - esp)
    sam = torch.acos(cosA)
    sam = torch.mean(sam) / np.pi
    return sam


class HLoss(torch.nn.Module):
    def __init__(self, la1, la2, sam=True, gra=True):
        super(HLoss, self).__init__()
        self.lamd1 = la1
        self.lamd2 = la2
        self.sam = sam
        self.gra = gra

        self.fidelity = torch.nn.L1Loss()
        self.gra = torch.nn.L1Loss()

    def forward(self, y, gt):
        loss1 = self.fidelity(y, gt)
        loss2 = self.lamd1 * cal_sam(y, gt)
        loss3 = self.lamd2 * self.gra(cal_gradient(y), cal_gradient(gt))
        loss = loss1 + loss2 + loss3
        return loss