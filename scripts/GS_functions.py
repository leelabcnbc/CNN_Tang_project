import os
import sys
import scipy.io as io
import numpy as np
import matplotlib.pyplot as plt
import PIL
import torch
import shutil
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts

from audtorch.metrics.functional import pearsonr

import imshowtools 
# sys.path.append(where you put this file),from GS_functions import GF
# sys.path.append('/user_data/shanggao/tang/'),from GS_functions import GF
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn



class GF:
    def img2matrix(imgmainpath,suffix='.png'):
        '''
        make sure all imgs are the same size
        '''
        pathlist=GF.filelist_suffix(imgmainpath,suffix)
        im_shape=np.array(PIL.Image.open(f"{imgmainpath}/{pathlist[0]}").convert('L')).shape[0]
        im_matrix=np.zeros((len(pathlist),im_shape,im_shape))
        for i in range(len(pathlist)):
            # print('Change img:',i+1)
            im_matrix[i,:,:]= np.array(PIL.Image.open(f"{imgmainpath}/{pathlist[i]}").convert('L'))
        
        return im_matrix
            
    def show_imgs_in1Page(img_matrix,cmap='gray',showsize=(10,10),columns=None,rows=None,padding=False,title=None):
        '''
        shape: (numbers,H,W)
        '''

        assert len(img_matrix.shape)==3
        assert isinstance(showsize,tuple)
        imshowtools.imshow(*img_matrix,cmap=cmap,size=showsize,columns=columns,rows=rows,padding=padding,title=title)
    def filelist_suffix(filepath, suffix=None):
        """
        this is to find all the file with certain suffix, and order it.
        REMEMBER: filenames has to be numbers (beside the suffix)
        """
        filelist = os.listdir(filepath)
        assert isinstance(suffix, (str, tuple, type(None)))
        if suffix != None:
            if isinstance(suffix, str):
                filelist = [f for f in filelist if f.endswith((str(suffix)))]
                filelist.sort(key=lambda x: int(x[:-4]))
                print("There are ", len(filelist), " files in this directory")
            if isinstance(suffix, tuple):
                filelist = [f for f in filelist if f.endswith(suffix)]
                filelist.sort(key=lambda x: int(x[:-4]))
                print("There are ", len(filelist), " files in this directory")

        elif suffix == None:
            filelist.sort(key=lambda x: int(x[:-4]))
            print("There are ", len(filelist), " files in this directory")
        filelist_final = np.array(filelist)
        return filelist_final

    def gen_range(start, stop, step,mode='accumulate'):
        """
        Generate list
        mode=accumulate/separate
        """
        mode=mode.lower()
        if mode not in ('accumulate','separate'):
            raise RuntimeError('mode=accumulate/separate')
        if mode=='accumulate':
            current = start
            while current < stop:
                next_current = current + step
                if next_current < stop:
                    yield (int(start), int(next_current))
                else:
                    yield (int(start), int(stop))
                current = next_current
        elif mode=='separate':
            current = start
            while current < stop:
                next_current = current + step
                if next_current < stop:
                    yield (int(current), int(next_current))
                else:
                    yield (int(current), int(stop))
                current = next_current

    def gen_list_tuple(start, stop, step,mode='accumulate'):
        '''
        mode=accumulate/separate; 
        default: accumulate
        '''
        a = []
        for i in GF.gen_range(start, stop, step,mode):
            a.append(i)
        return a

    # def gen_range(start, stop, step):
    #     """Generate list"""
    #     current = start
    #     while current < stop:
    #         next_current = current + step
    #         if next_current < stop:
    #             yield (current, next_current)
    #         else:
    #             yield (current, stop)
    #         current = next_current

    def norm_to_1(imagemat):
        """
        In: Input shape should be 4[BHWC or BCHW] or 3[CHW or HWC] or 2[HW] or 1[vector], tensor or numpy arrary.
        Out: Norm to 1 version , Batch and Channel seperate
        """
        assert (
            len(imagemat.shape) == 2
            or len(imagemat.shape) == 1
            or len(imagemat.shape) == 4,
            len(imagemat.shape) == 3,
        ), "Input shape should be 4[BHWC or BCHW] or 3[CHW or HWC] or 2[HW] or 1[vector]"
        assert isinstance(imagemat, torch.Tensor) or isinstance(
            imagemat, np.ndarray
        ), "input data should be torch tensor or numpy array"
        grad_mode = None
        if isinstance(imagemat, torch.Tensor):
            if imagemat.requires_grad:
                grad_mode = "True"
                GG = imagemat.grad
            else:
                grad_mode = "False"
                GG = None

            imagemat_new = imagemat.detach().clone()  # .detach().clone()
            print("---------------------------")
            imagemat_new = torch.tensor(imagemat_new, dtype=torch.float)  ### new line
        else:
            imagemat_new = imagemat.copy()
            imagemat_new = np.array(imagemat_new, dtype=np.float32)  ### new line
        if len(imagemat_new.shape) == 3:
            C, H, W = imagemat_new.shape
            assert (
                C == 1 or C == 3 or W == 1 or W == 3
            ), "Input should be CHW or HWC, and channel can only be 1 or 3"
            if C == 1 or C == 3:
                new_img = GF.channel_norm1(imagemat_new, mode="CHW")
            elif W == 1 or W == 3:
                new_img = GF.channel_norm1(imagemat_new, mode="HWC")
            else:
                raise RuntimeError("Check input")

        if len(imagemat_new.shape) == 2 or len(imagemat_new.shape) == 1:
            if imagemat_new.max() == imagemat_new.min():
                new_img = imagemat_new
            else:
                new_img = (imagemat_new - imagemat_new.min()) / (
                    imagemat_new.max() - imagemat_new.min()
                )

        if len(imagemat_new.shape) == 4:
            B, H, W, C = imagemat_new.shape
            assert H == 1 or H == 3 or C == 1 or C == 3, "Input should be BHWC or BCHW"
            if C == 1 or C == 3:
                mode = "HWC"
                for i in range(B):
                    imagemat_new[i, :, :, :] = GF.channel_norm1(
                        imagemat_new[i, :, :, :], mode=mode
                    )
                new_img = imagemat_new
            elif H == 1 or H == 3:
                mode = "CHW"
                for i in range(B):

                    imagemat_new[i, :, :, :] = GF.channel_norm1(
                        imagemat_new[i, :, :, :], mode=mode
                    )
                new_img = imagemat_new
            else:
                assert False == True, "Check whether your image channel is 1 or 3"
        if grad_mode == "True":
            new_img.requires_grad = True
            new_img.grad = GG
            # new_img = torch.tensor(new_img, requires_grad=True)
        return new_img

    def channel_norm1(mat, mode="CHW(HWC)"):
        if isinstance(mat, torch.Tensor):
            mat_new = mat.clone()
        else:
            mat_new = mat.copy()
        assert len(mat_new.shape) == 3, "Input shape should be 3D(CHW or HWC)"
        assert isinstance(mat_new, np.ndarray) or isinstance(
            mat_new, torch.Tensor
        ), "input should be numpy or torch tensor"
        if mode == "CHW(HWC)" or mode == "CHW":
            for i in range(mat_new.shape[0]):
                mat_new[i, :, :] = (mat_new[i, :, :] - mat_new[i, :, :].min()) / (
                    mat_new[i, :, :].max() - mat_new[i, :, :].min()
                )
            F_mat = mat_new
        elif mode == "HWC":
            for i in range(mat_new.shape[2]):
                mat_new[:, :, i] = (mat_new[:, :, i] - mat_new[:, :, i].min()) / (
                    mat_new[:, :, i].max() - mat_new[:, :, i].min()
                )
            F_mat = mat_new
        else:
            assert False == True, "Input mode: CHW or HWC"
        return F_mat

    def npy2mat(varname, npyfilepath, matsavepath):
        gg = np.load(npyfilepath)
        io.savemat(matsavepath, {varname: gg})
    def mat2npy(matfilename,varname):
        '''
        this method only works for matlab > v7 file.
        '''
        mat = io.loadmat(matfilename)
        rsp=mat[varname]
        return rsp
    def copy_allfiles(src, dest):
        """
        src:folder path
        dest: folder path
        this will not keep moving the folder to another folder
        this is moving the files in that folder to another folder
        """
        src_files = os.listdir(src)
        for file_name in src_files:
            full_file_name = os.path.join(src, file_name)
            if os.path.isfile(full_file_name):
                shutil.copy(full_file_name, dest)

    def mkdir(mainpath, foldername):
        """
        mainpath: path you want to create folders
        foldername: foldername, str, list or tuple
        Return: the path you generate.
        """
        assert isinstance(foldername, (str, tuple, list))
        if isinstance(foldername, str):
            pathname = GF.mkdir0(mainpath, foldername)
        if isinstance(foldername, (list, tuple)):
            pathname = []
            for i in foldername:
                pathname0 = GF.mkdir0(mainpath, i)
                pathname.append(pathname0)
        return pathname

    def mkdir0(mainpath, foldername):
        if mainpath[-1] == "/" or mainpath[-1] == "\\":
            pathname = mainpath + foldername + "/"
            folder = os.path.exists(mainpath + foldername + "/")
            if not folder:
                os.makedirs(mainpath + foldername)
                print("Create folders ing")
                print("done !")
            else:
                print("folder existed")
        else:
            pathname = mainpath + "/" + foldername + "/"
            folder = os.path.exists(mainpath + "/" + foldername + "/")
            if not folder:
                os.makedirs(mainpath + foldername)
                print("Create folders ing")
                print("done !")
            else:
                print("folder already existed")
        return pathname

    def sortTC(vector, sort_mode="Top_down"):
        """
        sort_mode: Top_down/Bottom_up(default:Top_down)

        """
        if sort_mode not in ("Top_down", "Bottom_up"):
            raise RuntimeError(
                "sort_mode args incorrect:\nPlease input:\n1.Top_down\n2.Bottom_up"
            )

        if sort_mode == "Top_down":
            value = np.sort(vector)[::-1]
            index = np.argsort(vector)[::-1]
        elif sort_mode == "Bottom_up":
            value = np.sort(vector)
            index = np.argsort(vector)
        return value, index

    def save_mat_file(filename, var, varname="data"):
        io.savemat(filename + ".mat", {varname: var})

    def tf_to_torch_shape(tf_img):
        assert len(tf_img.shape) == 4, "Shape should be 4 dimensional"
        B, H, W, C = tf_img.shape
        assert (
            C == 1 or C == 3
        ), "Input image shape should be BHWC, Channel judgement is 1 or 3"
        # newi = torch.from_numpy(tf_img).unsqueeze_(0).view(B, C, H, W)
        newi = torch.from_numpy(tf_img).view(B, C, H, W)
        newi = newi.type(torch.float)
        B1, C1, H1, W1 = newi.shape
        assert C1 == 1 or C1 == 3, "Check whether the input shape is BHWC "
        return newi

    def load_data(train_pic_path, train_rsp_path, val_pic_path, val_rsp_path):
        """
        load data from tf format(BHWC)-> norm picture to 1 -> Out is torch format(BCHW)
        """
        train_rsp = np.load(train_rsp_path)
        val_rsp = np.load(val_rsp_path)
        train_pics = np.load(train_pic_path)
        val_pics = np.load(val_pic_path)

        train_pics = GF.norm_to_1(train_pics)
        val_pics = GF.norm_to_1(val_pics)
        train_pics = GF.tf_to_torch_shape(train_pics)
        val_pics = GF.tf_to_torch_shape(val_pics)  # [B,H,W,C] -> [B,C,H,W]
        train_rsp, val_rsp = torch.tensor(train_rsp, dtype=torch.float), torch.tensor(
            val_rsp, dtype=torch.float
        )

        print("val pic shape:", val_pics.shape, "\n train pics shape", train_pics.shape)
        print("val rsp shape", val_rsp.shape, "\n train rsp shape", train_rsp.shape)

        return train_pics, val_pics, train_rsp, val_rsp  # [B/imgs,C,H,W] or [pics,cell]

    def get_all_metrics(pred, real, num_neurons, img_samples_size):
        # pred/real -> (imgs,cells)/(imgs,epoch)
        # assert pred.shape[0] > 500, "First dimension should be images"
        R_square = GF.get_all_R2(pred, real)
        R, VE = GF.get_corr_VE(pred, real, num_neurons, img_samples_size)
        return R_square, R, VE

    def get_corr_VE(pred, real, num_neurons, img_samples_size):
        # I: pred/real -> (imgs,cells), O: (cells,)
        assert (
            len(pred.shape) == 2 and len(real.shape) == 2
        ), "Input shape: (imgs, cells)"
        assert isinstance(pred, np.ndarray) and isinstance(
            real, np.ndarray
        ), "Data input type should be numpy array"
        R = np.zeros(num_neurons)
        VE = np.zeros(num_neurons)
        for neuron in range(num_neurons):
            val_pred = pred[:, neuron]
            val_real = real[:, neuron]
            u2 = np.zeros((2, img_samples_size))
            u2[0, :] = np.reshape(val_pred, (img_samples_size))
            u2[1, :] = np.reshape(val_real, (img_samples_size))
            c2 = np.corrcoef(u2)
            R[neuron] = c2[0, 1]
            VE0 = 1 - (np.var(val_pred - val_real) / np.var(val_real))
            VE[neuron] = VE0
        return R, VE

    def adjust_R_square(pred, real, sample_size=None, label_size=None):
        if isinstance(pred, np.ndarray) and isinstance(real, np.ndarray):
            # pred,real -> numpy, shape:(xx,)
            RSS = np.sum((real - pred) ** 2)
            TSS = np.sum((real - real.mean()) ** 2)
        if isinstance(pred, torch.Tensor) and isinstance(real, torch.Tensor):
            # pred,real -> numpy, shape:(xx,)
            RSS = torch.sum((real - pred) ** 2)
            TSS = torch.sum((real - real.mean()) ** 2)

        R_square = 1 - RSS / TSS

        if not label_size == None:
            n = sample_size
            p = label_size
            R_square_adjust = 1 - ((1 - R_square) * (n - 1)) / (n - p - 1)
        else:
            R_square_adjust = "None"
        return R_square

    def get_all_R2(pred, real):
        # pred or real, (imgs,cells)
        assert (
            len(pred.shape) == 2 and len(real.shape) == 2
        ), "Input shape: (imgs, cells)"
        assert isinstance(pred, np.ndarray) and isinstance(
            real, np.ndarray
        ), "Data input type should be numpy array"
        R2 = []
        for i in range(pred.shape[1]):
            pred1 = pred[:, i]
            real1 = real[:, i]
            # sample_size=pred1.shape[0]
            R = GF.adjust_R_square(pred1, real1)
            R2.append(R)
        return np.stack(R2)

    def different_loss(lossname, real, predict, val_or_train=None):
        if lossname in ("mix_corr_MAE", "mix_corr_MSE"):
            assert val_or_train != None
        if lossname == "mix_corr_MAE":
            criterion = torch.nn.L1Loss(reduction="none")
            if val_or_train == "train":
                loss = (
                    -pearsonr(predict, real, batch_first=True)
                    + 0.1 * torch.mean(torch.abs(real))
                    + 0.1 * torch.mean(criterion(predict, real))
                )
            if val_or_train == "val":
                loss = (
                    torch.mean(-pearsonr(predict, real, batch_first=False))
                    + 0.1 * torch.mean(torch.abs(real))
                    + 0.1 * torch.mean(criterion(predict, real))
                )
        elif lossname == "mix_corr_MSE":
            criterion = torch.nn.MSELoss(reduction="mean")
            if val_or_train == "train":
                loss = (
                    -pearsonr(predict, real, batch_first=True)
                    + 0.1 * torch.mean(torch.abs(real))
                    + 0.1 * torch.mean(criterion(predict, real))
                )
            if val_or_train == "val":
                loss = (
                    torch.mean(-pearsonr(predict, real, batch_first=False))
                    + 0.1 * torch.mean(torch.abs(real))
                    + 0.1 * torch.mean(criterion(predict, real))
                )
        elif lossname == "MAE":
            criterion = torch.nn.L1Loss(reduction="mean")
            loss = criterion(predict, real)
        elif lossname == "MSE":
            # criterion = torch.nn.MSELoss(reduction="sum")  # O have reduction "mean"
            criterion = torch.nn.MSELoss(reduction="mean")
            loss = criterion(predict, real)

        elif lossname == "RMSLE":
            criterion = RMSLELoss()
            loss = criterion(predict, real)
        elif lossname == "MSEN":
            print("_---------", real.shape)
            loss = torch.mean(
                torch.square(torch.relu(torch.abs(predict - real) - 0.1)), axis=-1
            )
            # 保证输入real和predict是（1，xxxx）
        return loss




class ImageDataset(Dataset):
    def __init__(self, data, labels, cell_start=None, cell_end=None, num_neurons=None):
        """
        cell_start: start from 1
        mode=num_neurons/startend
        """
        self.data = data
        self.labels = labels
        self.cell_start = cell_start
        self.cell_end = cell_end
        self.num_neurons = num_neurons

    def __len__(self):
        return self.data.shape[0]  # number of images

    def __getitem__(self, index):
        cell_start = self.cell_start
        cell_end = self.cell_end
        num_neurons = self.num_neurons
        assert ((cell_start == None and cell_end == None) and num_neurons != None) or (
            (cell_start != None and cell_end != None) and num_neurons == None
        )

        img = self.data[index]

        if num_neurons != None:
            label = self.labels[index, 0 : self.num_neurons]
        elif cell_start != None:
            label = self.labels[index, self.cell_start - 1 : self.cell_end]
        # print("img shape", img.shape, "rsp shape", label.shape)
        return img, label


class RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))
