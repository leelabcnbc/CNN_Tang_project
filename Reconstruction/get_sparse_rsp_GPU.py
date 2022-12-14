import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import PIL
import copy
import sys
import os
import gc

sys.path.append("/user_data/shanggao/tang/")
from GS_functions import GF
import scipy
from scipy.ndimage import rotate

sys.path.append("../")
from funs_sparse import SCR
from pytictoc import TicToc

t = TicToc()
import S_functions as funs

name = os.path.basename(os.getcwd())
sitename = f"m{name[1]}s{name[3]}"
datapath = "/user_data/shanggao/tang/V1_Data/all_sites_data_prepared/"  # todonote


def A_main(
    Final_dictionary_savename,
    degrees_num,
    sparse_basis_all,
    mainsavepath,
    device,
    namename,
    stride=1,
    center_shift_num=1,
    dtype=torch.float64,
    criteria_name="corr",
    activations_list=["relu", "square", "cube", "same", "relusquare"],
):
    ########### define path ##############

    final_npyfiles_sp = GF.mkdir(mainsavepath, "npyfiles")
    Final_dictionary_savepath=f"{final_npyfiles_sp}/{Final_dictionary_savename}"#/all_cell_dict_{namename}.npy"

    # ####
    assert len(sparse_basis_all.shape) == 2
    val_rsp = np.load(f"{datapath}New_response_data/valRsp_{sitename}.npy")
    # (1000,num of neurons)
    val_pics = np.load(f"{datapath}pics_data/val_img_{sitename}.npy")
    # (1000, 50, 50, 1)
    currectDir = "/user_data/shanggao/tang/new_experiments/A_sparsecode_rsp/"
    npyfiles_savepath = GF.mkdir(currectDir, "A_npyfiles")
    rotated_img_path = GF.mkdir(npyfiles_savepath, "Rotated_datasets")

    rotated_imgs_savepath = funs.create_rotated_images(
        pics=val_pics,
        degrees_num=degrees_num,
        rotated_img_path=rotated_img_path,
        sitename=sitename,
    )

    funs.A_get_all_results_all_neurons(
        rotated_imgs_savepath=rotated_imgs_savepath,
        degrees_num=degrees_num,
        basis=sparse_basis_all,
        real_rsp_all=val_rsp,
        stride=stride,
        center_shift_num=center_shift_num,
        dtype=dtype,
        device=device,
        criteria_name=criteria_name,
        activations_list=activations_list,
        Final_dictionary_savepath=Final_dictionary_savepath
    )
    
   # now change save process 

data_read_dict = {
    # SIZE 9
    "Bruno_BASIS1_NUM_64_size9": "../data/Bruno_BASIS1_NUM_64_size9.npy",
    "Bruno_BASIS1_NUM_128_size9": "../data/Bruno_BASIS1_NUM_128_size9.npy",
    "Bruno_BASIS1_NUM_256_size9": "../data/Bruno_BASIS1_NUM_256_size9.npy",
    "Bruno_BASIS1_NUM_512_size9": "../data/Bruno_BASIS1_NUM_512_size9.npy",
    "Bruno_BASIS1_NUM_1024_size9": "../data/Bruno_BASIS1_NUM_1024_size9.npy",
    "Bruno_BASIS1_NUM_2048_size9": "../data/Bruno_BASIS1_NUM_2048_size9.npy",
    "Bruno_BASIS1_NUM_4096_size9":"../data/Bruno_BASIS1_NUM_4096_size9.npy",
    # SIZE 16, start 7 (from 0)
    "Bruno_BASIS1_NUM_64_size16": "../data/Bruno_BASIS1_NUM_64_size16.npy",
    "Bruno_BASIS1_NUM_128_size16": "../data/Bruno_BASIS1_NUM_128_size16.npy",
    "Bruno_BASIS1_NUM_256_size16": "../data/Bruno_BASIS1_NUM_256_size16.npy",
    "Bruno_BASIS1_NUM_512_size16": "../data/Bruno_BASIS1_NUM_512_size16.npy",
    "Bruno_BASIS1_NUM_1024_size16": "../data/Bruno_BASIS1_NUM_1024_size16.npy",
    "Bruno_BASIS1_NUM_2048_size16": "../data/Bruno_BASIS1_NUM_2048_size16.npy",
    "Bruno_BASIS1_NUM_4096_size16":"../data/Bruno_BASIS1_NUM_4096_size16.npy"
}

name_list = list(data_read_dict.keys())
# data_name = name_list[6]
for data_name in name_list[:6]:
    sparse_basis_all = np.load(data_read_dict[data_name])
    degrees_num = 18 #TODO run 9, 1 
    all_result_p = GF.mkdir("./", "Results")
    center_shift_num = 5 # run 1,
    mainsavepath00 = GF.mkdir(all_result_p, data_name)  # TODO bruno_10overcomplete
    mainsavepath = GF.mkdir(
        mainsavepath00, f"center_shift_{center_shift_num}_degrees_num{degrees_num}"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device='cpu'
    namename = ""
    Final_dictionary_savename=f"all_cell_dict_{namename}.npy"

    print(mainsavepath)

    t.tic()
    A_main(
        Final_dictionary_savename=Final_dictionary_savename,
        degrees_num=degrees_num,
        sparse_basis_all=sparse_basis_all,
        mainsavepath=mainsavepath,
        device=device,
        namename=namename,
        center_shift_num=center_shift_num,
    )
    t.toc()
