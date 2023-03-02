import torchvision
import torchvision.transforms as transforms
import torch
from modeling.models.bethge import BethgeModel
from tqdm import tqdm
import numpy as np
from analysis.Stat_utils import get_site_corr
from modeling.train_utils import array_to_dataloader
from torch.utils.data import DataLoader

if __name__ == '__main__':
    device = 'cuda'

    batch_size = 10

    all_rsp = []
    all_val_rsp = []
    sites = ['m1s1', 'm1s2', 'm1s3', 'm2s1', 'm2s2', 'm3s1']
    all_kept_idx = []
    all_exclude_idx = np.load("D:/school/research/CNN_Tang_project/data/exclude_rc_large_idxs_70percent.npy",
                              allow_pickle=True)
    for site_idx, site in enumerate(sites):
        train_x = np.load('../data/Processed_Tang_data/all_sites_data_prepared/pics_data/train_img_' + site + '.npy')
        val_x = np.load('../data/Processed_Tang_data/all_sites_data_prepared/pics_data/val_img_' + site + '.npy')
        train_y = np.load(
            '../data/Processed_Tang_data/all_sites_data_prepared/New_response_data/trainRsp_' + site + '.npy')
        val_y = np.load('../data/Processed_Tang_data/all_sites_data_prepared/New_response_data/valRsp_' + site + '.npy')
        train_x = np.transpose(train_x, (0, 3, 1, 2))
        val_x = np.transpose(val_x, (0, 3, 1, 2))
        train_loader = array_to_dataloader(train_x, train_y, batch_size=50, shuffle=True)
        val_loader = array_to_dataloader(val_x, val_y, batch_size=50)

        channels = 256
        num_layers = 9
        input_size = 50
        output_size = val_y.shape[1]
        first_k = 9
        later_k = 3
        pool_size = 2
        factorized = True

        num_maps = 1

        net = BethgeModel(channels=channels, num_layers=num_layers, input_size=input_size,
                          output_size=output_size, first_k=first_k, later_k=later_k,
                          input_channels=1, pool_size=pool_size, factorized=True,
                          num_maps=num_maps).cuda()

        net.load_state_dict(torch.load('../saved_models/new_learned_models/' + site + '_9_model_version_0'))
        net.eval()
        net.to(device)
        corr, _ = get_site_corr(net, device, site)
        kept_idx = []
        exclude_idx = all_exclude_idx[site_idx]
        for i, c in enumerate(corr):
            if c > 0.7 and (not i in exclude_idx):
                kept_idx.append(i)
        kept_idx = np.stack(kept_idx)
        all_kept_idx.append(kept_idx)
        new_rsp = []
        for i, (x,_) in enumerate(tqdm(train_loader)):
            x = x.float().to(device)
            rsp = net(x)
            rsp_kept = rsp[:, kept_idx].detach().cpu().numpy()
            new_rsp.append(rsp_kept)

        new_rsp = np.concatenate(new_rsp)
        all_rsp.append(new_rsp)

        new_rsp = []
        for i, (x,_) in enumerate(tqdm(val_loader)):
            x = x.float().to(device)
            rsp = net(x)
            rsp_kept = rsp[:, kept_idx].detach().cpu().numpy()
            new_rsp.append(rsp_kept)

        new_rsp = np.concatenate(new_rsp)
        all_val_rsp.append(new_rsp)

    all_rsp = np.concatenate(all_rsp, axis=1)
    all_val_rsp = np.concatenate(all_val_rsp, axis=1)
    np.save("279_selected_rsp_tang.npy", all_rsp)
    np.save("279_selected_rsp_tang_test.npy", all_val_rsp)
