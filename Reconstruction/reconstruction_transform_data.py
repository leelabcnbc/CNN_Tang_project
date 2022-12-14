import torchvision
import torchvision.transforms as transforms
import torch
from modeling.models.bethge import BethgeModel
from tqdm import tqdm
import numpy as np
from analysis.Stat_utils import get_site_corr
from torch.utils.data import DataLoader

if __name__ == '__main__':
    device = 'cuda'

    size = 16
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize(50),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
         transforms.Grayscale()]
    )

    transform_crop = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize(50),
         transforms.CenterCrop(size),
         transforms.Pad((50 - size) // 2, 0.5),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
         transforms.Grayscale()]
    )

    batch_size = 10

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_crop)
    trainloader = DataLoader(trainset, batch_size=batch_size,
                             shuffle=False, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_crop)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)
    all_rsp = []
    all_val_rsp = []
    sites = ['m1s1', 'm1s2', 'm1s3', 'm2s1', 'm2s2', 'm3s1']
    all_kept_idx = []
    all_exclude_idx = np.load("D:/school/research/CNN_Tang_project/data/exclude_rc_large_idxs_70percent.npy",
                              allow_pickle=True)
    for site_idx, site in enumerate(sites):
        val_y = np.load('../data/Processed_Tang_data/all_sites_data_prepared/New_response_data/valRsp_' + site + '.npy')
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
        for i, (x, _) in enumerate(tqdm(trainloader)):
            x = x.float().to(device)
            rsp = net(x)
            rsp_kept = rsp[:, kept_idx].detach().cpu().numpy()
            new_rsp.append(rsp_kept)

        new_rsp = np.concatenate(new_rsp)
        all_rsp.append(new_rsp)

        new_rsp = []
        for i, (x, _) in enumerate(tqdm(testloader)):
            x = x.float().to(device)
            rsp = net(x)
            rsp_kept = rsp[:, kept_idx].detach().cpu().numpy()
            new_rsp.append(rsp_kept)

        new_rsp = np.concatenate(new_rsp)
        all_val_rsp.append(new_rsp)

    all_rsp = np.concatenate(all_rsp, axis=1)
    all_val_rsp = np.concatenate(all_val_rsp, axis=1)
    np.save("279_selected_rsp_cifar10_16.npy", all_rsp)
    np.save("279_selected_rsp_cifar10_test_16.npy", all_val_rsp)
