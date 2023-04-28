import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader

top_required = 8
max_ratio = 1.5


def check_max_valid(neuron, max_rsp, top_indexes, PS_labels_all, neuron_all_rsp):
    for index in top_indexes[neuron]:
        if PS_labels_all[index] < 2:
            if max_ratio * neuron_all_rsp[index, neuron] > max_rsp:
                return False
            else:
                return True
    return True


def majority_norm(num_list):
    count_list = np.zeros(6)
    for item in num_list:
        count_list[item] += 1
    max_idx = np.argmax(count_list)
    return max_idx


def majority_strict(num_list):
    count_list = np.zeros(6)
    for item in num_list:
        count_list[item] += 1
    max_idx = np.argmax(count_list)
    if count_list[max_idx] > len(num_list) * 0.8:
        return max_idx
    else:
        return -1


def majority_very_strict(neuron, num_list, max_rsp):
    for i in range(1, top_required):
        if not num_list[i] == num_list[i - 1]:
            return -1
        if num_list[0] > 1 and not check_max_valid(neuron, max_rsp):
            return -1
    else:
        return num_list[0]


def majority_percent(neuron, top_idxs, top_percent, neuron_all_rsp, PS_labels_all):
    max_idx = top_idxs[0]
    max_rsp = neuron_all_rsp[max_idx, neuron]
    mixed_flag = False
    for idx in top_idxs:
        if neuron_all_rsp[idx, neuron] > top_percent * max_rsp:
            if not PS_labels_all[idx] == PS_labels_all[max_idx]:
                if PS_labels_all[max_idx] < 2:
                    return PS_labels_all[max_idx]
                elif PS_labels_all[idx] == 0:
                    return 0
                elif PS_labels_all[idx] == 1:
                    continue
                else:
                    mixed_flag = True
        else:
            break
    if mixed_flag == True:
        return 6
    return PS_labels_all[max_idx]


def majority_percent_vote(neuron, top_idxs, top_percent, neuron_all_rsp, PS_labels_all):
    max_idx = top_idxs[0]
    max_rsp = neuron_all_rsp[max_idx, neuron]
    all_voters = []
    for idx in top_idxs:
        if neuron_all_rsp[idx, neuron] > top_percent * max_rsp:
            all_voters.append(PS_labels_all[idx])
        else:
            break
    return majority_norm(all_voters)

def get_label_pattern_stimuli (data, PS_labels, batch_size, device, net, stacked_data = False) :
    dataset = TensorDataset(torch.Tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size)

    neuron_all_rsp = torch.empty(0).to(device)
    with torch.no_grad():
        for ps_img in tqdm(loader):
            x = ps_img[0].to(device)
            rsp = net(x)
            neuron_all_rsp = torch.cat((neuron_all_rsp, rsp))
    neuron_all_rsp = neuron_all_rsp.detach().cpu().numpy()

    if stacked_data:
        stacked_data = np.stack(np.split(neuron_all_rsp,4))
        neuron_all_rsp = np.max(stacked_data, axis=0)

    top_indexes = np.argsort(neuron_all_rsp, axis=0)[::-1]
    top_rsp = np.sort(neuron_all_rsp, axis=0)[::-1]

    top_indexes = np.transpose(top_indexes)
    top_rsp = np.transpose(top_rsp)

    num_iters = 10
    all_neuron_labels = np.full((num_iters, top_indexes.shape[0]), -1)
    label_names = ['SS', 'EB', 'CN', 'CV', 'CRS', 'Other', 'MC']

    selected_idx = range(neuron_all_rsp.shape[1])

    for iter in tqdm(range(num_iters)):
        top_percent = 1 - (0.5 / num_iters) * (iter + 1)
        neuron_labels = np.full(top_indexes.shape[0], -1)
        for neuron in selected_idx:
            neuron_top = top_indexes[neuron]
            all_labels = []
            for img_idx in neuron_top:
                label = PS_labels[img_idx]
                all_labels.append(label)
            majority = majority_percent(neuron, neuron_top, top_percent, neuron_all_rsp, PS_labels)
            neuron_labels[neuron] = majority
        all_neuron_labels[iter] = neuron_labels
    return all_neuron_labels, top_indexes, top_rsp


def majority_num(neuron, top_idxs, top_num, neuron_all_rsp, PS_labels_all):
    max_idx = top_idxs[0]
    max_rsp = neuron_all_rsp[max_idx, neuron]
    if PS_labels_all[max_idx] < 2:
        return PS_labels_all[max_idx]
    for idx in top_idxs[:top_num]:
        if not PS_labels_all[idx] == PS_labels_all[max_idx]:
            if PS_labels_all[idx] < 2 and  neuron_all_rsp[idx, neuron] >= 0.8*max_rsp :
                return PS_labels_all[idx]
            else:
                return 6
    return PS_labels_all[max_idx]
def get_label_pattern_stimuli_Gale (neuron_all_rsp, PS_labels) :


    top_indexes = np.argsort(neuron_all_rsp, axis=0)[::-1]
    top_rsp = np.sort(neuron_all_rsp, axis=0)[::-1]

    top_indexes = np.transpose(top_indexes)
    top_rsp = np.transpose(top_rsp)

    num_iters = 6
    all_neuron_labels = np.full((num_iters, top_indexes.shape[0]), -1)
    label_names = ['SS', 'EB', 'CN', 'CV', 'CRS', 'Other', 'MC']

    selected_idx = range(neuron_all_rsp.shape[1])

    for iter in tqdm(range(num_iters)):
        top_num = 10-iter
        neuron_labels = np.full(top_indexes.shape[0], -1)
        for neuron in selected_idx:
            neuron_top = top_indexes[neuron]
            all_labels = []
            for img_idx in neuron_top:
                label = PS_labels[img_idx]
                all_labels.append(label)
            majority = majority_num(neuron, neuron_top, top_num, neuron_all_rsp, PS_labels)
            neuron_labels[neuron] = majority
        all_neuron_labels[iter] = neuron_labels
    return all_neuron_labels, top_indexes, top_rsp


def majority_num_NO_EB(neuron, top_idxs, top_num, neuron_all_rsp, PS_labels_all):
    max_idx = top_idxs[0]
    max_rsp = neuron_all_rsp[max_idx, neuron]
    if PS_labels_all[max_idx] < 2:
        return PS_labels_all[max_idx]
    for idx in top_idxs[:top_num]:
        if not PS_labels_all[idx] == PS_labels_all[max_idx]:
            if PS_labels_all[idx] == 0 and  neuron_all_rsp[idx, neuron] >= 0.8*max_rsp :
                return PS_labels_all[idx]
            else:
                return 6
    return PS_labels_all[max_idx]
def get_label_pattern_stimuli_Gale_NO_EB (neuron_all_rsp, PS_labels) :
    top_indexes = np.argsort(neuron_all_rsp, axis=0)[::-1]
    top_rsp = np.sort(neuron_all_rsp, axis=0)[::-1]

    top_indexes = np.transpose(top_indexes)
    top_rsp = np.transpose(top_rsp)

    num_iters = 6
    all_neuron_labels = np.full((num_iters, top_indexes.shape[0]), -1)
    label_names = ['SS', 'EB', 'CN', 'CV', 'CRS', 'Other', 'MC']

    selected_idx = range(neuron_all_rsp.shape[1])

    for iter in tqdm(range(num_iters)):
        top_num = 10 - iter
        neuron_labels = np.full(top_indexes.shape[0], -1)
        for neuron in selected_idx:
            neuron_top = top_indexes[neuron]
            all_labels = []
            for img_idx in neuron_top:
                label = PS_labels[img_idx]
                all_labels.append(label)
            majority = majority_num_NO_EB(neuron, neuron_top, top_num, neuron_all_rsp, PS_labels)
            neuron_labels[neuron] = majority
        all_neuron_labels[iter] = neuron_labels
    return all_neuron_labels, top_indexes, top_rsp

