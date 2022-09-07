import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader


def get_label_pattern_stimuli (data, PS_labels, batch_size, device, net) :
    dataset = TensorDataset(torch.Tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size)

    neuron_all_rsp = torch.empty(0).to(device)
    with torch.no_grad():
        for ps_img in tqdm(loader):
            x = ps_img[0].to(device)
            rsp = net(x)
            neuron_all_rsp = torch.cat((neuron_all_rsp, rsp))
    neuron_all_rsp = neuron_all_rsp.detach().cpu().numpy()
    top_indexes = np.argsort(neuron_all_rsp, axis=0)[::-1]
    top_10 = top_indexes[:10, :]
    top_10_rsp = np.sort(neuron_all_rsp, axis=0)[::-1][:10, :]

    top_10 = np.transpose(top_10)
    top_10_rsp = np.transpose(top_10_rsp)

    #get all neurons, use only the top as judgement
    top_required = 1
    max_ratio = 1

    def check_max_valid(neuron, max_rsp):
        for index in top_indexes[:, neuron]:
            if PS_labels[index] < 2:
                if max_ratio * neuron_all_rsp[index, neuron] > max_rsp:
                    return False
                else:
                    return True
        return True

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

    neuron_labels = np.full(top_10.shape[0], -1)
    for neuron, (neuron_top, neuron_top_rsp) in enumerate(zip(top_10, top_10_rsp)):
        all_labels = []
        for img_idx in neuron_top:
            all_labels.append(PS_labels[img_idx])
        majority = majority_very_strict(neuron, all_labels, neuron_top_rsp[0])
        neuron_labels[neuron] = majority
    label_names = ['SS', 'EB', 'CN', 'CV', 'CRS', 'Other']
    valid_neurons = []
    valid_neurons_labels = []
    for i, label in enumerate(neuron_labels):
        if label > -1:
            valid_neurons.append(i)
            valid_neurons_labels.append(label_names[label])
    return valid_neurons, valid_neurons_labels, top_10, top_10_rsp