import torch
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from modeling.models.bethge import BethgeModel
from modeling.train_utils import array_to_dataloader

def make_dir_try(dir):
    try:
        os.makedirs(dir, exist_ok=True)
        print("Directory '%s' created successfully" % dir)
    except OSError as error:
        print("Directory '%s' can not be created" % dir)

def adjust_R_square(pred, real, sample_size=None, label_size=None):
    RSS = np.sum((real - pred) ** 2)
    TSS = np.sum((real - real.mean()) ** 2)
    R_square = 1 - RSS / TSS

    if not label_size == None:
        n = sample_size
        p = label_size
        R_square_adjust = 1 - ((1 - R_square) * (n - 1)) / (n - p - 1)
    else:
        R_square_adjust = "None"
    return R_square, R_square_adjust

def get_stat_data(selected_idx, prediction, actual):
    R = []
    VE = []
    R_square = []
    for neuron in selected_idx:
        pred1 = prediction[:, neuron]
        val_y = actual[:, neuron]

        u2 = np.zeros((2, 1000))
        u2[0, :] = np.reshape(pred1, 1000)
        u2[1, :] = np.reshape(val_y, 1000)

        c2 = np.corrcoef(u2)
        R.append(c2[0, 1])

        VE.append(1 - np.var(pred1 - val_y) / np.var(val_y))

        rsquare, _ = adjust_R_square(pred1, val_y)
        R_square.append(rsquare)
    return R, VE, R_square


channels = 256
num_layers = 9
input_size = 50
first_k = 9
later_k = 3
pool_size = 2
factorized = True
num_maps = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sites = ['m1s1', 'm1s2', 'm1s3', 'm2s1', 'm2s2', 'm3s1']

for site in sites:
    directory_base = 'A:/school/College_Junior/research/CNN_Tang_project/analysis/Graphs/Sample_size_curves_all_sites/'+site+'/'
    make_dir_try(directory_base)
    val_x = np.load('../data/Processed_Tang_data/all_sites_data_prepared/pics_data/val_img_' + site + '.npy')
    val_y = np.load('../data/Processed_Tang_data/all_sites_data_prepared/New_response_data/valRsp_' + site + '.npy')
    val_x = np.transpose(val_x, (0, 3, 1, 2))
    num_neurons = val_y.shape[1]
    val_loader = array_to_dataloader(val_x, val_y, batch_size=200)
    selected_idx = np.load('../data/cell_id_new_dataIndex/'+site+'/cell_id.npy')
    selected_idx = [i-1 for i in selected_idx]
    num_laps = 10
    labels = []
    all_R = []
    all_VE = []
    all_R_square = []
    avg_R = []
    avg_VE = []
    avg_R_square = []
    avg_R_all_neuron = []
    avg_VE_all_neuron = []
    avg_R_square_all_neuron = []
    for i in range(num_laps):
        with torch.no_grad():
            net = BethgeModel(channels=channels, num_layers=num_layers, input_size=input_size,
                              output_size=num_neurons, first_k=first_k, later_k=later_k,
                              input_channels=1, pool_size=pool_size, factorized=True,
                              num_maps=num_maps).cuda()
            net.to(device)
            net.load_state_dict(torch.load('../saved_models/Sample_size_test_models/'+str(i)+'_10_'+site+'_shared_core_256_9'))
            net.eval()
            prediction = []
            actual = []
            for batch_num, (x, y) in enumerate(tqdm(val_loader)):
                x, y = x.to(device), y.to(device)
                outputs = net(x).cpu().numpy()
                prediction.extend(outputs)
                actual.extend(y.cpu().numpy())

            prediction = np.stack(prediction)
            actual = np.stack(actual)

            R, VE, R_square = get_stat_data(selected_idx, prediction, actual)
            R_all_neuron, VE_all_neuron, R_square_all_neuron = get_stat_data(range(num_neurons), prediction, actual)

            avg_VE.append(np.average(VE))
            avg_R.append(np.average(R))
            avg_R_square.append(np.average(R_square))

            avg_R_all_neuron.append(np.average(R_all_neuron))
            avg_VE_all_neuron.append(np.average(VE_all_neuron))
            avg_R_square_all_neuron.append(np.average(R_square_all_neuron))

            all_R.append(R)
            all_VE.append(VE)
            all_R_square.append(R_square)
            labels.append((i + 1) * 44000 / 10)

    plt.plot(labels, all_R, alpha=0.3, color='gray')
    plt.plot(labels, avg_R, alpha=1, color='black')
    plt.plot(labels, avg_R_all_neuron, alpha=0.5, color='blue')
    plt.savefig(directory_base + 'R_curve_selected_40')
    plt.show()
    plt.plot(labels, all_VE, alpha=0.3, color='gray')
    plt.plot(labels, avg_VE, alpha=1, color='black')
    plt.plot(labels, avg_VE_all_neuron, alpha=0.5, color='blue')
    plt.savefig(directory_base + 'VE_curve_selected_40')
    plt.show()
    plt.plot(labels, all_R_square, alpha=0.3, color='gray')
    plt.plot(labels, avg_R_square, alpha=1, color='black')
    plt.plot(labels, avg_R_square_all_neuron, alpha=0.5, color='blue')
    plt.savefig(directory_base + 'RS_curve_selected_40')
    plt.show()
