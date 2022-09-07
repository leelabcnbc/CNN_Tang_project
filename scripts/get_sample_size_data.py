import torch
import torch.nn as nn
from tqdm import tqdm

import matplotlib.pyplot as plt
from visualization_utils.wrappers import *
from modeling.models.bethge import *
from modeling.train_utils import array_to_dataloader
from scripts.visualize_SharedCore import visualize_neuron_single
from scripts.pattern_stimulus_test import get_label_pattern_stimuli
from analysis.Stat_utils import get_tuning_stat, get_top_imgs
from torchvision import transforms
from cv2 import resize

def make_dir_try(dir):
    try:
        os.makedirs(dir, exist_ok=True)
        print("Directory '%s' created successfully" % dir)
    except OSError as error:
        print("Directory '%s' can not be created" % dir)
def process_spatial_w_avg(net):
    spatial_w = net.fc[0].bank[0].weight_spatial.data

    for i, w in enumerate(spatial_w):
        sum = 0
        sum += torch.sum(w[:1,:])
        sum += torch.sum(w[:,20:])
        sum += torch.sum(w[20:,:])
        sum += torch.sum(w[:,:1])
        sum -= w[0,0]
        sum -= w[0,20]
        sum -= w[20,20]
        sum -= w[20,0]
        sum /= 80
        w[:1,:] = sum
        w[:,20:] = sum
        w[20:,:] = sum
        w[:,:1] = sum
        spatial_w[i] = w
    net.fc[0].bank[0].weight_spatial.data = spatial_w
    return net

nb_validation_samples = 1000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
channels = 256
num_layers = 9
input_size = 50
output_size = 299
first_k = 9
later_k = 3
pool_size = 2

num_laps = 10
sites = [ 'm1s1', 'm1s2', 'm1s3','m2s1', 'm2s2', 'm3s1']
sample_size_indexs = [ '9']

for site in sites:
    path = 'A:/school/College_Junior/research/CNN_Tang_project/analysis/Visualization/sample_size_comparison/' + site + '/'
    new_path = 'A:/school/College_Junior/research/CNN_Tang_project/analysis/Visualization/Shared_Core_avg_border_vis/' + site + '/'
    val_y = np.load('../data/Processed_Tang_data/all_sites_data_prepared/New_response_data/valRsp_' + site + '.npy')
    num_neurons = val_y.shape[1]

    for size_idx in sample_size_indexs:
        final_model = BethgeModel(channels=channels, num_layers=num_layers, input_size=input_size,
                                  output_size=num_neurons, first_k=first_k, later_k=later_k,
                                  input_channels=1, pool_size=2, factorized=True,
                                  num_maps=1)
        final_model.to(device)
        # only get neurons with valid labels
        final_model.load_state_dict(
            torch.load('../saved_models/Sample_size_test_models/' + size_idx + '_10_' + site + '_shared_core_256_9'))
        final_model = process_spatial_w_avg(final_model)
        final_model.eval()
        # Pattern_stimulus_data = np.load('../data/Processed_pattern_stimuli/crop_100_resize_3.npy')
        # Pattern_stimulus_label = np.load('../data/Processed_pattern_stimuli/labels.npy')
        # selected_idx, neuron_labels, top_10_ps_idx, top_10_ps_rsp = get_label_pattern_stimuli(Pattern_stimulus_data,
        #                                                                                       Pattern_stimulus_label,
        #                                                                                       256, device, final_model)

        for neuron in range(num_neurons):
            # neuron_ps_idx = top_10_ps_idx[neuron]
            # neuron_ps_rsp = top_10_ps_rsp[neuron]
            directory = path + str(neuron) + '/'
            # for img_idx, img_rsp in zip(neuron_ps_idx, neuron_ps_rsp):
            #     img = Pattern_stimulus_data[img_idx][0]
            #     make_dir_try(directory + '/pattern_stimulus_' + size_idx + '_' + neuron_labels[neuron] + '/')
            #     imsave(directory + '/pattern_stimulus_' + size_idx + '_' + neuron_labels[neuron] + '/' + str(img_rsp) + '.jpg', img * 256)

            #visualize_neuron_single(neuron, input_size, final_model, directory, size_idx, save=True)

            vis = imread(directory + '/' + str(neuron) + '_' + size_idx + '_.jpg')
            directory = new_path
            make_dir_try(directory)
            imsave(directory + '/' + str(neuron) + '.jpg', vis)
