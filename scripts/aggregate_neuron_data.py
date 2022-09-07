import torch
import torch.nn as nn
from tqdm import tqdm

import matplotlib.pyplot as plt
from visualization_utils.wrappers import *
from modeling.models.bethge import *
from modeling.train_utils import array_to_dataloader
from scripts.visualize_SharedCore import visualize_neuron
from scripts.pattern_stimulus_test import get_label_pattern_stimuli
from analysis.Stat_utils import get_tuning_stat, get_top_imgs
from torchvision import transforms

def make_dir_try(dir):
    try:
        os.makedirs(dir, exist_ok=True)
        print("Directory '%s' created successfully" % dir)
    except OSError as error:
        print("Directory '%s' can not be created" % dir)


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
sites = [ 'm1s1', 'm1s2', 'm1s3', 'm2s2', 'm3s1']

for site in sites:
    val_x = np.load('../data/Processed_Tang_data/all_sites_data_prepared/pics_data/val_img_' + site + '.npy')
    val_y = np.load('../data/Processed_Tang_data/all_sites_data_prepared/New_response_data/valRsp_' + site + '.npy')
    val_x = np.transpose(val_x, (0, 3, 1, 2))
    num_neurons = val_y.shape[1]
    val_loader = array_to_dataloader(val_x, val_y, batch_size=200)
    final_model = BethgeModel(channels=channels, num_layers=num_layers, input_size=input_size,
                              output_size=num_neurons, first_k=first_k, later_k=later_k,
                              input_channels=1, pool_size=2, factorized=True,
                              num_maps=1)
    final_model.to(device)
    # only get neurons with valid labels
    final_model.load_state_dict(
        torch.load('../saved_models/Sample_size_test_models/' + '9_10_' + site + '_shared_core_256_9'))
    final_model.eval()
    Pattern_stimulus_data = np.load('../data/Processed_pattern_stimuli/crop_100_resize_3.npy')
    Pattern_stimulus_label = np.load('../data/Processed_pattern_stimuli/labels.npy')
    selected_idx, neuron_labels, top_10_ps_idx, top_10_ps_rsp = get_label_pattern_stimuli(Pattern_stimulus_data, Pattern_stimulus_label, 256,
                                                                                          device, final_model)
    models_locations = []
    for lap in range(10):
        model_location = str(lap) + '_10_' + site + '_shared_core_256_9'
        models_locations.append(model_location)
    path = 'A:/school/College_Junior/research/CNN_Tang_project/analysis/Visualization/Aggregated_data/' + site + '/'

    num_models = len(models_locations)
    all_models = nn.ModuleList \
            (
            [
                BethgeModel(channels=channels, num_layers=num_layers, input_size=input_size,
                            output_size=num_neurons, first_k=first_k, later_k=later_k,
                            input_channels=1, pool_size=2, factorized=True,
                            num_maps=1)
                for i in range(num_models)
            ]
        )
    for i, model_location in enumerate(models_locations):
        all_models[i].to(device)
        full_location = '../saved_models/Sample_size_test_models/' + model_location
        all_models[i].load_state_dict(torch.load(full_location))
        all_models[i].eval()

    for i, neuron in tqdm(enumerate(selected_idx)):
        directory = path + str(neuron) + '_' + neuron_labels[i]
        make_dir_try(directory)

    # now for tuning curve
    get_tuning_stat(final_model, device, val_loader, selected_idx, path, neuron_labels, num_neurons)

    # now for getting neuron receptive field data

    g = np.zeros((299, 50, 50))
    avg_img = np.zeros((50, 50))
    for i, img in enumerate(tqdm(Pattern_stimulus_data[:1600])):
        affine_transfomer = transforms.RandomAffine(degrees=(30, 70), translate=(0.5, 0.5), scale=(0.5, 0.75))
        img_t = torch.FloatTensor(img).to(device)
        for j in range(4):
            img_t = affine_transfomer(img_t)
            actual_img = img_t.detach().cpu().numpy()
            avg_img += actual_img[0] / 6400
            input = torch.reshape(img_t, (1, 1, 50, 50))
            rsp = final_model(input).detach().cpu().numpy()
            for neuron in range(299):
                g[neuron, :, :] += (rsp[:, neuron] * actual_img[0]) / 6400

    for i, neuron in tqdm(enumerate(selected_idx)):
        directory = path + str(neuron) + '_' + neuron_labels[i]
        # save receptive field data
        plt.imsave(directory + '/receptive_field.jpg', g[neuron], cmap='gray')
        # now for top stimulus
        top_stimulus_sub_directory = directory + '/' + 'top_stimulus'
        make_dir_try(top_stimulus_sub_directory)
        get_top_imgs(np.reshape(val_x, (val_x.shape[0],50,50)), val_y, neuron, top_stimulus_sub_directory)
        # now for pattern stimulus pictures
        Pattern_stimulus_sub_directory = directory + '/' + 'Pattern_stimulus'
        make_dir_try(Pattern_stimulus_sub_directory)
        neuron_top = top_10_ps_idx[neuron]
        neuron_top_rsp = top_10_ps_rsp[neuron]
        for img_idx, img_rsp in zip(neuron_top, neuron_top_rsp):
            img = Pattern_stimulus_data[img_idx][0]
            imsave(Pattern_stimulus_sub_directory + '/' + str(img_rsp) + '.jpg', img * 256)
        # now for visualization
        Visualization_sub_directory = directory + '/' + 'Visualization_change'
        make_dir_try(Visualization_sub_directory)
        visualize_neuron(neuron, input_size, all_models, Visualization_sub_directory)

