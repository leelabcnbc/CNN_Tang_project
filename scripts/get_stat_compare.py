import torch
import torch.nn as nn
from tqdm import tqdm

import matplotlib.pyplot as plt
from visualization_utils.wrappers import *
from modeling.models.bethge import *
from analysis.Stat_utils import get_tuning_stat
from modeling.train_utils import array_to_dataloader

def make_dir_try(dir):
    try:
        os.makedirs(dir, exist_ok=True)
        print("Directory '%s' created successfully" % dir)
    except OSError as error:
        print("Directory '%s' can not be created" % dir)


nb_validation_samples = 1000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


num_laps = 10
sites = [ 'm1s1', 'm1s2', 'm1s3','m2s1', 'm2s2', 'm3s1']


for site in sites:

    vis_path = 'A:/school/College_Junior/research/CNN_Tang_project/analysis/Visualization/Shared_Core_avg_border_vis/' + site + '/'
    path = 'A:/school/College_Junior/research/CNN_Tang_project/analysis/Graphs/shared_core_tuning_with_rsp/' + site + '/'
    val_x = np.load('../data/Processed_Tang_data/all_sites_data_prepared/pics_data/val_img_' + site + '.npy')
    val_y = np.load('../data/Processed_Tang_data/all_sites_data_prepared/New_response_data/valRsp_' + site + '.npy')
    val_x = np.transpose(val_x, (0, 3, 1, 2))
    num_neurons = val_y.shape[1]
    val_loader = array_to_dataloader(val_x, val_y, batch_size=200)

    channels = 256
    num_layers = 9
    input_size = 50
    output_size = num_neurons
    first_k = 9
    later_k = 3
    pool_size = 2

    final_model = BethgeModel(channels=channels, num_layers=num_layers, input_size=input_size,
                              output_size=num_neurons, first_k=first_k, later_k=later_k,
                              input_channels=1, pool_size=2, factorized=True,
                              num_maps=1)
    final_model.to(device)
    # only get neurons with valid labels
    final_model.load_state_dict(
        torch.load('../saved_models/Sample_size_test_models/' + '9_10_' + site + '_shared_core_256_9'))
    final_model.eval()
    make_dir_try(path)
    get_tuning_stat(final_model,device,val_loader,range(num_neurons),path,None,num_neurons,use_vis_rsp=True,vis_directory=vis_path)


