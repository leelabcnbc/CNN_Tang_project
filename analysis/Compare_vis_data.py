import torch.nn as nn
import torch
import numpy as np
from tqdm import tqdm
from modeling.models.bethge import BethgeModel
from modeling.train_utils import array_to_dataloader
from modeling.models.SCNN import net_one_neuron
from skimage.transform import resize
from skimage.io import imread

nb_validation_samples = 1000
nb_training_samples = 49000
# val_y = np.load('../data/Processed_Tang_data/valRsp.npy')
# val_x = np.load('../data/Processed_Tang_data/val_x.npy')
# train_x = np.load('../data/Processed_Tang_data/train_x.npy')
# train_y = np.load('../data/Processed_Tang_data/Rsp.npy')
batch_size = 2048
num_neurons = 299

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

channels = 256
num_layers = 9
input_size = 50
output_size = 299
first_k = 9
later_k = 3
pool_size = 2
factorized = True
num_maps = 1

net_shared_core = BethgeModel(channels=channels, num_layers=num_layers, input_size=input_size,
                              output_size=output_size, first_k=first_k, later_k=later_k,
                              input_channels=1, pool_size=pool_size, factorized=True,
                              num_maps=num_maps).cuda()

net_shared_core.to(device)
net_shared_core.load_state_dict(torch.load('../saved_models/shared_core_256_9_model'))
net_shared_core.eval()

channels = 300
num_layers = 12
input_size = 50
output_size = 299
first_k = 5
later_k = 3
pool_size = 2
factorized = True
num_maps = 1
net_shared_core_300 = BethgeModel(channels=channels, num_layers=num_layers, input_size=input_size,
                              output_size=output_size, first_k=first_k, later_k=later_k,
                              input_channels=1, pool_size=pool_size, factorized=True,
                              num_maps=num_maps).cuda()

net_shared_core_300.to(device)
net_shared_core_300.load_state_dict(torch.load('../saved_models/shared_core_300_12_model'))
net_shared_core_300.eval()

net_SCNN = nn.ModuleList([net_one_neuron() for i in range(num_neurons)])
net_SCNN.to(device)
net_SCNN.load_state_dict(torch.load('../saved_models/model_mix_corr_MAE_ep40'))


def get_vis_img(path, neuron, format):
    vis_img = imread(path + str(neuron) + format, as_gray=True)
    if vis_img.shape[0] > 50 :
        vis_img = resize(vis_img, [50, 50], anti_aliasing=True)
    vis_img = torch.reshape(torch.tensor(vis_img).type(torch.FloatTensor), (1, 1, 50, 50)).to(device)
    vis_img -= vis_img.min()
    vis_img /= (vis_img.max() - vis_img.min())
    return vis_img


shared_core_vis_rsp = np.zeros((5, 299))
shared_core_300_vis_rsp = np.zeros((5, 299))
SCNN_vis_rsp = np.zeros((5, 299))

for neuron in range(num_neurons):
    vis_256_shared = get_vis_img('Visualization/shared_core/low_mod_256/', neuron, '.jpg')
    vis_300_shared = get_vis_img('Visualization/shared_core/low_mod_300/', neuron, '.jpg')
    vis_tf = get_vis_img('Visualization/Tensorflow/Modified_vis_imgs/', neuron + 1, '.bmp')
    vis_scnn_a = get_vis_img('Visualization/SCNN/visualization_seperate_cnn/', neuron, '.jpg')
    vis_scnn_b = get_vis_img('Visualization/SCNN/Origin_Visualization_imgs/', neuron + 1, '.png')
    vis = [vis_256_shared, vis_300_shared, vis_tf, vis_scnn_a, vis_scnn_b]
    subnet_SCNN = net_SCNN[neuron]
    subnet_SCNN.eval()
    # pass into models
    for i, img in enumerate(vis):
        rsp_sharedcore = net_shared_core(img)[:, neuron]
        rsp_sharedcore_300 = net_shared_core_300(img)[:, neuron]
        rsp_SCNN = subnet_SCNN(img)
        shared_core_vis_rsp[i][neuron] = rsp_sharedcore.detach().cpu().numpy()
        shared_core_300_vis_rsp[i][neuron] = rsp_sharedcore_300.detach().cpu().numpy()
        SCNN_vis_rsp[i][neuron] = rsp_SCNN.detach().cpu().numpy()
np.save('../data/tmp_data/shared_core_vis_rsp.npy', shared_core_vis_rsp)
np.save('../data/tmp_data/SCNN_vis_rsp.npy', SCNN_vis_rsp)
np.save('../data/tmp_data/shared_core_300_vis_rsp.npy', shared_core_300_vis_rsp)
