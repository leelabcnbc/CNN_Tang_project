import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from modeling.models.bethge import BethgeModel
from modeling.train_utils import array_to_dataloader
from scipy.stats import spearmanr
from scripts.visualize_SharedCore import visualize_neuron
import os

nb_validation_samples = 1000
nb_training_samples = 49000
train_x = np.load('../data/Processed_Tang_data/all_sites_data_prepared/pics_data/train_img_m2s1.npy')
val_x = np.load('../data/Processed_Tang_data/all_sites_data_prepared/pics_data/val_img_m2s1.npy')
train_y = np.load('../data/Processed_Tang_data/all_sites_data_prepared/New_response_data/trainRsp_m2s1.npy')
val_y = np.load('../data/Processed_Tang_data/all_sites_data_prepared/New_response_data/valRsp_m2s1.npy')
train_x = np.transpose(train_x, (0, 3, 1, 2))
val_x = np.transpose(val_x, (0, 3, 1, 2))
batch_size = 2048
num_neurons = 299

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

channels = 100
num_layers = 9
input_size = 50
output_size = 299
first_k = 9
later_k = 3
pool_size = 2
factorized = True
num_maps = 1

net = BethgeModel(channels=channels, num_layers=num_layers, input_size=input_size,
                  output_size=output_size, first_k=first_k, later_k=later_k,
                  input_channels=1, pool_size=pool_size, factorized=True,
                  num_maps=num_maps).cuda()

net.to(device)
net.load_state_dict(torch.load('../saved_models/C_' + str(channels) + '_L_' + str(num_layers) + '_model_shared_core'))

val_loader = array_to_dataloader(val_x, val_y, batch_size=200)
train_loader = array_to_dataloader(train_x, train_y, batch_size=200)

selected_idx = np.load('../data/tmp_data/selected_idx_R.npy')[:10]

with torch.no_grad():
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
    R = np.zeros(299)
    VE = np.zeros(299)
    SR = np.zeros(299)
    for neuron in (range(num_neurons)):
        # vis_img = plt.imread("corr+mae/" + str(neuron) + '.jpg')
        # vis_img = torch.reshape(torch.tensor(vis_img).type(torch.FloatTensor), (1, 1, 50, 50)).to(device)
        # vis_img -= vis_img.min()
        # vis_img /= (vis_img.max() - vis_img.min())
        # vis_rsp = net(vis_img).cpu().numpy()

        pred1 = prediction[:, neuron]
        val_y = actual[:, neuron]
        y_arg = np.argsort(val_y)

        u2 = np.zeros((2, nb_validation_samples))
        u2[0, :] = np.reshape(pred1, (nb_validation_samples))
        u2[1, :] = np.reshape(val_y, (nb_validation_samples))

        c2 = np.corrcoef(u2)
        R[neuron] = c2[0, 1]
        SR[neuron] = spearmanr(pred1, val_y)[0]
        VE[neuron] = 1 - np.var(pred1 - val_y) / np.var(val_y)

        # plt.plot(pred1[y_arg], label='pred')
        # plt.plot(np.sort(val_y), label='actual')
        # #plt.axhline(y=vis_rsp[:,neuron].item(), color='r', linestyle='-')
        # plt.title(str(R[neuron]) + " " + str(SR[neuron]))
        # plt.legend()
        # plt.savefig('Graphs/test/' + str(neuron) + 'tuning')
        # plt.show()

    # selected_idx_SR = np.argsort(SR)[::-1]
    # np.save('selected_idx_SR', selected_idx_SR)
    print(R)
    print(np.mean(SR))
    print(np.mean(R))
    print(np.mean(VE))


def make_dir_try(dir):
    try:
        os.makedirs(dir, exist_ok=True)
        print("Directory '%s' created successfully" % dir)
    except OSError as error:
        print("Directory '%s' can not be created" % dir)


directory = 'A:/school/College_Junior/research/CNN_Tang_project/analysis/Visualization/' + str(channels) + '_' + str(
    num_layers)
make_dir_try(directory)

top_idx = np.argsort(R)
selected_idx = top_idx[::-1][:20]
for neuron in range(20):
    visualize_neuron(neuron, input_size, [net], directory, save=True)
