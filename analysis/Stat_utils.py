import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from skimage.io import imread, imsave
from modeling.train_utils import array_to_dataloader

def get_top_imgs(val_imgs, val_rsp, neuron, directory):
    val_rsp_neuron = val_rsp[:, neuron]
    top_arg = np.argsort(val_rsp_neuron)[::-1][:10]
    top_imgs = val_imgs[top_arg]
    for i, top_img in enumerate(top_imgs):
        imsave(fname=directory + '/' + str(i) + '.jpg', arr=top_img)

def get_site_corr(net, device, site):
    val_x = np.load('../data/Processed_Tang_data/all_sites_data_prepared/pics_data/val_img_' + site + '.npy')
    val_y = np.load('../data/Processed_Tang_data/all_sites_data_prepared/New_response_data/valRsp_' + site + '.npy')
    val_x = np.transpose(val_x, (0, 3, 1, 2))
    num_neurons = val_y.shape[1]
    val_loader = array_to_dataloader(val_x, val_y, batch_size=200)
    prediction = []
    actual = []
    for batch_num, (x, y) in enumerate(tqdm(val_loader)):
        x, y = x.to(device), y.to(device)
        outputs = net(x).detach().cpu().numpy()
        prediction.extend(outputs)
        actual.extend(y.detach().cpu().numpy())

    prediction = np.stack(prediction)
    actual = np.stack(actual)
    R = np.zeros(num_neurons)
    VE = np.zeros(num_neurons)
    SR = np.zeros(num_neurons)
    for neuron in tqdm(range(num_neurons)):
        pred1 = prediction[:, neuron]
        val_y = actual[:, neuron]
        y_arg = np.argsort(val_y)

        u2 = np.zeros((2, 1000))
        u2[0, :] = np.reshape(pred1, (1000))
        u2[1, :] = np.reshape(val_y, (1000))

        c2 = np.corrcoef(u2)
        R[neuron] = c2[0, 1]
        SR[neuron] = spearmanr(pred1, val_y)[0]
        VE[neuron] = 1 - np.var(pred1 - val_y) / np.var(val_y)
    return R

def get_tuning_stat(net, device, val_loader, selected_idx, directory, neuron_extra_name, num_neurons, use_vis_rsp = False, vis_directory = '', ):
    prediction = []
    actual = []
    for batch_num, (x, y) in enumerate(tqdm(val_loader)):
        x, y = x.to(device), y.to(device)
        outputs = net(x).detach().cpu().numpy()
        prediction.extend(outputs)
        actual.extend(y.detach().cpu().numpy())

    prediction = np.stack(prediction)
    actual = np.stack(actual)
    R = np.zeros(num_neurons)
    VE = np.zeros(num_neurons)
    SR = np.zeros(num_neurons)
    for i, neuron in tqdm(enumerate(selected_idx)):
        if use_vis_rsp:
            vis_img = plt.imread(vis_directory + "/" + str(neuron) + '.jpg')
            vis_img = torch.reshape(torch.tensor(vis_img).type(torch.FloatTensor), (1, 1, 50, 50)).to(device)
            vis_img -= vis_img.min()
            vis_img /= (vis_img.max() - vis_img.min())
            vis_rsp = net(vis_img).detach().cpu().numpy()

        pred1 = prediction[:, neuron]
        val_y = actual[:, neuron]
        y_arg = np.argsort(val_y)

        u2 = np.zeros((2, 1000))
        u2[0, :] = np.reshape(pred1, (1000))
        u2[1, :] = np.reshape(val_y, (1000))

        c2 = np.corrcoef(u2)
        R[neuron] = c2[0, 1]
        SR[neuron] = spearmanr(pred1, val_y)[0]
        VE[neuron] = 1 - np.var(pred1 - val_y) / np.var(val_y)

        predict_curve = pred1[y_arg]
        actual_curve = np.sort(val_y)

        diff = (predict_curve - actual_curve) / (np.max(actual_curve) - np.min(actual_curve))
        top_diff = np.mean(diff[::-1][10])

        plt.figure()
        plt.plot(predict_curve, label='pred')
        plt.plot(actual_curve, label='actual')
        if use_vis_rsp:
            plt.axhline(y=vis_rsp[:, neuron].item(), color='r', linestyle='-')
        plt.title("corr=" + str(R[neuron]) + " VE=" + str(VE[neuron]))
        plt.legend()
        if neuron_extra_name is None:
            plt.savefig(directory + str(neuron))
        else:
            plt.savefig(directory + str(neuron) + '_' + neuron_extra_name[i])

    # selected_idx_SR = np.argsort(SR)[::-1]
    # np.save('selected_idx_SR', selected_idx_SR)
    # print(SR)
    # print(np.mean(SR))
    # print(np.mean(R))
    # print(np.mean(VE))
