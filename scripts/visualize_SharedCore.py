import torch
import torch.nn as nn
from tqdm import tqdm

from visualization_utils.wrappers import *
from modeling.models.bethge import *
import matplotlib.pyplot as plt

nb_validation_samples = 1000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_top_corr_neuron_idx(num_neurons, val_loader, device, top_num, net):
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
    for neuron in range(num_neurons):
        pred1 = prediction[:, neuron]
        val_y = actual[:, neuron]
        y_arg = np.argsort(val_y)

        u2 = np.zeros((2, nb_validation_samples))
        u2[0, :] = np.reshape(pred1, (nb_validation_samples))
        u2[1, :] = np.reshape(val_y, (nb_validation_samples))

        c2 = np.corrcoef(u2)
        R[neuron] = c2[0, 1]

        VE[neuron] = 1 - np.var(pred1 - val_y) / np.var(val_y)
    selected_idx = np.argsort(R)[::-1][:top_num]
    return selected_idx


def visualize_neuron(neuron, input_size, all_models, directory, save=False):
    # if multiple models are considered, then load each model one by one. Each neuron is contained in one folder, with
    # pictures equal to the number of models
    for model_num, net in enumerate(all_models):
        newimg, loss, best_loss = visualize(net, net.fc[1], (0, neuron), img_shape=(1, input_size, input_size),
                                            init_range=(0, 1), max_iter=1000, lr=np.linspace(3, 0.5, 1000),
                                            sigma=np.linspace(3, 1, 1000),
                                            debug=False)
        # plt.plot(loss)
        # plt.show()
        with torch.no_grad():
            newimg -= newimg.min()
            newimg /= (newimg.max() - newimg.min())
            img = newimg.detach().cpu().numpy().squeeze() * 256
            if (save):
                imsave(fname=directory + '/' + str(neuron) + '_' + str(model_num) + '_' + '.jpg', arr=img)
            else:
                plt.imshow(img)

def visualize_neuron_single(neuron, input_size, net, directory, name,save=False):
    # if multiple models are considered, then load each model one by one. Each neuron is contained in one folder, with
    # pictures equal to the number of models
    newimg, loss, best_loss = visualize(net, net.fc[1], (0, neuron), img_shape=(1, input_size, input_size),
                                        init_range=(0, 1), max_iter=1000, lr=np.linspace(3, 0.5, 1000),
                                        sigma=np.linspace(3, 1, 1000),
                                        debug=False)
    # plt.plot(loss)
    # plt.show()
    with torch.no_grad():
        newimg -= newimg.min()
        newimg /= (newimg.max() - newimg.min())
        img = newimg.detach().cpu().numpy().squeeze() * 256
        if (save):
            imsave(fname=directory + '/' + str(neuron) + '_' + name + '_' + '.jpg', arr=img)
        else:
            plt.imshow(img)

def visualize_target(target, input_size, net, directory, save=False):
    # if multiple models are considered, then load each model one by one. Each neuron is contained in one folder, with
    # pictures equal to the number of models
    newimg, loss, best_loss = visualize(net, net.fc[1], None, img_shape=(1, input_size, input_size),
                                        init_range=(0, 1), max_iter=1000, lr=np.linspace(3, 0.5, 1000),
                                        sigma=np.linspace(3, 1, 1000), target=target,
                                        debug=False)
    # plt.plot(loss)
    # plt.show()
    with torch.no_grad():
        newimg -= newimg.min()
        newimg /= (newimg.max() - newimg.min())
        img = newimg.detach().cpu().numpy().squeeze() * 256
        if (save):
            imsave(fname=directory + '/reconstruct' + '_' + '.jpg', arr=img)
        else:
            plt.imshow(img)
