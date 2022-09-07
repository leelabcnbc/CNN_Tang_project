import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from modeling.models.bethge import BethgeModel
from modeling.train_utils import array_to_dataloader


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



nb_validation_samples = 1000
val_y = np.load('../data/Processed_Tang_data/valRsp.npy')
batch_size = 2048
num_neurons = 299
val_x = np.load('../data/Processed_Tang_data/val_x.npy')
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

val_loader = array_to_dataloader(val_x, val_y, batch_size=200)




with torch.no_grad():
    net = BethgeModel(channels=channels, num_layers=num_layers, input_size=input_size,
                      output_size=output_size, first_k=first_k, later_k=later_k,
                      input_channels=1, pool_size=pool_size, factorized=True,
                      num_maps=num_maps).cuda()

    net.to(device)
    net.load_state_dict(torch.load('../saved_models/Sample_size_test_models/9_10_model_shared_core_256_9_cut'))
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


selected_idx = np.argsort(R)[::-1][:40]


num_laps = 10
labels = []
all_R = []
all_VE = []
all_R_square = []
avg_R = []
avg_VE = []
avg_R_square = []
for i in range(num_laps):
    with torch.no_grad():
        net = BethgeModel(channels=channels, num_layers=num_layers, input_size=input_size,
                          output_size=output_size, first_k=first_k, later_k=later_k,
                          input_channels=1, pool_size=pool_size, factorized=True,
                          num_maps=num_maps).cuda()

        net.to(device)
        net.load_state_dict(torch.load('../saved_models/Sample_size_test_models/' + str(i) + '_10_model_shared_core_256_9_cut'))
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
        R = []
        VE = []
        R_square = []
        for neuron in selected_idx:
            pred1 = prediction[:, neuron]
            val_y = actual[:, neuron]

            u2 = np.zeros((2, nb_validation_samples))
            u2[0, :] = np.reshape(pred1, (nb_validation_samples))
            u2[1, :] = np.reshape(val_y, (nb_validation_samples))

            c2 = np.corrcoef(u2)
            R.append(c2[0, 1])

            VE.append(1 - np.var(pred1 - val_y) / np.var(val_y))

            rsquare, _ = adjust_R_square(pred1,val_y)
            R_square.append(rsquare)

        avg_VE.append(np.average(VE))
        avg_R.append(np.average(R))
        avg_R_square.append(np.average(R_square))
        all_R.append(R)
        all_VE.append(VE)
        all_R_square.append(R_square)
        labels.append((i + 1) * 44000 / 10)

plt.plot(labels, all_R, alpha=0.3, color='gray')
plt.plot(labels, avg_R, alpha=1, color='black')
plt.savefig('../analysis/Graphs/R_curve_40_cut')
plt.show()
plt.plot(labels, all_VE, alpha=0.3, color='gray')
plt.plot(labels, avg_VE, alpha=1, color='black')
plt.savefig('../analysis/Graphs/VE_curve_40_cut')
plt.show()
plt.plot(labels, all_R_square, alpha=0.3, color='gray')
plt.plot(labels, avg_R_square, alpha=1, color='black')
plt.savefig('../analysis/Graphs/R_square_curve_40_cut')
plt.show()
