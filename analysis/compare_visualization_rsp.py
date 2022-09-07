import numpy as np
import matplotlib.pyplot as plt

val_set = np.load('../data/tmp_data/valRsp_m2s1.npy')

shared_core_data = np.load('../data/tmp_data/shared_core_vis_rsp.npy')
shared_core_300_data = np.load('../data/tmp_data/shared_core_300_vis_rsp.npy')
SCNN_data = np.load('../data/tmp_data/SCNN_vis_rsp.npy')
tf_data = np.load('../data/tmp_data/tf_vis_rsp.npy')

model_labels = ["SH_256_9", "SH_300_data", "SCNN_torch", "SCNN_tf"]
vis_labels = ['256_SH_vis', '300_SH_vis', 'SCNN_TF_vis', 'SCNN_torch1', 'SCNN_torch2']
combined_data = np.transpose(np.stack([shared_core_data, shared_core_300_data, SCNN_data, tf_data]), (1, 0, 2))

average_data = np.zeros((5,4))
std_data = np.zeros((5,4))
for vis in range(5):
    for model in range(4):
        all_values = np.zeros(299)
        for neuron in range(299):
            max_rsp = np.max(val_set[:, neuron])
            min_rsp = np.min(val_set[:, neuron])
            rsp = combined_data[vis, model, neuron]
            metric = (rsp - max_rsp) / (max_rsp - min_rsp)
            combined_data[vis, model, neuron] = metric
            all_values[neuron] = metric
        average_data[vis, model] = np.mean(all_values)
        std_data[vis, model] = np.std(all_values)
for vis in range(5):
    plt.figure()
    plt.ylim([-3, 12])
    for model in range(4):
        plt.plot(combined_data[vis, model], label=model_labels[model])
    plt.title(vis_labels[vis])
    plt.legend()
    plt.savefig('Graphs/' + vis_labels[vis])
    plt.show()

for vis in range(5):
    plt.figure()
    plt.ylim([-0.8, 1.8])
    plt.bar(model_labels, average_data[vis])
    average_value = np.average(average_data[vis])
    plt.title(vis_labels[vis] + '   Average:' + str(average_value))
    #plt.legend()
    plt.savefig('Graphs/' + vis_labels[vis] + '_average')
    plt.show()

for vis in range(5):
    plt.figure()
    plt.ylim([0, 2])
    plt.bar(model_labels, std_data[vis])
    average_value = np.average(std_data[vis])
    plt.title(vis_labels[vis] + '   Average_std:' + str(average_value))
    #plt.legend()
    plt.savefig('Graphs/' + vis_labels[vis] + '_std')
    plt.show()
#
for model in range(4):
    plt.figure()
    plt.ylim([-3, 12])
    for vis in range(5):
        plt.plot(combined_data[vis, model], label=vis_labels[vis])
    plt.title(model_labels[model])
    plt.legend()
    plt.savefig('Graphs/' + model_labels[model])
    plt.show()

for model in range(4):
    plt.figure()
    plt.ylim([-0.8, 1.8])
    plt.bar(vis_labels, average_data[:, model])
    average_value = np.average(average_data[:, model])
    plt.title(model_labels[model] + '   Average:' + str(average_value))
    plt.savefig('Graphs/' + model_labels[model]+ '_average')
    plt.show()

for model in range(4):
    plt.figure()
    plt.ylim([0, 2])
    plt.bar(vis_labels, std_data[:, model])
    average_value = np.average(std_data[:, model])
    plt.title(model_labels[model] + '   Average_std:' + str(average_value))
    plt.savefig('Graphs/' + model_labels[model]+ '_std')
    plt.show()