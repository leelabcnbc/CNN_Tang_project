import numpy as np
import matplotlib.pyplot as plt

all_kept_center = np.load("D:/school/research/CNN_Tang_project/analysis/all_kept_center.npy")
all_kept_surround = np.load("D:/school/research/CNN_Tang_project/analysis/all_kept_surround.npy")
#%%
surround = np.average(all_kept_surround, axis = 0)
center = np.average(all_kept_center, axis = 0)
plt.figure()
plt.plot(np.arange(1,26)*2/15, surround, label='surround only')
plt.plot(np.arange(1,26)*2/15, center, label='center only')
plt.xlabel('Diameter of center aperture in degree visual angle')
plt.ylabel('Average neural response')
plt.legend()
plt.savefig('surround_supression.png')
#%%
def translate_degree(x):
    return x*2/15

def find_minmax_idx(center_rsp):
    all_diff = []
    for neuron in range(center_rsp.shape[0]):
        curve = center_rsp[neuron]
        max_point = np.argmax(curve)
        low_point_beyond = np.argmin(curve[max_point:])
        low_point = low_point_beyond + max_point
        max_point_d = translate_degree(max_point)
        low_point_d = translate_degree(low_point)
        diff = (low_point_d- max_point_d)/low_point_d
        all_diff.append(diff)
