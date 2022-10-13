import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
from visualization_utils.wrappers import *
from modeling.models.bethge import *
from modeling.train_utils import array_to_dataloader
from scripts.visualize_SharedCore import visualize_neuron
from scripts.pattern_stimulus_test import get_label_pattern_stimuli
from analysis.Stat_utils import get_tuning_stat, get_top_imgs
import cv2
from scipy import optimize
from analysis.Stat_utils import get_site_corr
def make_dir_try(dir):
    try:
        os.makedirs(dir, exist_ok=True)
        print("Directory '%s' created successfully" % dir)
    except OSError as error:
        print("Directory '%s' can not be created" % dir)
def gaussian(height, center_x, center_y, width_x, width_y):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x,y: height*np.exp(
                -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)

def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    col = data[:, int(y)]
    width_x = np.sqrt(np.abs((np.arange(col.size)-x)**2*col).sum()/col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(np.abs((np.arange(row.size)-y)**2*row).sum()/row.sum())
    height = data.max()
    return height, x, y, width_x, width_y

def fitgaussian(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    params = moments(data)
    errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) -
                                 data)
    p, success = optimize.leastsq(errorfunction, params)
    return p

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
sites = [ 'm2s1', 'm3s1']
data_partial_1 = np.load(
    'A:/school/College_Junior/research/CNN_Tang_project/data/Processed_pattern_stimuli/crop_100_resize_1.npy')
data_partial_2 = np.load(
    'A:/school/College_Junior/research/CNN_Tang_project/data/Processed_pattern_stimuli/crop_100_resize_2.npy')
data_partial_3 = np.load(
    'A:/school/College_Junior/research/CNN_Tang_project/data/Processed_pattern_stimuli/crop_100_resize_3.npy')
data_partial_4 = np.load(
    'A:/school/College_Junior/research/CNN_Tang_project/data/Processed_pattern_stimuli/crop_100_resize_4.npy')
Pattern_stimulus_data = np.concatenate((data_partial_1, data_partial_2, data_partial_3, data_partial_4), 0)
Pattern_stimulus_label = np.load('../data/Processed_pattern_stimuli/labels.npy')
rc_test_img1 = np.load('A:/school/College_Junior/research/CNN_Tang_project/data/bar_img_matrix/img_matrix_blanktilt45.npy')
rc_test_img2 = np.load('A:/school/College_Junior/research/CNN_Tang_project/data/bar_img_matrix/img_matrix_blanktilt135.npy')
rc_test_img3 = np.load('A:/school/College_Junior/research/CNN_Tang_project/data/bar_img_matrix/img_matrix_blankvertical.npy')
rc_test_img4 = np.load('A:/school/College_Junior/research/CNN_Tang_project/data/bar_img_matrix/img_matrix_blankHorizontal.npy')
rc_test_img5 = np.load('A:/school/College_Junior/research/CNN_Tang_project/data/bar_img_matrix/img_matrix_blanktilt45_reverse.npy')
rc_test_img6 = np.load('A:/school/College_Junior/research/CNN_Tang_project/data/bar_img_matrix/img_matrix_blankHorizontal_reverse.npy')
rc_test_img7 = np.load('A:/school/College_Junior/research/CNN_Tang_project/data/bar_img_matrix/img_matrix_blanktilt135_reverse.npy')
rc_test_img8 = np.load('A:/school/College_Junior/research/CNN_Tang_project/data/bar_img_matrix/img_matrix_blankvertical_reverse.npy')

rc_imgs = np.stack([rc_test_img1,rc_test_img2,rc_test_img3,rc_test_img4,rc_test_img5,rc_test_img6,rc_test_img7,rc_test_img8])
rc_imgs = np.reshape(rc_imgs, (rc_imgs.shape[0],input_size,input_size,rc_imgs.shape[2],rc_imgs.shape[3],rc_imgs.shape[4]))

label_names = ['SS', 'EB', 'CN', 'CV', 'CRS', 'Other', 'MC']
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
        torch.load('../saved_models/new_learned_models/' + site + '_9_model_version_0'))
    final_model.eval()

    neuron_labels, top_ps_idx, top_ps_rsp = get_label_pattern_stimuli(Pattern_stimulus_data, Pattern_stimulus_label, 256,
                                                                                          device, final_model, stacked_data=True)
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

    Corr = get_site_corr(final_model, device, site)
    selected_idx = range(num_neurons)

    for i, neuron in tqdm(enumerate(selected_idx)):
        directory = path + str(neuron)
        make_dir_try(directory)

    # now for tuning curve
    get_tuning_stat(final_model, device, val_loader, selected_idx, path, num_neurons)

    # now for getting neuron receptive field data
    rsp_matrix = np.zeros((rc_imgs.shape[0], num_neurons, 50, 50))
    for img_set, imgs in enumerate(rc_imgs):
        for i in tqdm(range(input_size)):
            for j in range(input_size):
                img = np.reshape(imgs[i, j], (1, 1, 50, 50))
                input = torch.FloatTensor(img).to(device)
                rsp = final_model(input).detach().cpu().numpy()
                a = np.zeros((50, 50))
                for neuron in range(num_neurons):
                    rsp_matrix[img_set, neuron, i, j] = rsp[:, neuron]
    rsp_matrix_sum = np.max(rsp_matrix, axis=0)


    for i, neuron in tqdm(enumerate(selected_idx)):
        directory = path + str(neuron)
        #save label data
        plt.close()
        plt.figure()
        table = plt.table(cellText=[[label_names[label] for label in neuron_labels[:, neuron]]],
                          colLabels= [str(round(1 - (0.5 / 10) * (n + 1),2)) for n in range(10)],
                          loc='center')
        plt.axis('off')
        plt.grid('off')
        plt.gcf().canvas.draw()
        points = table.get_window_extent(plt.gcf()._cachedRenderer).get_points()
        points[0, :] -= 10
        points[1, :] += 10
        nbbox = matplotlib.transforms.Bbox.from_extents(points / plt.gcf().dpi)
        plt.savefig(directory + '/labels.png', bbox_inches=nbbox, )

        # save receptive field data
        data = rsp_matrix_sum[neuron]
        plt.figure(neuron)
        plt.jet()

        plt.matshow(data)
        data -= data.min()
        data /= (data.max() - data.min())
        data = np.abs(cv2.resize(data, (50, 50)))
        plt.colorbar()

        params = fitgaussian(data)
        fit = gaussian(*params)

        ct_data = np.indices(data.shape)
        ct_z = fit(*ct_data)
        plt.contour(ct_z, linewidths=0.5, colors='k')
        ax = plt.gca()
        (height, x, y, width_x, width_y) = params

        plt.text(0.95, 0.05, """
                x : %.1f
                y : %.1f
                width_x : %.1f
                width_y : %.1f""" % (x, y, width_x, width_y),
                 fontsize=16, horizontalalignment='right',
                 verticalalignment='bottom', transform=ax.transAxes)
        plt.savefig(directory+'/receptive_field')
        plt.close()

        # now for top stimulus
        top_stimulus_sub_directory = directory + '/' + 'top_stimulus'
        make_dir_try(top_stimulus_sub_directory)
        get_top_imgs(np.reshape(val_x, (val_x.shape[0],50,50)), val_y, neuron, top_stimulus_sub_directory)
        # now for pattern stimulus pictures
        Pattern_stimulus_sub_directory = directory + '/' + 'Pattern_stimulus'
        make_dir_try(Pattern_stimulus_sub_directory)
        neuron_top = top_ps_idx[neuron][:10]
        neuron_top_rsp = top_ps_rsp[neuron][:10]
        for img_idx, img_rsp in zip(neuron_top, neuron_top_rsp):
            img = Pattern_stimulus_data[img_idx][0] * 256
            imsave(Pattern_stimulus_sub_directory + '/' + str(img_rsp) + '.jpg', img)
        # now for visualization
        if Corr[neuron]>0.5:
            Visualization_sub_directory = directory + '/' + 'Visualization_change'
            make_dir_try(Visualization_sub_directory)
            visualize_neuron(neuron, input_size, all_models,Visualization_sub_directory, save=True)

