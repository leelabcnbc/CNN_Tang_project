import numpy as np




def crop(img, size):

    edge_size = (50 - size) // 2
    return img[edge_size:edge_size + size,edge_size:edge_size + size]
sites = ['m1s1', 'm1s2', 'm1s3', 'm2s1', 'm2s2', 'm3s1']
sizes = [15, 25, 35]

for site in sites:
    train_x = np.load('../data/Processed_Tang_data/all_sites_data_prepared/pics_data/train_img_' + site + '.npy')
    val_x = np.load('../data/Processed_Tang_data/all_sites_data_prepared/pics_data/val_img_' + site + '.npy')

    for size in sizes:
        val_x_new = np.stack([crop(x, size) for x in val_x])
        train_x_new = np.stack([crop(x, size) for x in train_x])
        np.save(f'../data/cropped_tang_data/c_train_img_site_{site}_size_{size}', train_x_new)
        np.save(f'../data/cropped_tang_data/c_val_img_site_{site}_size_{size}', val_x_new)
