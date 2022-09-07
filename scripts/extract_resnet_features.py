### 041220 -- extract features from resnet-34 for each image
### from every residual block (16 total)
### save them to disk for use in transfer learning models
import os
import numpy as np
import torch

from torchvision import models
from skimage.util import crop
from analysis.data_utils import DATA_DIR
from modeling.data_utils import get_images
from modeling.train_utils import array_to_dataloader

# some constants/parameters
IMGNET_MEAN = np.array([[0.485, 0.456, 0.406]])[..., np.newaxis, np.newaxis]
IMGNET_STD = np.array([[0.229, 0.224, 0.225]])[..., np.newaxis, np.newaxis]
BATCH_SIZE = 256 # GPU can maybe fit bigger

# get the network set up
# (downloads the weights if you don't already have them)
net = models.resnet34(pretrained=True).cuda()
blocks = [m for m in net.modules() if isinstance(m, models.resnet.BasicBlock)]
# confirm the length:
print(f'{len(blocks)} layers selected for extraction.')

# treat each dataset separately
for dataset_name in ['googim']:
    print(f'starting on {dataset_name}')
    # get the data and make 'RGB'
    images = get_images(dataset_name, normalize=False)
    images = np.concatenate([images, images, images], axis=1)

    # center-crop to 224x224 (fine, only cropping out aperture)
    crop_size = (images.shape[-1] - 224) // 2
    images = crop(images, ((0, 0), (0, 0), (crop_size, crop_size),
        (crop_size, crop_size)), copy=True)

    # normalize the data according to torchvision standards
    # (0 to 1 range, then imagenet mean+sd)
    images = images - images.min()
    images = images / images.max()
    images = torch.FloatTensor((images - IMGNET_MEAN) / IMGNET_STD)
    # and put into batch-iterable form
    images = torch.utils.data.TensorDataset(images)
    images = torch.utils.data.DataLoader(images, BATCH_SIZE)

    # set up save directory
    if dataset_name == 'googim':
        dir_name = 'google-imagenet'
    else:
        dir_name = 'tang'
    save_dir = DATA_DIR + dir_name + '/resnet34/'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # now do each layer separately
    # to limit memory usage
    for block_num, module in enumerate(blocks):
        print(f'starting on layer {block_num}')
        # set up hooks to extract activations
        act_list = []
        hook = module.register_forward_hook(
                lambda m,i,o: act_list.append(o.data.cpu().numpy()))

        # iterate and extract activations 
        with torch.no_grad():
            for batch in images:
                _ = net(batch[0].cuda()) # hook works automatically

        # compile and save activations
        acts = np.concatenate(act_list, axis=0)
        save_file = save_dir + f'block{block_num}'
        np.save(save_file, acts)

        # cleanup -- might not all be necessary
        hook.remove()
        del acts
        del act_list[:]
