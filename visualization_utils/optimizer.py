import torch
import numpy as np
from .utils import *
import cv2

class Optimizer():
    """
    Optimize an image to produce some result in a deep net.
    """

    def __init__(self, net, layer, loss_func, first_layer=None):
        """
        Parameters:

        net: nn.Module, presumably a deep net
        layer: nn.Module, part of the network that gives relevant output
        first_layer: nn.Module: the input layer of the network; will try
        to determine automatically if not specified
        loss_func: callable taking layer output, target output, and image,
        returning the loss
        first_layer: nn.Module: the input layer of the network; will try
        to determine automatically if not specified
        """
        super().__init__()

        self.net = net
        self.layer = layer
        self.loss_func = loss_func
        self.first_layer = first_layer

        # will only define hooks during optimization so they can be removed
        self.acts = []
        self.grads = []

    def optimize(self, image, target, constant_area=None, max_iter=1000,
                 lr=np.linspace(5, 0.5, 1000), clip_image=False,
                 grayscale=False, sigma=0, debug=False, early_stopper=None):
        """
        Parameters:

        image: image to start from, presumably where the target was 
        modified from

        target: target activation, to be passed into loss_func

        constant_area: indices such that image[0:1, 2:3, :] stays
        constant each iteration of gradient ascent
        
        max_iter: maximum number of iterations to run

        lr: 'learning rate' (multiplier of gradient added to image at
        each step, or iterable of same length as max_iter with varying values)

        clip_image: whether or not to clip the image to real (0-256) pixel
        values, with standard torchvision transformations

        sigma: sigma of the Gaussian smoothing at each iteration
        (default value 0 means no smoothing), can be an iterable of
        length max_iter like 'lr'

        debug: whether or not to print loss each iteration

        early_stopper: function that takes the list of losses so far,
        returns True if the optimization process should stop

        Returns:

        optimized image
        loss for the last iteration
        """
        image.requires_grad_(False)
        new_img = torch.tensor(image, requires_grad=True)
        img_length = new_img.shape[2]
        # change it to an array even if it's constant, for the iterating code
        if isinstance(lr, int) or isinstance(lr, float):
            lr = [lr] * max_iter

        if isinstance(sigma, float) or isinstance(sigma, int):
            sigma = [sigma] * max_iter

        # want the actual, atomic first layer object for hooking
        if self.first_layer is None:
            children = [child for child in self.net.modules()
                        if len(list(child.children())) == 0]
            first_layer = children[0]
            print(first_layer)
        else:
            first_layer = self.first_layer

        # set up hooks

        forw_hook = self.layer.register_forward_hook(
            lambda m, i, o: self.acts.append(o))

        Ind = range(10)

        # RFsize = np.zeros(len(Ind)) + 15
        # ic = np.zeros(len(Ind)) + 24.5
        # jc = np.zeros(len(Ind)) + 24.5
        #
        # #following part only for tang data visualization
        # temp = np.zeros((img_length, img_length)).astype(np.float32)
        # for i in range(img_length):
        #     for j in range(img_length):
        #         if (i - ic[0]) ** 2 + (j - jc[0]) ** 2 < (RFsize[0] + 1) ** 2:
        #             temp[i, j] = 1
        # mask = np.zeros((1, img_length, img_length, 1), dtype=np.float32)
        # mask[0, :, :, 0] = cv2.GaussianBlur(1 - temp, (21, 21), 4.4)
        # back1 = np.zeros((1, img_length, img_length, 1), dtype=np.float32) + 0.1875
        # torch_back1 = torch.tensor(back1).cuda()
        # #loss_l2 = torch.nn.MSELoss()

        best_loss = 500
        best_img = torch.clone(new_img)
        # now do gradient ascent
        losses = []
        counter = 0
        for i in range(max_iter):
            # get gradient
            _ = self.net(new_img)
            loss = self.loss_func(self.acts[0], target, new_img)
            #add l2 mask here

            # mask1 = np.repeat(mask, 1, 0)
            #
            # torch_mask1 = torch.tensor(mask1).cuda()
            #
            # lap_diff1 = torch.linalg.norm(new_img[:, :, :, 1:] - new_img[:, :, :, :img_length - 1]) \
            #             + torch.linalg.norm(new_img[:, :, 1:, :] - new_img[:, :, :img_length - 1, :])
            #
            # l2 = torch.linalg.norm((new_img - torch_back1) * torch_mask1)


            if(loss < best_loss):
                best_img = torch.clone(new_img)
                best_loss = loss
            losses.append(loss)
            self.net.zero_grad()
            loss.backward()

            if debug:
                print(f'loss for iter {i}: {loss}')

            # all processing of gradient was done in loss_func
            # even momentum if applicable; none is done here
            with torch.no_grad():
                #new_img.data = new_img.data - lr[i] * self.grads[0].data
                new_img.data = new_img.data - lr[i] * new_img.grad
                if clip_image:
                    new_img.data = clip_img(new_img.data)

                if sigma[i] > 0:
                    new_img.data = torch_gaussian_filter(new_img, sigma[i])

                if constant_area is not None:
                    # assuming torchvision structure (BCHW) here
                    # TODO: change this
                    new_img.data[:, :, constant_area[0]:constant_area[1],
                    constant_area[2]:constant_area[3]] = image[:, :,
                                                         constant_area[0]:constant_area[1],
                                                         constant_area[2]:constant_area[3]]

                if grayscale:
                    # keep the image grayscale
                    gray_vals = new_img.data.mean(1)
                    new_img.data = torch.stack([gray_vals] * 3, 1)

                #keep in bounds of (0,1) just for Tang data
                new_img.data -= new_img.data.min()
                new_img.data /= (new_img.data.max() - new_img.data.min())

            self.acts.clear()
            self.grads.clear()

            if early_stopper is not None and early_stopper(losses):
                print(f'early stopping at iter {i}')
                break

        # avoid side effects
        forw_hook.remove()

        return new_img, losses, best_loss
