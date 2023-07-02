import torch
import numpy as np
from .utils import *
from torchvision.transforms import CenterCrop,Resize
from skimage.draw import disk
from tqdm import tqdm
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
<<<<<<< Updated upstream
=======


class GANOptimzier():
    """
    Optimize an image to produce some result in a deep net.
    """

    def __init__(self, gan, net, loss_func, first_layer=None):
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

        self.gan = gan
        self.net = net
        self.loss_func = loss_func
        self.first_layer = first_layer
        self.rr, self.cc = disk((25, 25), 25, shape=(50,50))
        # will only define hooks during optimization so they can be removed
        self.grads = []

    def partial_grating(self,img,gray=0.5,
                        background=None):
        """
        put an aperture over a sine-wave grating image
        """
        # outside aperture will be gray
        # unless it's something like a diff orientation
        if background is None:
            new_img = torch.full(img.shape, fill_value=gray).to('cuda')
        else:
            new_img = background
        new_img[self.rr, self.cc] = img[self.rr, self.cc]
        return new_img

    def image_transform(self,img):
        x = img[0]
        x = (x + 1) / 2
        x = x[0] * 299 / 1000 + x[1] * 587 / 1000 + x[2] * 114 / 1000
        x = CenterCrop(128)(x)
        x = torch.reshape(x, (1, 128, 128))
        x = Resize(50, antialias=True)(x)[0]
        x = self.partial_grating(x, gray=0.0)
        return torch.reshape(x, (1,1,x.shape[0],x.shape[1]))
    def optimize(self, input,target,max_iter=1000,
                 lr=np.linspace(5, 0.5, 1000),
                 debug=False, early_stopper=None):
        """
        Parameters:

        image: image to start from, presumably where the target was
        modified from

        target: target activation, to be passed into loss_func

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
        input.requires_grad_(False)
        new_input = torch.tensor(input, requires_grad=True)
        print(new_input.shape)
        # change it to an array even if it's constant, for the iterating code
        if isinstance(lr, int) or isinstance(lr, float):
            lr = [lr] * max_iter

        # set up hooks
        best_loss = 500
        # now do gradient ascent
        losses = []
        for i in tqdm(range(max_iter)):
            # get gradient
            image = self.gan(z=new_input)
            new_img = self.image_transform(image)
            out = self.net(new_img)
            loss = self.loss_func(out, target, new_img)

            if (loss < best_loss):
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
                new_input.data = new_input.data - lr[i] * new_input.grad
                torch.clamp(new_input.data, -1, 1)
            self.grads.clear()

            if early_stopper is not None and early_stopper(losses):
                print(f'early stopping at iter {i}')
                break

        return new_input, losses, best_loss
>>>>>>> Stashed changes
