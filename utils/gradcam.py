import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM():
    '''Implementation for Grad-CAM
    Steps:
        - Load a pre-trained model
        - Load an image that can be processed by this model (224x224 for VGG16 why?)
        - Infer the image and get the topmost class index
        - Take the output of the final convolutional layer
        - Compute the gradient of the class output value w.r.t to L feature maps
        - Pool the gradients over all the axes leaving out the channel dimension
        - Weigh the output feature map with the computed gradients (+ve)
        - Average the weighted feature maps along channels
        - Normalize the heat map to make the values between 0 and 1
    '''
    def __init__(self, model_layer):
        self.fmap = None
        self.grad = None

        #Register the forward and backward hooks
        handle_fw = model_layer.register_forward_hook(self.fwd_hook)
        handle_bw = model_layer.register_backward_hook(self.bkw_hook)
        self.handles = [handle_fw, handle_bw]
    
    def fwd_hook(self, module, input, output):
        print(module)
        self.fmap = output.detach()
        print('Inside ' + self.__class__.__name__ + ' forward')
        # print('')
        # print('input: ', type(input))
        # print('input[0]: ', type(input[0]))
        # print('output: ', type(output))
        # print('')
        # print('input size:', input[0].size())
        # print('output size:', output.data.size())
        # print('output norm:', output.data.norm())

    def bkw_hook(self, module, grad_input, grad_output):
        self.grad = grad_output[0].detach()
        print('Inside ' + self.__class__.__name__ + ' backward')
        # print('')
        # print('grad_input: ', type(grad_input))
        # print('grad_input[0]: ', type(grad_input[0]))
        # print('grad_output: ', type(grad_output))
        # print('grad_output[0]: ', type(grad_output[0]))
        # print('')
        # print('grad_input size:', grad_input[0].size())
        # print('grad_output size:', grad_output[0].size())
        # print('grad_input norm:', grad_input[0].norm())
        # print(self.grad.shape)

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()

    def generate(self, inputs, input_shape, index):

        grad = self.grad[index].unsqueeze(dim=0)
        fmap = self.fmap[index].unsqueeze(dim=0)

        input_image = inputs[index]
        # print(input_image.shape)
        input_image = input_image.numpy()
        input_image = np.transpose(input_image, (1,2,0))

        weights = F.adaptive_avg_pool2d(grad, 1)
        # print(weights.shape)

        gcam = torch.mul(fmap, weights).sum(dim=1, keepdim=True)
        # print(gcam.shape)

        gcam = F.relu(gcam)
        gcam = F.interpolate(
            gcam, input_shape, mode="bilinear", align_corners=False
        )

        B, C, H, W = gcam.shape
        gcam = gcam.view(B, -1)
        gcam -= gcam.min(dim=1, keepdim=True)[0]
        gcam /= gcam.max(dim=1, keepdim=True)[0]
        gcam = gcam.view(B, C, H, W)

        #get the heatmap
        gcam = gcam.numpy()      
        cmap = cm.jet(gcam)[..., :3] #* 255.0
        gcam_image = (cmap.astype(np.float) + input_image.astype(np.float)) / 2

        return gcam_image


