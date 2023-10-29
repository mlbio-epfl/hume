# Author: Mathieu Blondel
# License: Simplified BSD

"""
PyTorch implementation of

Learning Classifiers with Fenchel-Young Losses:
    Generalized Entropies, Margins, and Algorithms.
Mathieu Blondel, André F. T. Martins, Vlad Niculae.
https://arxiv.org/abs/1805.09717
"""


import torch

# begin: From OpenNMT-py
def threshold_and_support(z, dim=0):
    """
    z: any dimension
    dim: dimension along which to apply the sparsemax
    """
    sorted_z, _ = torch.sort(z, descending=True, dim=dim)
    z_sum = sorted_z.cumsum(dim) - 1  # sort of a misnomer
    k = torch.arange(1, sorted_z.size(dim) + 1, device=z.device).type(z.dtype).view(
        torch.Size([-1] + [1] * (z.dim() - 1))
    ).transpose(0, dim)
    support = k * sorted_z > z_sum

    k_z_indices = support.sum(dim=dim).unsqueeze(dim)
    k_z = k_z_indices.type(z.dtype)
    tau_z = z_sum.gather(dim, k_z_indices - 1) / k_z
    return tau_z, k_z


class SparsemaxFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, dim=0):
        """
        input (FloatTensor): any shape
        returns (FloatTensor): same shape with sparsemax computed on given dim
        """
        ctx.dim = dim
        tau_z, k_z = threshold_and_support(input, dim=dim)
        output = torch.clamp(input - tau_z, min=0)
        ctx.save_for_backward(k_z, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        k_z, output = ctx.saved_tensors
        dim = ctx.dim
        grad_input = grad_output.clone()
        grad_input[output == 0] = 0

        v_hat = (grad_input.sum(dim=dim) / k_z.squeeze()).unsqueeze(dim)
        grad_input = torch.where(output != 0, grad_input - v_hat, grad_input)
        return grad_input, None


sparsemax = SparsemaxFunction.apply


class Sparsemax(torch.nn.Module):

    def __init__(self, dim=0):
        self.dim = dim
        super(Sparsemax, self).__init__()

    def forward(self, input):
        return sparsemax(input, self.dim)
# end: From OpenNMT-py
