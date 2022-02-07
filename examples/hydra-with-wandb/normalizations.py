"""
Normalization Hook Class for W&B Normalization Series.
Author: @captain-pool <adrish@wandb.com>
"""
import collections.abc
import functools
import inspect
import math

import torch


class NormalizationHook:
    def __init__(self, init_dims, **kwargs):
        """
        Object factory of NormalizationHook class.
        Args:
          init_dims: (H, W) of each data sample.
          kwargs: param
        """
        self.seq_dims = [init_dims]
        self._kwargs = kwargs

    def normalization(self, idim):
        raise NotImplementedError(
            "This is an interface class."
            "Inherit this class and define your own normalizations."
        )

    def filter(self, function):
        """
        Removes spurious parameters from custom parameters and
        creates a partial function with the filtered parameter set.
        Args:
          function: Function to filter
        """
        params = self._kwargs
        original_params = set(inspect.signature(function).parameters)
        param_keys = set(params)
        to_be_removed = param_keys - original_params
        for param_name in to_be_removed:
            params.pop(param_name)
        return functools.partial(function, **params)

    @staticmethod
    def calc_conv2d_shape(
        idim, ksize, dilation=(1, 1), stride=(1, 1), padding=(0, 0), ceil_mode=False
    ):
        """
        Calculate output shape of Conv2d.
        (ref: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
        Args:
          idim: input dimensions of the convolution (i.e., HW)
          ksize: size of the kernel window
          dilation: size of dilation window
          stride: stride length along (H, W)
          padding: padding size along (H, W)
        returns:
          returns shape of convoluted outputs (H_{new}, W_{new})
        """

        def shape_each_dim(i):
            odim_i = idim[i] + 2 * padding[i] - dilation[i] * (ksize[i] - 1) - 1
            return (odim_i / stride[i]) + 1

        round_fn = math.ceil if ceil_mode else math.floor
        return round_fn(shape_each_dim(0)), round_fn(shape_each_dim(1))

    def register_dim(self, dim):
        """
        Register output dimension manually for layers other than Conv2d
        """
        self.seq_dims.append(dim)
        return torch.Identity()

    def conv_hook(self, layer_obj, norm=False):
        """
        Register Convolution Layer or MaxPool Layer for calculating output dimensions
        without calling forward. Currently only supports `Conv2d(...)` or `MaxPool2d(...)`.
        Args:
          layer_obj: Layer Object of Convolution Layer or MaxPool Layer
          norm: adds a normalization layer after convolution defined in self.normalization()
        """

        def tuplify(obj):
            if not isinstance(obj, collections.abc.Sequence):
                return (obj, obj)
            return obj

        supported_layers = (torch.nn.Conv2d, torch.nn.MaxPool2d)
        message = ", ".join(["{.__name__}"] * len(supported_layers))
        message = "only supports: " + message
        assert isinstance(layer_obj, supported_layers), message.format(
            *supported_layers
        )
        self.seq_dims.append(
            NormalizationHook.calc_conv2d_shape(
                self.seq_dims[-1],
                tuplify(layer_obj.kernel_size),
                tuplify(layer_obj.dilation),
                tuplify(layer_obj.stride),
                tuplify(layer_obj.padding),
                getattr(layer_obj, "ceil_mode", False),
            )
        )
        modules = [layer_obj]
        if norm:
            modules.append(
                self.normalization([layer_obj.out_channels, *self.seq_dims[-1]]),
            )
        return torch.nn.Sequential(*modules)


class BatchNormedModel(NormalizationHook):
    def __init__(self, idim, **kwargs):
        super(BatchNormedModel, self).__init__(idim, **kwargs)
        self.filtered_fn = self.filter(torch.nn.BatchNorm2d)

    def normalization(self, idim):
        return self.filtered_fn(num_features=idim[0])


class LayerNormedModel(NormalizationHook):
    def __init__(self, idim, **kwargs):
        super(LayerNormedModel, self).__init__(idim, **kwargs)
        self.filtered_fn = self.filter(torch.nn.LayerNorm)

    def normalization(self, idim):
        return self.filtered_fn(normalized_shape=idim)


class InstanceNormedModel(NormalizationHook):
    def __init__(self, idim, **kwargs):
        super(InstanceNormedModel, self).__init__(idim, **kwargs)
        self.filtered_fn = self.filter(torch.nn.InstanceNorm2d)

    def normalization(self, idim):
        return self.filtered_fn(num_features=idim[0])


class GroupNormedModel(NormalizationHook):
    def __init__(self, idim, **kwargs):
        super(GroupNormedModel, self).__init__(idim, **kwargs)
        assert "num_groups" in kwargs, "num_groups variable is not present"
        self.filtered_fn = self.filter(torch.nn.GroupNorm)

    def normalization(self, idim):
        return self.filtered_fn(num_channels=idim[0])


class NoNormedModel(NormalizationHook):
    def __init__(self, idim, **kwargs):
        super(NoNormedModel, self).__init__(idim, **kwargs)

    def normalization(self, idim):
        return torch.Identity()


NORMALIZATIONS = {
    "batch": BatchNormedModel,
    "layer": LayerNormedModel,
    "group": GroupNormedModel,
    "instance": InstanceNormedModel,
    "nonorm": NoNormedModel,
}
