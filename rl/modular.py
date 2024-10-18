from typing import Any
import torch as tt
from torch import Tensor
import torch.nn as nn
from torch.nn import Module, ModuleList
from io import BytesIO


class Hyper:
    def __init__(self, **kwargs) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


def Identity(x):
    return x  # we can use nn.Identity


def numel(shape):
    r"""returns no of total elements (or addresses) in a multi-dim array
    Note: for torch tensor use Tensor.numel()"""
    return tt.prod(tt.tensor(shape)).item()


def arange(shape, start=0, step=1, dtype=None):
    r"""returns arange for multi-dimensional array (reshapes)"""
    return tt.arange(
        start=start, end=start + step * numel(shape), step=step, dtype=dtype
    ).reshape(shape)


def shares_memory(a: Tensor, b: Tensor) -> bool:
    r"""checks if two tensors share same underlying storage, in which case, changing values of one will change values in other as well
    Note: this is different from Tensor.is_set_to(Tensor) function which checks shape as well
    """
    return a.storage().data_ptr() == b.storage().data_ptr()


def absdiff(a: Tensor, b: Tensor):
    return tt.sum(tt.abs(a - b)).item()


def save_state(path, module: Module):
    tt.save(module.state_dict(), path)  # simply save the state dictionary


def load_state(path, module: Module):
    module.load_state_dict(tt.load(path))  # simply load the state dictionary


def save(module: Module, path: str):
    tt.save(module, path)


def load(path: str):
    return tt.load(path)


def count(module: Module, requires_grad=None):
    r"""Counts the total number of parameters (numel) in a params

    :param requires_grad:
        if None, counts all parameters
        if True, counts trainable parameters
        if False, counts non-trainiable (frozen) parameters
    """
    return sum(
        ([p.numel() for p in module.parameters()])
        if requires_grad is None
        else (
            [p.numel() for p in module.parameters() if p.requires_grad is requires_grad]
        )
    )


def show(module: Module, values: bool = False):
    r"""Prints the parameters of a params

    :param values: if True, prints the full tensors otherwise prints only shape
    """
    nos_trainable, nos_frozen = 0, 0
    print("=====================================")
    for i, p in enumerate(module.parameters()):
        iparam = p.numel()
        if p.requires_grad:
            nos_trainable += iparam
        else:
            nos_frozen += iparam
        print(
            f"#[{i}]\tShape[{p.shape}]\tParams: {iparam}\tTrainable: {p.requires_grad}"
        )
        if values:
            print("=====================================")
            print(f"{p}")
            print("=====================================")
    print(
        f"\nTotal Parameters: {nos_trainable+nos_frozen}\tTrainable: {nos_trainable}\tFrozen: {nos_frozen}"
    )
    print("=====================================")
    return


@tt.no_grad()
def state(module: Module, values=False):
    r"""prints the parameters using `nn.Module.parameters` iterator, use `values=True` to print full parameter tensor"""
    sd = module.state_dict()
    for i, (k, v) in enumerate(sd.items()):
        print(f"#[{i+1}]\t[{k}]\tShape[{v.shape}]")
        if values:
            print(f"{v}")
    return


@tt.no_grad()
def diff(module1: Module, module2: Module, do_abs: bool = True, do_sum: bool = True):
    r"""Checks the difference between the parameters of two modules.
        This can be used to check if two models have exactly the same parameters.

    :param do_abs: if True, finds the absolute difference
    :param do_sum: if True, finds the sum of difference

    :returns: a list of differences in each parameter or their sum if ``do_sum`` is True.
    """
    d = [
        (abs(p1 - p2) if do_abs else (p1 - p2))
        for p1, p2 in zip(module1.parameters(), module2.parameters())
    ]
    if do_sum:
        d = [tt.sum(p) for p in d]
    return d


@tt.no_grad()
def copy(module_from: Module, module_to: Module) -> None:
    r"""Copies the parameters of a params to another - both modules are supposed to be identical"""
    for pt, pf in zip(module_to.parameters(), module_from.parameters()):
        pt.copy_(pf)


def clones(module: Module, n_copies: int):
    r"""Replicates a params by storing it in a buffer and retriving many copies
    NOTE: this will preserve the ```require_grad``` attribute on all tensors."""
    # from io import BytesIO
    if n_copies < 1:
        return None
    buffer = BytesIO()
    tt.save(module, buffer)
    model_copies = []
    for _ in range(n_copies):
        buffer.seek(0)
        model_copy = tt.load(buffer)
        model_copies.append(model_copy)
    buffer.close()
    del buffer
    return model_copies


def clone(module: Module):
    return clones(module, 1).pop()


def duplicate(module: Module, n_copies):
    return ModuleList(clones(module, n_copies))


def requires_grad_(module: Module, requires: bool, *names):
    r"""Sets requires_grad attribute on tensors in params
    if no names are provided, sets requires_grad on all tensors
    NOTE: careful with *names, if a buffer's name is provided
        and it is in the state_dict then its grad will be enabled
        which is undesirable.
        not providing any names will target the parameters only
    """
    if names:  # if we know which params to freeze, we can provide them
        state_dict = module.state_dict()
        for n in names:
            state_dict[n].requires_grad_(requires)
    else:  # if we want to do on all params
        for p in module.parameters():
            p.requires_grad_(requires)
    return module


@tt.no_grad()
def zero_(module: Module, *names):
    r"""Sets requires_grad attribute on tensors in params
    if no names are provided, sets requires_grad on all tensors

    NOTE: careful with *names, if a buffer's name is provided
        and it is in the state_dict then it will be zeroed too
        which is actually desirable in some cases.
        pass a single blank string to zero everything in state dict
        not providing any names will target the parameters only
    """
    if names:
        state_dict = module.state_dict()
        if " " in names:
            for p in state_dict.values():
                p.zero_()
        else:
            for n in names:
                state_dict[n].zero_()
    else:
        for p in module.parameters():
            p.zero_()
    return module


def zero_like(module: Module) -> dict:
    return zero_(clone(module), " ")


def dense(
    in_dim,
    layer_dims,
    out_dim,
    actFs,
    use_bias=True,
    use_biasL=True,
    dtype=None,
    device=None,
):
    r"""
    Creats a stack of fully connected (dense) layers which is usually connected at end of other networks
    Args:
        in_dim          `integer`       : in_features or input_size
        layer_dims      `List/Tuple`    : size of hidden layers
        out_dim         `integer`       : out_features or output_size
        actF            `nn.Module`     : activation function at hidden layer
        actFA           `dict`          : args while initializing actF
        actL            `nn.Module`     : activation function at last layer
        actLA           `dict`          : args while initializing actL
        use_bias        `bool`          : if True, uses bias at hidden layers
        use_biasL       `bool`          : if True, uses bias at last layer

    Returns:
        `nn.Module` : an instance of nn.Sequential
    """
    layers = []
    # first layer
    layers.append(
        nn.Linear(in_dim, layer_dims[0], bias=use_bias, dtype=dtype, device=device)
    )
    if actFs:
        layers.append(actFs.pop(0))
    # remaining layers
    for i in range(len(layer_dims) - 1):
        layers.append(
            nn.Linear(
                layer_dims[i],
                layer_dims[i + 1],
                bias=use_bias,
                dtype=dtype,
                device=device,
            )
        )
        if actFs:
            layers.append(actFs.pop(0))
    # last layer
    layers.append(
        nn.Linear(layer_dims[-1], out_dim, bias=use_biasL, dtype=dtype, device=device)
    )
    if actFs:
        layers.append(actFs.pop(0))
    return nn.Sequential(*layers)


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=


class Remap:
    r"""
    Provides a mapping between ranges, works with scalars, ndarrays and tensors.

    :param Input_Range:     *FROM* range for ``i2o`` call, *TO* range for ``o2i`` call
    :param Output_Range:    *TO* range for ``i2o`` call, *FROM* range for ``o2i`` call

    .. note::
        * :func:`~known.basic.REMAP.i2o`: maps an input within `Input_Range` to output within `Output_Range`
        * :func:`~known.basic.REMAP.o2i`: maps an input within `Output_Range` to output within `Input_Range`

    Examples::

        >>> mapper = REMAP(Input_Range=(-1, 1), Output_Range=(0,10))
        >>> x = np.linspace(mapper.input_low, mapper.input_high, num=5)
        >>> y = np.linspace(mapper.output_low, mapper.output_high, num=5)

        >>> yt = mapper.i2o(x)  #<--- should be y
        >>> xt = mapper.o2i(y) #<----- should be x
        >>> xE = np.sum(np.abs(yt - y)) #<----- should be 0
        >>> yE = np.sum(np.abs(xt - x)) #<----- should be 0
        >>> print(f'{xE}, {yE}')
        0, 0
    """

    def __init__(self, Input_Range: tuple, Output_Range: tuple) -> None:
        r"""
        :param Input_Range:     `from` range for ``i2o`` call, `to` range for ``o2i`` call
        :param Output_Range:    `to` range for ``i2o`` call, `from` range for ``o2i`` call
        """
        self.set_input_range(Input_Range)
        self.set_output_range(Output_Range)

    def set_input_range(self, Range: tuple) -> None:
        r"""set the input range"""
        self.input_low, self.input_high = Range
        self.input_delta = self.input_high - self.input_low

    def set_output_range(self, Range: tuple) -> None:
        r"""set the output range"""
        self.output_low, self.output_high = Range
        self.output_delta = self.output_high - self.output_low

    def backward(self, X):
        r"""maps ``X`` from ``Output_Range`` to ``Input_Range``"""
        return (
            (X - self.output_low) * self.input_delta / self.output_delta
        ) + self.input_low

    def forward(self, X):
        r"""maps ``X`` from ``Input_Range`` to ``Output_Range``"""
        return (
            (X - self.input_low) * self.output_delta / self.input_delta
        ) + self.output_low

    def __call__(self, X, backward=False):
        return self.backward(X) if backward else self.forward(X)

    def swap_range(self):
        Input_Range, Output_Range = (self.output_low, self.output_high), (
            self.input_low,
            self.input_high,
        )
        self.set_input_range(Input_Range)
        self.set_output_range(Output_Range)


# =-=-=-=-
