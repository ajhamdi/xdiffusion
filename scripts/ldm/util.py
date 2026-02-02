import importlib
import torchvision
import torch
from torch import optim
import numpy as np
from inspect import isfunction
from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
import cv2
from carvekit.api.high import HiInterface
from typing import List, Optional
from packaging import version

if version.parse(torch.__version__) >= version.parse("1.7.0"):
    import torch.fft  # type: ignore


def check_backward_validity(inputs):
    if not any(inp.requires_grad for inp in inputs if isinstance(inp, torch.Tensor)):
        import warnings
        warnings.warn("None of the inputs have requires_grad=True. Gradients will be None")


def get_device_states(*args):
    unique_devices = set(
        arg.get_device() for arg in args
        if isinstance(arg, torch.Tensor) and arg.is_cuda
    )
    return list(unique_devices), [torch.cuda.get_rng_state(d) for d in unique_devices]


def set_device_states(devices, states):
    for device, state in zip(devices, states):
        torch.cuda.set_rng_state(state, device)


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, preserve_rng_state, *args):
        check_backward_validity(args)
        ctx.run_function = run_function
        ctx.preserve_rng_state = preserve_rng_state

        ctx.had_autocast_in_fwd = torch.is_autocast_enabled()
       

        if preserve_rng_state:
            ctx.fwd_cpu_state = torch.random.get_rng_state()
            ctx.had_cuda_in_fwd = False
            if torch.cuda._initialized:
                ctx.had_cuda_in_fwd = True
                ctx.fwd_gpu_devices, ctx.fwd_gpu_states = get_device_states(*args)
        ctx.save_for_backward(*args)
        with torch.no_grad():
            outputs = run_function(*args)
        return outputs

    @staticmethod
    def backward(ctx, *args):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "Checkpointing is not compatible with .grad(), use .backward() if possible"
            )
        inputs = ctx.saved_tensors

        rng_devices = []
        if ctx.preserve_rng_state and ctx.had_cuda_in_fwd:
            rng_devices = ctx.fwd_gpu_devices

        with torch.random.fork_rng(devices=rng_devices, enabled=ctx.preserve_rng_state):
            if ctx.preserve_rng_state:
                torch.random.set_rng_state(ctx.fwd_cpu_state)
                if ctx.had_cuda_in_fwd:
                    set_device_states(ctx.fwd_gpu_devices, ctx.fwd_gpu_states)

            detached_inputs = tuple(
                x.detach().requires_grad_(x.requires_grad) for x in inputs
            )

            # ── NEW ──────────────────────────────────────────────────────
            # Restore the autocast state that was active during forward so
            # the recomputed activations have the same dtypes as the
            # originals.  Without this, FP16 mixed-precision training
            # crashes inside any op that sees mismatched input/weight dtypes
            # (most visibly: LayerNorm via F.layer_norm).
            with torch.cuda.amp.autocast(enabled=ctx.had_autocast_in_fwd):
                with torch.enable_grad():
                    outputs = ctx.run_function(*detached_inputs)
            # ─────────────────────────────────────────────────────────────

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)

        outputs_with_grad = []
        args_with_grad = []
        for i in range(len(outputs)):
            if torch.is_tensor(outputs[i]) and outputs[i].requires_grad:
                outputs_with_grad.append(outputs[i])
                args_with_grad.append(args[i])
        if len(outputs_with_grad) == 0:
            raise RuntimeError(
                "None of the outputs have requires_grad=True. Gradients will be None"
            )
        torch.autograd.backward(outputs_with_grad, args_with_grad)
        grads = tuple(
            inp.grad if isinstance(inp, torch.Tensor) else inp
            for inp in detached_inputs
        )
        return (None, None) + grads


def checkpoint(run_function, *args, **kwargs):
    preserve = kwargs.pop('preserve_rng_state', True)
    if kwargs:
        raise ValueError("Unexpected keyword arguments: " + str(kwargs))
    return CheckpointFunction.apply(run_function, preserve, *args)



def default(val, d):
    return val if val is not None else (d() if callable(d) else d)

def exists(x):
    return x is not None


def noop(*args, **kwargs):
    pass


# --- FFT helpers (version-aware) ---------------------------------------------

def fft2c_old(data):
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")
    data = ifftshift(data, dim=[-3, -2])
    data = torch.fft(data, 2, normalized=True)
    data = fftshift(data, dim=[-3, -2])
    return data


def ifft2c_old(data):
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")
    data = ifftshift(data, dim=[-3, -2])
    data = torch.ifft(data, 2, normalized=True)
    data = fftshift(data, dim=[-3, -2])
    return data


def fft2c_new(data):
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")
    data = ifftshift(data, dim=[-3, -2])
    data = torch.view_as_real(
        torch.fft.fftn(torch.view_as_complex(data), dim=(-2, -1), norm="ortho")
    )
    data = fftshift(data, dim=[-3, -2])
    return data


def ifft2c_new(data):
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")
    data = ifftshift(data, dim=[-3, -2])
    data = torch.view_as_real(
        torch.fft.ifftn(torch.view_as_complex(data), dim=(-2, -1), norm="ortho")
    )
    data = fftshift(data, dim=[-3, -2])
    return data


if version.parse(torch.__version__) >= version.parse("1.7.0"):
    fft2c = fft2c_new
    ifft2c = ifft2c_new
else:
    fft2c = fft2c_old
    ifft2c = ifft2c_old


# --- roll / shift ------------------------------------------------------------

def roll_one_dim(x, shift, dim):
    shift = shift % x.size(dim)
    if shift == 0:
        return x
    left  = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)
    return torch.cat((right, left), dim=dim)


def roll(x, shift, dim):
    if len(shift) != len(dim):
        raise ValueError("len(shift) must match len(dim)")
    for (s, d) in zip(shift, dim):
        x = roll_one_dim(x, s, d)
    return x


def fftshift(x, dim=None):
    if dim is None:
        dim = list(range(x.dim()))
    shift = [x.shape[d] // 2 for d in dim]
    return roll(x, shift, dim)


def ifftshift(x, dim=None):
    if dim is None:
        dim = list(range(x.dim()))
    shift = [(x.shape[d] + 1) // 2 for d in dim]
    return roll(x, shift, dim)


# --- logging / image helpers -------------------------------------------------

def log_txt_as_img(wh, xc, size=10):
    from PIL import Image, ImageDraw, ImageFont
    b = len(xc)
    txts = list()
    for bi in range(b):
        txt = Image.new("RGB", wh, color="white")
        draw = ImageDraw.Draw(txt)
        nc = int(40 * (wh[0] / 256))
        lines = "\n".join(
            xc[bi][start:start + nc] for start in range(0, len(xc[bi]), nc)
        )
        try:
            draw.text((0, 0), lines, fill="black")
        except UnicodeEncodeError:
            print("Can't encode string for logging. Skipping.")
        txt = np.array(txt).transpose(2, 0, 1) / 127.5 - 1.0
        txts.append(txt)
    txts = np.stack(txts)
    txts = torch.tensor(txts)
    return txts


def ismap(x):
    if not isinstance(x, torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] > 3)


def isimage(x):
    if not isinstance(x, torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] == 3 or x.shape[1] == 1)


def mean_flat(tensor):
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params


# --- instantiation helpers ---------------------------------------------------

def instantiate_from_config(config):
    if "target" not in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)



class AdamWwithEMAandWings(optim.Optimizer):
    # credit to https://gist.github.com/crowsonkb/65f7265353f403714fce3b2595e0b298
    def __init__(self, params, lr=1.e-3, betas=(0.9, 0.999), eps=1.e-8,  # TODO: check hyperparameters before using
                 weight_decay=1.e-2, amsgrad=False, ema_decay=0.9999,   # ema decay to match previous code
                 ema_power=1., param_names=()):
        """AdamW that saves EMA versions of the parameters."""
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= ema_decay <= 1.0:
            raise ValueError("Invalid ema_decay value: {}".format(ema_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, ema_decay=ema_decay,
                        ema_power=ema_power, param_names=param_names)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            ema_params_with_grad = []
            state_sums = []
            max_exp_avg_sqs = []
            state_steps = []
            amsgrad = group['amsgrad']
            beta1, beta2 = group['betas']
            ema_decay = group['ema_decay']
            ema_power = group['ema_power']

            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                grads.append(p.grad)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of parameter values
                    state['param_exp_avg'] = p.detach().float().clone()

                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])
                ema_params_with_grad.append(state['param_exp_avg'])

                if amsgrad:
                    max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                # update the steps for each param group update
                state['step'] += 1
                # record the step after step update
                state_steps.append(state['step'])

            optim._functional.adamw(params_with_grad,
                    grads,
                    exp_avgs,
                    exp_avg_sqs,
                    max_exp_avg_sqs,
                    state_steps,
                    amsgrad=amsgrad,
                    beta1=beta1,
                    beta2=beta2,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    eps=group['eps'],
                    maximize=False)

            cur_ema_decay = min(ema_decay, 1 - state['step'] ** -ema_power)
            for param, ema_param in zip(params_with_grad, ema_params_with_grad):
                ema_param.mul_(cur_ema_decay).add_(param.float(), alpha=1 - cur_ema_decay)

        return loss