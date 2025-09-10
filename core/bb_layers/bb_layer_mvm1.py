"""bb_layer_mvm1.

MVM1 (free-space diffraction-inspired) BB layer.

This file adapts utilities from an external repo to Astralora's BB API:
    create_bb_layer_mvm1(d_inp, d_out) -> (bb_func, w0, dw)
"""

import math
import numpy as np
import torch


def reshape_and_padding(x, grid_size):
    """Transformation of tensor x to 2d matrixs.
    - x is a [batch_size, d_inp] tensor of float; its values are in (0, 1).
        or x is a [d_inp], if x single vector.
    - grid_size is a grid_size of our grid. It is int.
    Function returns the [batch_size, d_out] tensor of float.
        or [d_inp] tensor.
    """
    side = int(np.ceil(np.sqrt(x.size()[-1])))
    assert side <= grid_size, f"Invalid inequality of `side` and `grid_side`:  {side} (`side`) > {grid_size} (`grid_size`)"

    out = x.reshape(x.shape[:-1] + (side, side))
    blank = torch.zeros(x.shape[:-1] + (grid_size, grid_size), device=x.device, dtype=torch.complex64)
    blank[
        ...,
        (grid_size - side) // 2 : (grid_size - side) // 2 + side,
        (grid_size - side) // 2 : (grid_size - side) // 2 + side,
    ] = out
    return blank


def output_prepare(y, y_pad):
    side = y.shape[-1]
    assert y_pad <= side, f"Invalid inequality of `side` and `y_pad`:  {side} (`side`) < {y_pad} (`y_pad`)"
    y = torch.nn.functional.avg_pool2d(y, kernel_size=side - y_pad + 1, stride=1)
    return y.flatten(start_dim=-2)


def padding(x, n_pad):
    n = n_pad - x.shape[-1]
    assert n >= 0, f"Invalid padding size: {n}"
    x_padded = torch.concat([x, torch.zeros(x.shape[:-1] + (n,), device=x.device)], dim=-1)
    return x_padded


def encode_weights(weights, encoding):
    phase_weights, amplitude_weights = None, None
    if encoding == "phase":
        phase_weights = torch.exp(2.0j * torch.pi * weights)
    elif encoding == "amplitude":
        amplitude_weights = weights
    elif encoding == "phase_amplitude":
        n = weights.shape[0]
        phase_weights, amplitude_weights = torch.exp(2.0j * torch.pi * weights[: n // 2]), weights[n // 2 :]
    else:
        raise NotImplementedError(f"Unknown encoding method: {encoding}. Use `phase`, `amplitude` or `phase_amplitude`.")
    return phase_weights, amplitude_weights


def apply_weights(x, phase_weights, amplitude_weights, grid_size, w_idx, layer_len):
    if phase_weights is not None:
        x = x * reshape_and_padding(phase_weights[w_idx : w_idx + layer_len], grid_size)
    if amplitude_weights is not None:
        x = x * reshape_and_padding(amplitude_weights[w_idx : w_idx + layer_len], grid_size)
    return x


def free_space_propagator(x, propagator):
    return torch.fft.ifft2(torch.fft.fft2(x) * propagator)


def lens_diffusor_propagator(x, diffusor):
    x = torch.fft.fft2(x)
    x = x * diffusor
    x = torch.fft.ifft2(x)
    return x


def calculate_intensity_single_beam(x, w, diffusors, params_distr, grid_size, weight_encoding, propagators):
    phase_weights, amplitude_weights = encode_weights(w, weight_encoding)

    if propagators is None:
        w_idx = 0
        x = reshape_and_padding(x, grid_size)  # coding input data in laser beam
        for layer_num, layer_len in enumerate(params_distr):  # sublayers
            x = lens_diffusor_propagator(x, diffusors[layer_num])
            x = apply_weights(x, phase_weights, amplitude_weights, grid_size, w_idx, layer_len)
            w_idx += layer_len
        x = lens_diffusor_propagator(x, diffusors[layer_num + 1])
    else:
        w_idx = 0
        x = reshape_and_padding(x, grid_size)
        x = x * diffusors[0]
        x = free_space_propagator(x, propagators[0])
        for layer_len in params_distr:  # sublayers
            x = apply_weights(x, phase_weights, amplitude_weights, grid_size, w_idx, layer_len)
            x = free_space_propagator(x, propagators[1])
            w_idx += layer_len
    return torch.abs(x)


def calculate_intensity_double_beam(x, w, diffusors, params_distr, grid_size, weight_encoding, propagators):
    phase_weights, amplitude_weights = encode_weights(w, weight_encoding)

    if propagators is None:
        w_idx = 0
        layer_num = 0
        x = reshape_and_padding(x, grid_size)
        x1 = 0.5 * x.clone()
        x2 = 0.5 * x.clone()
        for layer_len in params_distr[: len(params_distr) // 2]:  # sublayers x1
            x1 = lens_diffusor_propagator(x1, diffusors[layer_num])
            x1 = apply_weights(x1, phase_weights, amplitude_weights, grid_size, w_idx, layer_len)
            w_idx += layer_len
            layer_num += 1
        for layer_len in params_distr[len(params_distr) // 2 :]:  # sublayers x2
            x2 = lens_diffusor_propagator(x2, diffusors[layer_num])
            x2 = apply_weights(x2, phase_weights, amplitude_weights, grid_size, w_idx, layer_len)
            w_idx += layer_len
            layer_num += 1
        x1 = lens_diffusor_propagator(x1, diffusors[layer_num])
        x2 = lens_diffusor_propagator(x2, diffusors[layer_num + 1])
    else:
        x = reshape_and_padding(x, grid_size)
        w_idx = 0
        x = x * diffusors[0]
        x = free_space_propagator(x, propagators[0])
        x1 = 0.5 * x.clone()
        x2 = 0.5 * x.clone()
        for layer_len in params_distr[: len(params_distr) // 2]:  # sublayers x1
            x1 = apply_weights(x1, phase_weights, amplitude_weights, grid_size, w_idx, layer_len)
            x1 = free_space_propagator(x1, propagators[1])
            w_idx += layer_len
        for layer_len in params_distr[len(params_distr) // 2 :]:  # sublayers x2
            x2 = apply_weights(x2, phase_weights, amplitude_weights, grid_size, w_idx, layer_len)
            x2 = free_space_propagator(x2, propagators[1])
            w_idx += layer_len
    return torch.abs(x1) - torch.abs(x2)

CALCULATION_BEAM_FUNCS = {
    "single": calculate_intensity_single_beam,
    "double": calculate_intensity_double_beam,
}


def build_diffraction_func(
    grid_size,
    diffusors,
    x_pad,
    params_distr,
    d_out,
    propagator_list,
    weight_encoding,
    calculate_intensity_func,
):
    """Diffusor as a discrete black box.
    x is a [bsz, d_inp] tensor of float; its values are in (0, 1).
    w is a [d] tensor of float; its values are in (0, 1).
    Function returns the [bsz, d_out] tensor of float.
    """

    def func(x, w, *largs):
        x = padding(x, x_pad)
        # Ensure precomputed tensors are on the same device as input
        diffusors_local = diffusors.to(x.device)
        propagator_list_local = propagator_list.to(x.device) if propagator_list is not None else None
        y_pad = int(np.ceil(np.sqrt(d_out)))
        y = calculate_intensity_func(
            x,
            w,
            diffusors_local,
            params_distr,
            grid_size,
            weight_encoding,
            propagators=propagator_list_local,
        )
        y = output_prepare(y.type(dtype=torch.float32), y_pad)
        y = y[..., :d_out]
        return y

    return func


def _build_free_space_propagators(grid_size, distance_list):
    """Build frequency-domain propagators for free-space propagation."""
    mm = 0.001
    area_size = 10 * mm
    wavelength = 632.0 * 1e-9
    k = 1 / wavelength
    dL = area_size / grid_size
    Kx = torch.fft.fftfreq(grid_size, float(dL))
    Ky = Kx.reshape((-1, 1))
    Kz = torch.sqrt(k**2 - Kx**2 - Ky**2)
    propagator_list = torch.cat([torch.exp(-1j * 2 * np.pi * Kz * z)[None, ::] for z in distance_list])
    return propagator_list


def create_bb_layer_mvm1(d_inp, d_out, scheme="single", weight_encoding="amplitude"):
    """Create MVM1 BB layer.

    Returns a tuple (bb_func, w0, dw) compatible with AstraloraLayer.
    """
    assert scheme in ("single", "double")
    assert weight_encoding in ("amplitude", "phase", "phase_amplitude")

    # Choose square paddings for input/output mapping
    x_pad = int(np.ceil(np.sqrt(d_inp))) ** 2
    y_pad = int(np.ceil(np.sqrt(d_out))) ** 2

    # Parameter distribution: one sublayer with x_pad parameters (per encoding)
    params_distr = [x_pad] if scheme == "single" else [x_pad, x_pad]
    d = sum(params_distr)
    if weight_encoding == "phase_amplitude":
        d *= 2

    grid_size = int(np.ceil(np.sqrt(max(x_pad, y_pad))))

    # Diffusor(s): complex phase screens
    n_diffusors = 1  # for free-space variant used below
    diffusors = torch.rand(n_diffusors, grid_size**2) * torch.pi * 2.0
    diffusors = reshape_and_padding(diffusors, grid_size)
    diffusors = torch.exp(1.0j * diffusors)

    # Free-space propagators
    distance_list = [0.5, 0.5, 0.5]
    propagator_list = _build_free_space_propagators(grid_size, distance_list)

    calculate_intensity_func = CALCULATION_BEAM_FUNCS[scheme]
    bb = build_diffraction_func(
        grid_size=grid_size,
        diffusors=diffusors,
        x_pad=x_pad,
        params_distr=params_distr,
        d_out=d_out,
        propagator_list=propagator_list,
        weight_encoding=weight_encoding,
        calculate_intensity_func=calculate_intensity_func,
    )

    # Initialize weights
    w = torch.empty(d)
    torch.nn.init.uniform_(w, a=0.0, b=1.0)

    dw = 1.0e-4
    return bb, w, dw
