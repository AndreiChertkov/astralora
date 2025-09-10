"""bb_layer_id.

Zero-parameter identity-like BB layer that adapts dimensions via resampling.

Behavior (last-dimension only):
- If d_inp == d_out: pass-through.
- If d_inp != d_out: upsample/downsample using 1D interpolation (linear when possible,
  nearest when a length is 1).
"""
import torch
import torch.nn.functional as F


def create_bb_layer_id(d_inp, d_out):
    def bb(x, w):
        # Resample last dimension to d_out using 1D interpolation
        cur_len = x.shape[-1]
        if cur_len == d_out:
            return x

        # Flatten leading dims into batch for interpolation: (B, 1, L)
        batch = int(x.numel() // cur_len)
        x_reshaped = x.reshape(batch, 1, cur_len)

        # Choose interpolation mode
        if cur_len < 2 or d_out < 2:
            mode = 'nearest'
            align = None
        else:
            mode = 'linear'
            align = True

        if mode == 'nearest':
            y = F.interpolate(x_reshaped, size=d_out, mode=mode)
        else:
            y = F.interpolate(x_reshaped, size=d_out, mode=mode, align_corners=align)

        y = y.reshape(*x.shape[:-1], d_out) * w
        return y

    # Zero parameters
    w = torch.ones(1)

    # Any value; unused by this layer but required by framework
    dw = 1.E-4

    return bb, w, dw