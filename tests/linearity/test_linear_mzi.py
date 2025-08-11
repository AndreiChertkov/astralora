import torch
import pytest

from core.bb_layers.bb_layer_mzi import create_bb_layer_mzi


def _relative_error(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    diff = y_pred - y_true
    num = torch.linalg.norm(diff)
    denom = torch.linalg.norm(y_true)
    # Guard against division by zero
    if denom.item() == 0.0:
        denom = torch.tensor(1.0, device=y_true.device, dtype=y_true.dtype)
    return (num / denom).item()


@pytest.mark.parametrize(
    "d_inp,d_out",
    [
        (3, 3),
        (5, 7),
        (8, 4),
        (9, 9),  # odd -> internal even N
        (1, 2),
        (16, 16),
    ],
)
@pytest.mark.parametrize(
    "a,b",
    [
        (1.0, 1.0),
        (2.5, -0.3),
        (-1.2, 0.7),
        (0.0, 1.0),
        (1.0, 0.0),
    ],
)
def test_linear_mzi_general_linear_combination(d_inp: int, d_out: int, a: float, b: float):
    torch.manual_seed(0)

    bb, w, _ = create_bb_layer_mzi(d_inp, d_out)

    batch_size = 13
    x1 = torch.randn(batch_size, d_inp)
    x2 = torch.randn(batch_size, d_inp)

    with torch.no_grad():
        y_combo = bb(a * x1 + b * x2, w)
        y_lin = a * bb(x1, w) + b * bb(x2, w)

    rel_err = _relative_error(y_combo, y_lin)

    # Expect strict linearity up to numerical precision
    assert rel_err < 1e-6, (
        f"Linearity relative error too high: {rel_err:.3e} for dims {d_inp}->{d_out} "
        f"with a={a}, b={b}"
    )


@pytest.mark.parametrize("d_inp,d_out", [(5, 7), (6, 6), (100, 10), (500, 200)])
def test_linear_mzi_single_vector(d_inp: int, d_out: int):
    torch.manual_seed(123)

    bb, w, _ = create_bb_layer_mzi(d_inp, d_out)

    x = torch.randn(d_inp)
    alpha = -3.7

    with torch.no_grad():
        y_scaled = bb(alpha * x, w)
        y = bb(x, w)

    rel_err = _relative_error(y_scaled, alpha * y)
    assert rel_err < 1e-6, (
        f"Homogeneity relative error too high: {rel_err:.3e} for dims {d_inp}->{d_out} with alpha={alpha}"
    )


@pytest.mark.parametrize("d_inp,d_out", [(200, 200), (300, 300), (500, 500), (500, 200), (100, 10)])
def test_linear_mzi_large_sizes(d_inp: int, d_out: int):
    torch.manual_seed(42)

    bb, w, _ = create_bb_layer_mzi(d_inp, d_out)

    # Keep batch small to control runtime/memory
    batch_size = 2
    x1 = torch.randn(batch_size, d_inp)
    x2 = torch.randn(batch_size, d_inp)

    a, b = 1.3, -0.7

    with torch.no_grad():
        y_combo = bb(a * x1 + b * x2, w)
        y_lin = a * bb(x1, w) + b * bb(x2, w)

    rel_err = _relative_error(y_combo, y_lin)
    assert rel_err < 1e-6, (
        f"Large-size linearity relative error too high: {rel_err:.3e} for dims {d_inp}->{d_out}"
    )


