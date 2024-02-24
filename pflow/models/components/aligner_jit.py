import numpy as np
from numba import jit, prange
import torch


@jit(nopython=True, parallel=True)
def maximum_path_numba(paths, values, t_ys, t_xs):
    b = paths.shape[0]
    for i in prange(b):
        t_y, t_x = t_ys[i], t_xs[i]
        for y in range(t_y):
            for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
                if x == y:
                    v_cur = -1e9
                else:
                    v_cur = values[i, y - 1, x] if y > 0 else -1e9
                v_prev = values[i, y - 1, x - 1] if x > 0 and y > 0 else (0. if x == 0 and y == 0 else -1e9)
                values[i, y, x] += max(v_prev, v_cur)

        index = t_x - 1
        for y in range(t_y - 1, -1, -1):
            paths[i, y, index] = 1
            if index != 0 and (index == y or values[i, y - 1, index] < values[i, y - 1, index - 1]):
                index -= 1


def maximum_path_jit(neg_cent, mask):
    device = neg_cent.device
    dtype = neg_cent.dtype

    neg_cent_np = neg_cent.cpu().numpy().astype(np.float32)
    mask_np = mask.cpu().numpy().astype(np.bool_)

    paths = np.zeros_like(neg_cent_np, dtype=np.int32)
    values = np.copy(neg_cent_np)

    t_ys = mask_np.sum(axis=1)[:, 0].astype(np.int32)
    t_xs = mask_np.sum(axis=2)[:, 0].astype(np.int32)

    maximum_path_numba(paths, values, t_ys, t_xs)

    return torch.from_numpy(paths).to(device=device, dtype=dtype)
