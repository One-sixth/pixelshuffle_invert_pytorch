import torch
from typing import Tuple


@torch.jit.script
def pixelshuffle(x: torch.Tensor, factor_hw: Tuple[int, int]):
    pH = factor_hw[0]
    pW = factor_hw[1]
    y = x
    B, iC, iH, iW = y.shape
    oC, oH, oW = iC//(pH*pW), iH*pH, iW*pW
    y = y.reshape(B, oC, pH, pW, iH, iW)
    y = y.permute(0, 1, 4, 2, 5, 3)     # B, oC, iH, pH, iW, pW
    y = y.reshape(B, oC, oH, oW)
    return y


@torch.jit.script
def pixelshuffle_invert(x: torch.Tensor, factor_hw: Tuple[int, int]):
    pH = factor_hw[0]
    pW = factor_hw[1]
    y = x
    B, iC, iH, iW = y.shape
    oC, oH, oW = iC*(pH*pW), iH//pH, iW//pW
    y = y.reshape(B, iC, oH, pH, oW, pW)
    y = y.permute(0, 1, 3, 5, 2, 4)     # B, iC, pH, pW, oH, oW
    y = y.reshape(B, oC, oH, oW)
    return y


if __name__ == '__main__':
    import torch.nn.functional as F

    print('Check function correct')
    print()

    for s in [1, 2, 4, 8, 16]:
        print('Checking scale {}'.format(s))
        x = torch.rand(5, 256, 128, 128)   # BCHW

        y1 = F.pixel_shuffle(x, s)
        y2 = pixelshuffle(x, (s, s))

        assert torch.allclose(y1, y2)
        print('pixelshuffle works correctly.')

        rev_x = pixelshuffle_invert(y1, (s, s))

        assert torch.allclose(x, rev_x)
        print('pixelshuffle_invert works correctly.')
        print()
