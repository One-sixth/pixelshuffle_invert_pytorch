import torch
import torch.nn.functional as F
from pixelshuffle_invert import pixelshuffle
import time


def test_speed(func, args, epoch=50):
    a = torch.rand(10, 64, 512, 512)
    a.requires_grad = True

    start_record = torch.cuda.Event(enable_timing=True)
    end_record = torch.cuda.Event(enable_timing=True)

    start_time = time.perf_counter()
    start_record.record()
    for _ in range(epoch):
        loss = func(a, **args).mean()
        loss.backward()
    end_record.record()
    end_time = time.perf_counter()

    torch.cuda.synchronize()

    print('cuda time', start_record.elapsed_time(end_record))
    print('perf_counter time', end_time - start_time)


if __name__ == '__main__':
    print('Warm up')
    test_speed(pixelshuffle, {'factor_hw': (2, 2)}, epoch=1)
    test_speed(F.pixel_shuffle, {'upscale_factor': 2}, epoch=1)
    print('Warm up finish')
    print()

    print('Testing my speed')
    test_speed(pixelshuffle, {'factor_hw': (2, 2)})
    print()

    print('Testing pytorch {} official speed'.format(torch.__version__))
    test_speed(F.pixel_shuffle, {'upscale_factor': 2})
    print()
