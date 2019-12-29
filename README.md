# pixelshuffle_invert_pytorch
 Fast pixelshuffle and pixelshuffle-invert implementations via pytorch.  
 
pixelshuffle_invert 是 pixelshuffle 的逆向操作。  
pixelshuffle_invert is the reverse operation of pixelshuffle.  

我的实现与官方(v1.3.1)相比支持不同宽高比的输入。  
My implementation supports input with different aspect ratios compared to the official (v1.3.1).  

# 速度测试 / Speed Test

PixelShuffle  

测试输出。看起来比官方实现还快一点点:)  
Test output. Looks a little faster than the official implementation :)  
```
Warm up
cuda time 1165.0928955078125
perf_counter time 4.8265442
cuda time 1009.6998901367188
perf_counter time 1.0099245999999997
Warm up finish

Testing my speed
cuda time 60574.80859375
perf_counter time 60.57620209999999

Testing pytorch 1.3.1 official speed
cuda time 64837.421875
perf_counter time 64.8391723


Process finished with exit code 0

```

你可以在你的计算机上测试。  
You can test on your computer.  
```
python test_pixelshuffle_speed.py
```

# 怎么用 / How to use

## PixelShuffle
official  
```
import torch
import torch.nn.functional as F

x = torch.rand(5, 256, 128, 128)   # BCHW
y = F.pixel_shuffle(x, 2)
print(y.shape)
```
my code  
```
import torch
from pixelshuffle_invert import pixelshuffle

x = torch.rand(5, 256, 128, 128)   # BCHW
y = pixelshuffle(x, (2, 2))
print(y.shape)
```

## PixelShuffle_invert
no official implementation  

my code  
```
import torch
from pixelshuffle_invert import pixelshuffle_invert

x = torch.rand(5, 256, 128, 128)   # BCHW
y = pixelshuffle_invert(x, (2, 2))
print(y.shape)
```

# References
https://arxiv.org/abs/1609.05158
Wait to add...
