import triton.language as tl
import triton
import math
import torch

@triton.jit
def add_kernel(
    a_ptr, b_ptr, output_ptr, # 指针指向GPU内存
    n, # 数组的长度
    BLOCK_SIZE: tl.constexpr, # 每个线程处理几个数
):
    pid = tl.program_id(0)
    # tl.arrange(0, BLOCK_SIZE): 线程n 这个函数就返回n 
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n
    a = tl.load(a_ptr + offset, mask=mask)
    b = tl.load(b_ptr + offset, mask=mask)
    c = a + b
    tl.store(output_ptr + offset, c, mask=mask)

# 包装函数
def add(a, b):
    assert a.shape == b.shape, "The shape must same"
    assert a.device == b.device, "must on same device"
    assert a.is_cuda, "must in GPU"
    
    output = torch.empty_like(a)
    n = a.numel()
    BLOCK_SIZE = 256
    # 需要的块数
    num_block = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    add_kernel[(num_block,)](
        a, b, output,
        n,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output

a = torch.tensor([1, 2, 3, 4], device = 'cuda')
b = torch.tensor([5, 6, 7, 8], device = 'cuda')
c = add(a, b)
print(c)

