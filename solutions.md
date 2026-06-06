

### 1
```
@triton.jit
def add_kernel(x_ptr, z_ptr, N0, B0: tl.constexpr):
    offset = tl.arange(0, B0)
    mask = offset < N0
    x = tl.load(x_ptr + offset, mask)
    tl.store(z_ptr + offset, x + 10, mask)
```

### 2
```
@triton.jit
def add_mask2_kernel(x_ptr, z_ptr, N0, B0: tl.constexpr):
    pid = tl.program_id(axis=0)
    blk_start = pid * B0
    offset = blk_start + tl.arange(0, B0)
    mask = offset < N0
    x = tl.load(x_ptr + offset, mask)
    tl.store(z_ptr + offset, x + 10, mask)
```

### 3
```
@triton.jit
def add_vec_kernel(x_ptr, y_ptr, z_ptr, N0, N1, B0: tl.constexpr, B1: tl.constexpr):

    pid = tl.program_id(axis=0)
    x_offset = tl.arange(0, B0)
    y_offset = tl.arange(0, B1)
    x_mask = x_offset < N0
    y_mask = y_offset < N1
    
    x = tl.load(x_ptr + x_offset, x_mask)
    y = tl.load(y_ptr + y_offset, y_mask)
    
    # out = y[:, None] + x[None, :]
    out = y[:, None] + x

    # z_mask = y_mask[:, None] & x_mask[None, :]
    z_mask = y_mask[:, None] & x_mask
    # z_offset = y_offset[:, None] * N0 + x_offset[None, :]
    z_offset = y_offset[:, None] * N0 + x_offset

    tl.store(z_ptr + z_offset, out, z_mask)
```
