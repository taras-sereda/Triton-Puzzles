

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

### 4
```
@triton.jit
def add_vec_block_kernel(x_ptr, y_ptr, z_ptr, N0, N1, B0: tl.constexpr, B1: tl.constexpr):
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)
    # not used, but helpful to get block dims
    # x_n_blk = tl.num_programs(0)
    # y_n_blk = tl.num_programs(1)
    # print("n blocks", x_n_blk, y_n_blk)

    x_offset = pid_0*B0 + tl.arange(0, B0)
    y_offset = pid_1*B1 + tl.arange(0, B1)

    x_mask = x_offset < N0
    y_mask = y_offset < N1
    
    x = tl.load(x_ptr + x_offset, x_mask)
    y = tl.load(y_ptr + y_offset, y_mask)
    
    out = y[:, None] + x

    z_mask = y_mask[:, None] & x_mask

    # mental model and memo, y and x offsets are already absolute.
    # so linear index lookup will boil down to the classic: y_idx*(x_size) + x_idx
    z_offset = y_offset[:, None] * N0 + x_offset

    tl.store(z_ptr + z_offset, out, z_mask)
```
