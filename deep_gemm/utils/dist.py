from __future__ import annotations
import inspect
import os
import torch
import torch.distributed as dist
import paddle.distributed as paddle_dist
from typing import Tuple

_local_rank = None


def init_dist(local_rank: int, num_local_ranks: int) -> Tuple[int, int, paddle_dist.Group]:
    # NOTES: you may rewrite this function with your own cluster settings
    ip = os.getenv('MASTER_ADDR', '127.0.0.1')
    port = int(os.getenv('MASTER_PORT', '8361'))
    num_nodes = int(os.getenv('WORLD_SIZE', 1))
    node_rank = int(os.getenv('RANK', 0))

    # Set local rank
    global _local_rank
    _local_rank = local_rank

    # Use paddle native distributed init
    world_size = num_nodes * num_local_ranks
    rank = node_rank * num_local_ranks + local_rank

    os.environ['PADDLE_TRAINER_ID'] = str(rank)
    os.environ['PADDLE_TRAINERS_NUM'] = str(world_size)
    os.environ['PADDLE_TRAINER_ENDPOINTS'] = ','.join(
        [f'{ip}:{port + i}' for i in range(world_size)]
    )
    os.environ['PADDLE_CURRENT_ENDPOINT'] = f'{ip}:{port + rank}'
    os.environ['FLAGS_selected_gpus'] = str(local_rank)

    paddle_dist.init_parallel_env()
    torch.set_default_device('cuda')
    torch.cuda.set_device(local_rank)

    group = paddle_dist.new_group(list(range(world_size)))
    return paddle_dist.get_rank(), paddle_dist.get_world_size(), group


def uneven_all_gather(tensor: torch.Tensor, dim: int = 0, group: paddle_dist.Group = None) -> torch.Tensor:
    world_size = paddle_dist.get_world_size()

    # Exchange sizes
    local_dim_size = torch.tensor([tensor.shape[dim]], device=tensor.device, dtype=torch.long)
    all_dim_sizes = [torch.zeros_like(local_dim_size) for _ in range(world_size)]
    paddle_dist.all_gather(all_dim_sizes, local_dim_size, group=group)
    all_dim_sizes = [s.item() for s in all_dim_sizes]
    max_dim_size = max(all_dim_sizes)

    # Pad
    if tensor.shape[dim] < max_dim_size:
        pad_shape = list(tensor.shape)
        pad_shape[dim] = max_dim_size - tensor.shape[dim]
        padding = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
        tensor_padded = torch.cat([tensor, padding], dim=dim)
    else:
        tensor_padded = tensor.contiguous()

    # All-gather
    gathered = [torch.zeros_like(tensor_padded) for _ in range(world_size)]
    paddle_dist.all_gather(gathered, tensor_padded, group=group)

    # Remove padding
    trimmed = [
        torch.narrow(gathered[i], dim, 0, all_dim_sizes[i])
        for i in range(world_size)
    ]
    return torch.cat(trimmed, dim=dim)


def dist_print(s: str = '', once_in_node: bool = False) -> None:
    global _local_rank
    assert _local_rank is not None
    if not once_in_node or _local_rank == 0:
        print(s, flush=True)
    paddle_dist.barrier()
