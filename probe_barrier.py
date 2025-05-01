# probe_barrier.py
import os, sys, time, datetime, argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def log(msg: str, rank: int):
    """timestamped, unbuffered print"""
    print(f"[{rank}|{time.time():.3f}] {msg}", flush=True)

def worker(rank: int, world_size: int, backend: str):
    # ─── mandatory NCCL housekeeping ────────────────────────────
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29501")
    os.environ["RANK"]       = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    if backend == "nccl":
        torch.cuda.set_device(rank)           # 1 GPU per rank
    # ────────────────────────────────────────────────────────────

    dist.init_process_group(
        backend          = backend,
        rank             = rank,
        world_size       = world_size,
        timeout          = datetime.timedelta(seconds=30)  # fail fast
    )

    log("reached barrier()", rank)
    dist.barrier()
    log("*** passed  barrier()", rank)

    # Try another collective just to be sure
    tensor = torch.tensor([rank], device="cuda" if backend == "nccl" else "cpu")
    dist.all_reduce(tensor)
    log(f"all_reduce ok, value={tensor.item()}", rank)

    dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nprocs",  type=int,   default=2)
    parser.add_argument("--backend", choices=["gloo", "nccl"], default="gloo")
    args = parser.parse_args()

    mp.spawn(
        worker,
        args=(args.nprocs, args.backend),
        nprocs=args.nprocs,
        join=True
    )

if __name__ == "__main__":
    # Completely unbuffered stdout/stderr
    os.environ["PYTHONUNBUFFERED"] = "1"
    main()
