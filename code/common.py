import os
import torch


def get_args():
  envvars = [
    "WORLD_SIZE",
    "RANK",
    "LOCAL_RANK",
    "NODE_RANK",
    "NODE_COUNT",
    "HOSTNAME",
    "MASTER_ADDR",
    "MASTER_PORT",
    "NCCL_SOCKET_IFNAME",
    "OMPI_COMM_WORLD_RANK",
    "OMPI_COMM_WORLD_SIZE",
    "OMPI_COMM_WORLD_LOCAL_RANK",
    "AZ_BATCHAI_MPI_MASTER_NODE",
    "AMLT_OUTPUT_DIR",
    "AMLT_DATA_DIR",
  ]
  args = dict(gpus_per_node=torch.cuda.device_count())
  missing = []
  for var in envvars:
    if var in os.environ:
      args[var] = os.environ.get(var)
      try:
        args[var] = int(args[var])
      except ValueError:
        pass
    else:
      missing.append(var)
  print(f"II Args: {args}")
  if missing:
    print(f"II Environment variables not set: {', '.join(missing)}.")
  return args
