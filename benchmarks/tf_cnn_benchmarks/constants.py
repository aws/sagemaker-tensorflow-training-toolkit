"""Constants used in tf_cnn_benchmarks."""

from enum import Enum


class NetworkTopology(str, Enum):
  """Network topology describes how multiple GPUs are inter-connected.
  """
  # DGX-1 uses hybrid cube mesh topology with the following device peer to peer
  # matrix:
  # DMA: 0 1 2 3 4 5 6 7
  # 0:   Y Y Y Y Y N N N
  # 1:   Y Y Y Y N Y N N
  # 2:   Y Y Y Y N N Y N
  # 3:   Y Y Y Y N N N Y
  # 4:   Y N N N Y Y Y Y
  # 5:   N Y N N Y Y Y Y
  # 6:   N N Y N Y Y Y Y
  # 7:   N N N Y Y Y Y Y
  DGX1 = "dgx1"

  # V100 in GCP are connected with the following device peer to peer matrix.
  # In this topology, bandwidth of the connection depends on if it uses NVLink
  # or PCIe link.
  # DMA: 0 1 2 3 4 5 6 7
  # 0:   Y Y Y Y N Y N N
  # 1:   Y Y Y Y N N N N
  # 2:   Y Y Y Y N N N Y
  # 3:   Y Y Y Y N N N N
  # 4:   N N N N Y Y Y Y
  # 5:   Y N N N Y Y Y Y
  # 6:   N N N N Y Y Y Y
  # 7:   N N Y N Y Y Y Y
  GCP_V100 = "gcp_v100"
