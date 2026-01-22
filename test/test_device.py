# import pytest
# import torch

# @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
# def test_linear_system_forward_autodevice_move():
#     """Ensure LinearSystem.forward doesn't crash when x is on a different device.

#     We intentionally construct a small system on CUDA but pass x on CPU; the
#     implementation should auto-move x to the system device (with a warning).
#     """
#     from core.linear_system_pair import LinearSystem

#     X, Y, Z = 4, 4, 4
#     A = torch.ones(X, Y, Z, dtype=torch.float16)
#     b = torch.ones(Y, X, dtype=torch.float16)

#     system = LinearSystem([A], [b], device="cuda:0")

#     x_cpu = torch.ones(X, Y, Z, dtype=torch.float16, device="cpu")
#     out = system.forward(x_cpu)
#     assert out.is_cuda
#     assert out.numel() == system.valid_indices.numel()
