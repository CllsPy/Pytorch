import torch

# device = "cuda" if not torch.cuda.is_available() else "cpu"

my_tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32,
                         device="cpu")

print(my_tensor, my_tensor.dtype, my_tensor.device,
      my_tensor.shape)

## other initialization methods
x = torch.empty(size = (3, 3))
print(x, x.device)

y = torch.zeros((3, 3))
print(y)

z = torch.rand((3, 3))
print(z)