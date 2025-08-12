import torch

tensor = torch.arange(4)

print(tensor, tensor.dtype)
print(tensor.bool(), tensor.dtype)
print(tensor.short(), tensor.dtype)
print(tensor.long(), tensor.dtype)
print(tensor.half(), tensor.dtype)
print(tensor.float(), tensor.dtype)
print(tensor.double(), tensor.dtype)

print("\n")
print("############")
print("\n")

import numpy as np
np_array = np.zeros((5, 3))
print(np_array.dtype)
tensor = torch.from_numpy(np_array)
print(tensor.dtype)
back_array = tensor.numpy()
print(back_array.dtype)