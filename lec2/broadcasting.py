import torch

x1 = torch.rand((5, 5))
x2 = torch.rand((1, 5))

print(x1 ** x2)

print("\n")
print("\n")

import torch
main_dish = torch.tensor([[1, 2, 3],
                          [4, 5, 6],
                          [7, 8, 9],
                          [10, 11, 12]])  # shape: (4, 3)

dessert = torch.tensor([10, 20, 30])     # shape: (3,)  

meal = main_dish + dessert  # dessert is broadcast to shape (4, 3)

print(meal)

