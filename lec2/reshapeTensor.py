import torch

x = torch.arange(9)

x_33 = x.view(3, 3)
x_33 = x.reshape(3, 3)

y = x_33.t()

x1 = torch.rand((2, 5))
x2 = torch.rand((2, 5))

print("\n")
print("\n")

print(x1)

print("\n")
print("\n")

z = x1.view(-1)
print(z.shape)