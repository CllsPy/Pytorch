import torch

x = torch.tensor([1, 2, 3])
y = torch.tensor([9, 8, 7])

z1 = torch.empty(3)
torch.add(x, y, out=z1)
print(z1, z1.dtype)

print("\n")
print("\n")

z2 = torch.add(x, y) # same thing
z3 = x + y # same thing

# division
z = torch.true_divide(x, y) # element wise div
print(z)

print("\n")
print("\n")

t = torch.zeros(3)
print(t)
print(t.add_(y))

print("\n")
print("\n")

# Expo
z = x.pow(2) # element wise
z = x**2 # same thing
print(z)

print("\n")
print("\n")

z = x > 0 # element wise
print(z)

print("\n")
print("\n")

# mul
x1 = torch.rand((2, 5))
x2 = torch.rand((5, 3))
x3 = torch.mm(x1, x2)
x3 = x1.mm(x2) # samething

print(x3)

print("\n")
print("\n")

matrix_exp = torch.rand(5, 5) # uma matriz pela outra

print(matrix_exp)
print("\n")
print(matrix_exp.matrix_power(2))

print("\n")
print("\n")

# element wise mul
z = x * y
print(z)

# dot produt (mul de uma matriz ela outra)
z = torch.dot(x, y)
print(z)

print("\n")
print("\n")

# bach mul
batch = 32
n = 10
m = 20
p = 30

tensor1 = torch.rand((batch, n, m))
tensor2 = torch.rand((batch, m, p))
out_bmm = torch.bmm(tensor1, tensor2)
print(out_bmm)

print("\n")
print("\n")

## outras operações

sum_t1 = torch.sum(x, dim=0) # soma uma linha/coluna
print(sum_t1)

print("\n")
print("\n")

values, indices = torch.max(x, dim=0) # acha maior valor e sua posição
print(values, indices)

"""
abs_x = torch.abs(x)

z = torch.argmax(x, dim=0) => igual max mas retorna apenas o idx
z = torch.argmin(x, dim=0)

torch.mean(x.float, dim=0)

torch.clamp(x, min=0, max=10) => se for maior que 10, fica 10. se for menor que 0, fica zero.
"""

z = torch.eq(x, y)
print(z) # comparação elemento por elemento

print("\n")
print("\n")

print(y)
print(torch.sort(y, dim=0, descending=False))