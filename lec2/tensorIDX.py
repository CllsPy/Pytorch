import torch

batch_size = 10
features = 25

x = torch.rand((batch_size, features))

print(x[0].shape)
print(x[:, 0].shape)

print("\n")
print("\n")

print(x)
print("\n")
print(x[:, 0])
print(x[2, 0:10]) # second rowm (element 0 to 9)

print("\n")
print("\n")

x[0, 0] = 1000 # mudar valor nessa posição
print(x[0, 0])

print("\n")
print("\n")

x = torch.arange(10)
print(x)
indices = [2, 5, 8]
print(x[indices])

x = torch.rand((3, 5))
print(x)

print("\n")
print("\n")

rows = torch.tensor([1, 0])
cols = torch.tensor([4, 0])

print(x[rows, cols].shape)

## Advanced Indexing

print("\n")
print("\n")

x = torch.arange(10)
print(x[(x < 2) | (x > 8)])

'''
x.remaider(2) == 0 remaider é o que sobra da divisão.
torch.wehre(x > 5, x, x*2)
torch.tensor([0, 0, 1, 2, 3, 4]).unique())
print(x.ndimension()) => retorna as dimensões. 5x5x5 retornaria 3
x.nuemal() => número de elementos
'''

