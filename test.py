import torch

def sinusoidal_embedding(n, d):
    # Returns the standard positional embedding
    embedding = torch.zeros(n, d)
    wk = torch.tensor([1 / 10_000 ** (2 * j / d) for j in range(d)])
    wk = wk.reshape((1, d))
    t = torch.tensor([1,5,2,4,3,4,2,6,8,1]).reshape((n, 1))
    embedding[:,::2] = t * wk[:,::2]
    embedding[:,1::2] = t * wk[:,::2] * 0

    print(t)
    print("=====")
    print(wk[:,::2])
    print("=====")
    print(embedding)
    print("=====")

    return embedding

sinusoidal_embedding(10,6)