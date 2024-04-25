import random
import imageio
import numpy as np
from argparse import ArgumentParser

from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import einops
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from torchvision.transforms import Compose, ToTensor, Lambda
from torchvision.datasets.mnist import MNIST, FashionMNIST

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

STORE_PATH_MNIST = f"ddpm_model_mnist.pt"
STORE_PATH_FASHION = f"ddpm_model_fashion.pt"

no_train = False
fashion = True
batch_size = 20
n_epochs = 20
lr = 0.001
store_path = "ddpm_fashion.pt" if fashion else "ddpm_mnist.pt"
device = "cpu"

def show_images(images, title=""):
    if type(images) is torch.Tensor:
        images = images.detach().cpu().numpy()

    fig = plt.figure(figsize=(8, 8))
    rows = int(len(images) ** 0.5)
    cols = round(len(images) / rows)



    idx = 0
    for r in range(rows):
        for c in range(cols):
            fig.add_subplot(rows, cols, idx + 1)

            if idx < len(images):
                plt.imshow(images[idx][0], cmap="gray")
                idx += 1
    fig.suptitle(title, fontsize=30)

    plt.show()

def show_first_batch(loader):
    for batch in loader:
        show_images(batch[0], "First batch")
        break

def show_forward(ddpm, loader, device):
    for batch in loader:
        imgs = batch[0]

        show_images(imgs, "Original")

        for percent in [0.25, 0.5, 0.75, 1]:
            show_images(
                ddpm(imgs.to(device),[int(percent * ddpm.n_steps) - 1 for _ in range(len(imgs))]), f"DDPM Noisy images {int(percent * 100)}%")
        break

def generate_new_images(ddpm, n_samples=16, device=None, frames_per_gif=100, gif_name="sampling.gif",c=1, h=28, w=28):
    frame_idxs = np.linespace(0,ddpm.n_steps, frames_per_gif).astype(np.uint)
    frames = []

    with torch.no_grad():
        if device is None:
            device = ddpm.device

        x = torch.randn(n_samples, c, h ,w).to(device)

        for idx, t in enumerate(list(range(ddpm.n_steps))[::-1]):
            time_tensor = (torch.ones(n_samples, 1) * t).to(device).long()
            eta_theta = ddpm.backward(x,time_tensor)

            alpha_t = ddpm.alphas[t]
            alpha_t_bar = ddpm.alpha_bars[t]

            x = (1 / alpha_t.sqrt()) * (x - ( 1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta)

            if t > 0:
                z = torch.randn(n_samples, c, h, w).to(device)

                beta_t = ddpm.betas[t]
                sigma_t = beta_t.sqrt()

                prev_alpha_t_bar = ddpm.alpha_bars[t-1] if t > 0 else ddpm.alphas[0]
                beta_tilda_t = ((1 - prev_alpha_t_bar) / (1 - alpha_t_bar)) * beta_t
                sigma_t = beta_tilda_t.sqrt()

                x = x + sigma_t * z

            if idx in frame_idxs or t == 0:
                normalized = x.clone()
                for i in range(len(normalized)):
                    normalized[i] -= torch.min(normalized[i])
                    normalized[i] *= 255 / torch.max(normalized[i])

                frame = einops.rearrange(normalized, "(b1 b2) c h w -> (b1 h) (b2 w) c", b1=int(n_samples ** 0.5))
                frame = frame.cpu().numpy().astype(np.uint8)

                frames.append(frame)

        with imageio.get_write(gif_name, mode="I") as writer:
            for idx, frame in enumerate(frames):
                writer.append_data(frame)
                if idx == len(frames) - 1:
                    for _ in range(frames_per_gif // 3):
                        writer.append_data(frames[-1])

        return x

def sinusoidal_embedding(n,d):
    embedding = torch.zeros(n, d)
    wk = torch.tensor([1 / 10_000 ** (2 * j / d) for j in range(d)])
    wk = wk.reshape((1,d))
    t = torch.arange(n).reshape((n, 1))
    embedding[:,::2] = torch.sin(t * wk[:,::2])
    embedding[:,1::2] = torch.cos(t * wk[:,::2])    # isn't wk[:,1::2] correct?
    return embedding



transform = Compose([
    ToTensor(),
    Lambda(lambda x : (x-0.5) * 2)
])
ds_fn = FashionMNIST if fashion else MNIST
dataset = ds_fn("./datasets",download=True, train=True, transform=transform)
loader = DataLoader(dataset,batch_size, shuffle=True)


show_first_batch(loader)

class MyDDPM(nn.Module):
    def __init__(self, network,n_steps=200, min_beta=10**(-4), max_beta = 0.02, device=None, image_chw=(1,28,28)):
        super(MyDDPM, self).__init__()
        self.n_steps = n_steps
        self.device = device
        self.image_chw = image_chw
        self.network = network.to(device)
        self.betas = torch.linspace(min_beta, max_beta, n_steps).to(device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.tensor([torch.prod(self.alphas[:i+1]) for i in range(len(self.alphas))]).to(device)

    def forward(self, x0, t, eta=None):
        n, c, h, w = x0.shape
        a_bar = self.alpha_bars[t]

        if eta is None:
            eta = torch.randn(n, c, h, w).to(self.device)

        noisy = a_bar.sqrt().reshape(n,1,1,1) * x0 + (1 - a_bar).sqrt().reshape(n,1,1,1) * eta
        return noisy

    def backward(self, x, t):
        self.network(x,t)

class MyBlock(nn.Module):
    def __init__ (self, shape, in_c, out_c, kernel_size=3, stride=1, padding=1, activation=None, normalize=True):
        super(MyBlock, self).__init__()
        self.ln = nn.LayerNorm(shape)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size, stride,padding )
        self.conv2 = nn.Conv2d(in_c, out_c, kernel_size, stride,padding )
        self.activation = nn.SiLU() if activation is None else activation
        self.normalize = normalize

    def forward(self, x):
        out = self.ln(x) if self.normalize else x
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.activation(out)
        return out

class MyUnet(nn.Module):
    def __init__(self, n_steps=1000, time_emb_dim=100):
        super(MyUnet, self).__init__()

        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)

        self.te1 = self._make_te(time_emb_dim, 1)
        self.b1 = nn.Sequential(
            MyBlock((1,28,28), 1, 10),
            MyBlock((10,28,28), 10, 10),
            MyBlock((10,28,28), 10 , 10)
        )
        self.down1 = nn.Conv2d(10,10,4,2,1)

        self.te2 = self._make_te(time_emb_dim, 10)
        self.b2 = nn.Sequential(
            MyBlock((10,14,14), 10, 20),
            MyBlock((20,14,14), 20, 20),
            MyBlock((20,14,14), 20, 20)
        )
        self.down2 = nn.Conv2d(20,20,4,2,1)

        self.te3 = self._make_te(time_emb_dim, 20)
        self.b3 = nn.Sequential(
            MyBlock((20,7,7), 20, 40),
            MyBlock((40,7,7), 40, 40),
            MyBlock((40,7,7), 40, 40),
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(40,40,2,1),
            nn.SiLU(),
            nn.Conv2d(40,40,4,2,1)
        )

        self.te_mid = self._make_te(time_emb_dim,40)
        self.b_mid = nn.Sequential(
            MyBlock((40,3,3), 40, 20),
            MyBlock((20,3,3), 20, 20),
            MyBlock((20,3,3), 20, 40)
        )

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(40,40,4,2,1),
            nn.SiLU(),
            nn.ConvTranspose2d(40,40,2,1)
        )
        self.te4 = self._make_te(time_emb_dim, 80)
        self.b4 = nn.Sequential(
            MyBlock((80,7,7),80,40),
            MyBlock((40,7,7),40,20),
            MyBlock((20,7,7),20,20),
        )

        self.up2 = nn.ConvTranspose2d(20, 20, 4, 2, 1)
        self.te5 = self._make_te(time_emb_dim, 40)
        self.b5 = nn.Sequential(
            MyBlock((40, 14, 14), 40, 20),
            MyBlock((20, 14, 14), 20, 10),
            MyBlock((10, 14, 14), 10, 10)
        )

        self.up3 = nn.ConvTranspose2d(10, 10, 4, 2, 1)
        self.te_out = self._make_te(time_emb_dim, 20)
        self.b_out = nn.Sequential(
            MyBlock((20, 28, 28), 20, 10),
            MyBlock((10, 28, 28), 10, 10),
            MyBlock((10, 28, 28), 10, 10, normalize=False)
        )

        self.conv_out = nn.Conv2d(10, 1, 3, 1, 1)


    def forward(self, x, t):
        t = self.time_embed(t)
        n = len(x)
        out1 = self.b1(x + self.te1(t).reshape(n, -1, 1, 1))
        out2 = self.b2(self.down1(out1) + self.te2(t).reshape(n, -1, 1,1))
        out3 = self.b3(self.down2(out2) + self.te3(t).reshape(n,-1,1,1))

        out_mid = self.b_mid(self.down3(out3) + self.te_mid(t).reshape(n,-1,1,1))

        out4 = torch.cat((out3, self.up1(out_mid)), dim=1)  # (N, 80, 7, 7)
        out4 = self.b4(out4 + self.te4(t).reshape(n, -1, 1, 1))  # (N, 20, 7, 7)

        out5 = torch.cat((out2, self.up2(out4)), dim=1)  # (N, 40, 14, 14)
        out5 = self.b5(out5 + self.te5(t).reshape(n, -1, 1, 1))  # (N, 10, 14, 14)

        out = torch.cat((out1, self.up3(out5)), dim=1)  # (N, 20, 28, 28)
        out = self.b_out(out + self.te_out(t).reshape(n, -1, 1, 1))  # (N, 1, 28, 28)

        out = self.conv_out(out)

        return out

    def _make_te(self, dim_in, dim_out):
        return nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out)
        )

n_steps, min_beta, max_beta = 1000, 10 ** -4, 0.02  # Originally used by the authors
ddpm = MyDDPM(MyUnet(n_steps), n_steps=n_steps, min_beta=min_beta, max_beta=max_beta, device=device)

show_forward(ddpm, loader, device)