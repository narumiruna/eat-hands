import argparse
import os
import glob

import torch
from torch import nn
from torch.autograd import grad
from torch.utils import data
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torchvision.datasets.folder import pil_loader
from model import Generator, Discriminator
from dataset import Hands

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str, default='data')
parser.add_argument('--output-dir', '-d', type=str, default='results')
parser.add_argument('--batch-size', '-bs', type=int, default=128)
parser.add_argument('--learning-rate', '-lr', type=float, default=1e-4)
parser.add_argument('--channels', type=int, default=128)
parser.add_argument('--workers', type=int, default=0)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--log-interval', type=int, default=10)
parser.add_argument('--no-cuda', action='store_true')
parser.add_argument('--penalty-coefficient', type=int, default=10)
args = parser.parse_args()
print(args)

os.makedirs(args.output_dir, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda
                      else 'cpu')

transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

hand_loader = data.DataLoader(
    Hands(args.data_dir, transform=transform),
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.workers)

net_d = Discriminator(args.channels).to(device)
net_g = Generator(args.channels).to(device)

optimizer_d = torch.optim.Adam(
    net_d.parameters(), lr=args.learning_rate, betas=(0, 0.9))
optimizer_g = torch.optim.Adam(
    net_g.parameters(), lr=args.learning_rate, betas=(0, 0.9))

losses_d = []
losses_g = []


def train(epoch):
    net_g.train()

    for i, x in enumerate(hand_loader):
        # train discriminator
        x = x.to(device)
        z = torch.randn(x.size(0), 100).to(device)

        epsilon = torch.rand(x.size(0), 1, 1, 1).to(device)
        grad_outputs = torch.ones(x.size(0)).to(device)

        x_fake = net_g(z).detach()
        interpolates = torch.tensor(
            epsilon * x + (1 - epsilon) * x_fake, requires_grad=True).to(device)

        gradients = grad(
            net_d(interpolates),
            interpolates,
            grad_outputs=grad_outputs,
            create_graph=True)[0]
        gradient_penalty = (
            gradients.view(x.size(0), -1).norm(2, dim=1) - 1).pow(2)

        loss_d = net_d(x_fake).mean() \
               - net_d(x).mean() \
               + args.penalty_coefficient * gradient_penalty.mean()

        optimizer_d.zero_grad()
        loss_d.backward()
        optimizer_d.step()

        # train generator
        z = torch.randn(x.size(0), 100).to(device)

        loss_g = -net_d(net_g(z)).mean()

        optimizer_g.zero_grad()
        loss_g.backward()
        optimizer_g.step()

        # log
        losses_d.append(float(loss_d.data))
        losses_g.append(float(loss_g.data))

        if i % args.log_interval == 0:
            print('{}: {}, {}: {}, {}: {:.4f}, {}: {:.4f}.'.format(
                'Epoch', epoch, 'index', i, 'd loss:', float(loss_d.data),
                'g loss', float(loss_g.data)))
            plot_losses()


def plot_losses():
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(nrows=2)
    ax[0].plot(losses_d, c='r', label='d loss')
    ax[0].legend()
    ax[1].plot(losses_g, c='b', label='g loss')
    ax[1].legend()

    fig.savefig(os.path.join(args.output_dir, 'losses.png'))
    plt.close(fig)


def plot_sample(epoch):
    net_g.eval()

    with torch.no_grad():
        z = torch.randn(16 * 16, 100).to(device)
        fake = net_g(z)

    filename = os.path.join(args.output_dir,
                            'samples_{:03d}.jpg'.format(epoch + 1))
    save_image(fake.data, filename, normalize=True, nrow=16)


for epoch in range(args.epochs):
    train(epoch)
    plot_sample(epoch)