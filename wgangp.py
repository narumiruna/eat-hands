import argparse
import os
import glob

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils import data
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torchvision.datasets.folder import pil_loader
from model import Generator, Discriminator
from dataset import Hands

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str, default='data')
parser.add_argument('--image-dir', '-d', type=str, default='wgangp')
parser.add_argument('--batch-size', '-bs', type=int, default=128)
parser.add_argument('--learning-rate', '-lr', type=float, default=2e-4)
parser.add_argument('--channels', type=int, default=128)
parser.add_argument('--workers', type=int, default=0)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--log-interval', type=int, default=10)
parser.add_argument('--no-cuda', action='store_true')
parser.add_argument('--penalty-coefficient', type=int, default=10)
args = parser.parse_args()
print(args)

os.makedirs(args.image_dir, exist_ok=True)

use_cuda = torch.cuda.is_available() and not args.no_cuda
if use_cuda:
    print("Use CUDA.")

transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

hand_loader = data.DataLoader(Hands(args.data_dir,
                                    transform=transform),
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.workers)

d = Discriminator(args.channels)
g = Generator(args.channels)

if use_cuda:
    d.cuda()
    g.cuda()

optimizer_d = torch.optim.RMSprop(d.parameters(),
                                  lr=args.learning_rate)
optimizer_g = torch.optim.RMSprop(g.parameters(),
                                  lr=args.learning_rate)

losses_d = []
losses_g = []

def train(epoch):
    g.train()

    for i, x in enumerate(hand_loader):
        # train discriminator
        x = Variable(x)
        z = Variable(torch.randn(len(x), 100))

        epsilon = Variable(torch.rand(len(x), 1, 1, 1))
        grad_outputs = Variable(torch.ones(len(x)))

        if use_cuda:
            x = x.cuda()
            z = z.cuda()
            epsilon = epsilon.cuda()
            grad_outputs = grad_outputs.cuda()

        x_fake = g(z).detach()
        interpolates = epsilon * x + (1 - epsilon) * x_fake
        interpolates.requires_grad = True

        gradients = grad(d(interpolates),
                         interpolates,
                         grad_outputs=grad_outputs,
                         create_graph=True)[0]
        gradient_penalty = (gradients.view(len(x), -1).norm(2, dim=1) - 1).pow(2)

        loss_d = d(x_fake).mean() - d(x).mean() + args.penalty_coefficient * gradient_penalty.mean()

        optimizer_d.zero_grad()
        loss_d.backward()
        optimizer_d.step()

        # train generator
        z = Variable(torch.randn(len(x), 100))

        if use_cuda:
            z = z.cuda()

        loss_g = - d(g(z)).mean()

        optimizer_g.zero_grad()
        loss_g.backward()
        optimizer_g.step()

        # log
        losses_d.append(float(loss_d.data))
        losses_g.append(float(loss_g.data))

        if i % args.log_interval == 0:
            print('{}: {}, {}: {}, {}: {:.4f}, {}: {:.4f}.'.format(
                'Epoch', epoch,
                'index', i,
                'd loss:', float(loss_d.data),
                'g loss', float(loss_g.data)))
            plot_losses()

    plot_samples(epoch)

def plot_losses():
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(nrows=2)
    ax[0].plot(losses_d, c='r', label='d loss')
    ax[0].legend()
    ax[1].plot(losses_g, c='b', label='g loss')
    ax[1].legend()

    fig.savefig(os.path.join(args.image_dir, 'losses.png'))
    plt.close(fig)


def plot_samples(epoch):
    g.eval()

    z = Variable(torch.randn(16 * 16, 100), volatile=True)
    if use_cuda:
        z = z.cuda()

    filename = os.path.join(args.image_dir, 'samples_epoch_{}.jpg'.format(str(epoch).zfill(3)))
    save_image(g(z).data, filename, normalize=True, nrow=16)


for epoch in range(args.epochs):
    train(epoch)
