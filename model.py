from torch import nn

class Generator(nn.Module):
    def __init__(self, ch):
        super(Generator, self).__init__()

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(100, ch * 8, 4, 1, 0, 0, bias=False),
            nn.InstanceNorm2d(ch * 8, affine=True),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(ch * 8, ch * 4, 5, 2, 2, 1, bias=False),
            nn.InstanceNorm2d(ch * 4, affine=True),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(ch * 4, ch * 2, 5, 2, 2, 1, bias=False),
            nn.InstanceNorm2d(ch * 2, affine=True),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(ch * 2, ch, 5, 2, 2, 1, bias=False),
            nn.InstanceNorm2d(ch, affine=True),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(ch, 3, 5, 2, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input_):
        return self.conv(input_.view(-1, 100, 1, 1))


class Discriminator(nn.Module):
    def __init__(self, ch):
        super(Discriminator, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, ch, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ch, ch * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ch * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ch * 2, ch * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ch * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ch * 4, ch * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ch * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ch * 8, 1, 4, 1, 0, bias=False)
        )

    def forward(self, input_):
        return self.conv(input_).view(-1)
