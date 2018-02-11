from torch.utils import data
import pandas as pd
from os.path import join
from torchvision.datasets.folder import pil_loader


class Hands(data.Dataset):
    img_folder = 'Hands'
    info_file = 'HandInfo.txt'

    def __init__(self, root, aspect_of_hand='dorsal right', transform=None):
        self.root = root
        self.transform = transform

        df = pd.read_csv(join(root, self.info_file))
        self.image_names = df[df.aspectOfHand == aspect_of_hand].imageName.values.tolist()

    def __getitem__(self, index):
        path = join(self.root, self.img_folder, self.image_names[index])
        image = pil_loader(path)

        if self.transform:
            image = self.transform(image)

        return image

    def __len__(self):
        return len(self.image_names)
