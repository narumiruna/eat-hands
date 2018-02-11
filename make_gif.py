import imageio
import glob

paths = sorted(glob.glob('wgan/samples*.jpg'))
images = []
for path in paths:
    images.append(imageio.imread(path))
imageio.mimsave('wgan/sample.gif', images)
