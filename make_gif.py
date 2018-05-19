import imageio
import glob

paths = sorted(glob.glob('results/samples*.jpg'))
images = []
for path in paths:
    images.append(imageio.imread(path))
imageio.mimsave('results/sample.gif', images)
