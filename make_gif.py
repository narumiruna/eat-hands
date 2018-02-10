import imageio
import glob

paths = sorted(glob.glob('images/samples*.jpg'))
images = []
for path in paths:
    images.append(imageio.imread(path))
imageio.mimsave('images/sample.gif', images, fps=0.5)


