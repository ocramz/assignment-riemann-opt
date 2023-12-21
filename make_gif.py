import imageio
import os.path
from datetime import datetime

N = 500
images = []
# for filename in glob.glob('ani/frame_*.png'):
#     images.append(imageio.imread(filename))
for i in range(N):
    fname = f'ani/frame_{i}.png'
    if os.path.isfile(fname):
        images.append(imageio.imread(fname))
    else:
        break

unix_timestamp = round((datetime.now() - datetime(1970, 1, 1)).total_seconds())
imageio.mimsave(f'ani/out/movie_{unix_timestamp}.gif', images, fps=30, loop=0)
