import numpy as np
import matplotlib.pyplot as plt
from spectral_coloring import graph_spectrum
import imageio
import cairosvg
import matplotlib


num_evecs = 100
cairosvg.svg2png(
    url="h.svg",
    write_to="h.png",
    output_width=3000,
    output_height=3000,
)
image = imageio.imread("h.png")
if len(image.shape) == 3:
    image = image.mean(axis=2)
image = image > (255 / 5)
image = image.astype(np.float32)

# crop image to be smallest square containing the letter
padding = 10
nonzero = np.argwhere(image)
min_x, min_y = nonzero.min(axis=0)
max_x, max_y = nonzero.max(axis=0)
image = image[min_x - padding : max_x + padding, min_y - padding : max_y + padding]
imageio.imwrite("h.png", (image * 255).astype("uint8"))

ii, outs_unscale, w = graph_spectrum(image, num_evecs=num_evecs + 1)

# rescale all eigenvectors to be between 0 and 1
outs = outs_unscale - outs_unscale.min(axis=0)
outs = outs / outs.max(axis=0)
outs = outs[:, 1:]  # remove the first eigenvector

inv_ii_mask = np.zeros_like(image)
inv_ii_mask[ii[:, 0], ii[:, 1]] = 1
inv_ii = np.argwhere(inv_ii_mask == 0)
out_images = np.zeros((*image.shape, outs.shape[1]))
out_images[ii[:, 0], ii[:, 1]] = outs.get()
out_images[inv_ii[:, 0], inv_ii[:, 1]] = 0.0


# make ts an array of floats from 0 to num_evecs, with every integer appearing.
# additionally add equally spaced points between each integer equal to count
count = 5
ts = np.concatenate(
    [np.linspace(i, i + 1, count, endpoint=False) for i in range(num_evecs - 1)]
)

# add the last integer
ts = np.concatenate([ts, [num_evecs - 1]])

# interpolate eigenvectors
cm = plt.get_cmap("inferno")
imgs = []
for i, t in enumerate(ts):
    if i != len(ts) - 1:
        a = out_images[:, :, int(t)]
        b = out_images[:, :, int(t) + 1]
        frac = t - int(t)
        print(t, frac)
        interp = frac * b + (1 - frac) * a
        img = cm(1 - interp)
        img[inv_ii[:, 0], inv_ii[:, 1], :] = 0.0
        img[inv_ii[:, 0], inv_ii[:, 1], 3] = 0.0

    else:
        img = cm(1 - out_images[:, :, -1])
        img[inv_ii[:, 0], inv_ii[:, 1], :] = 0.0
        img[inv_ii[:, 0], inv_ii[:, 1], 3] = 0.0

    imgs.append((img * 255).astype("uint8"))

imageio.mimsave("h.gif", imgs[:], loop=0, fps=10)
