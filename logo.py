import numpy as np
from spectral_coloring import spectral_color
import imageio
import cairosvg
import matplotlib
from scipy.ndimage import binary_dilation

cairosvg.svg2png(
    url="e.svg",
    write_to="e.png",
    output_width=1500,
    output_height=1500,
)
image = imageio.imread("e.png")[:, :, 3]
image = image > 128
binary_dilation(image, output=image, iterations=15)
image = image.astype(np.float32)
image = image[130:-210, 170:-170]
imageio.imwrite("e.png", (image * 255).astype("uint8"))


ii, outs_unscale, w = spectral_color(image)

# rescale all eigenvectors to be between 0 and 1
outs = outs_unscale - outs_unscale.min(axis=0)
outs = outs / outs.max(axis=0)

outs = outs[:, 1:]
# inv_ii, inv_outs_unscale = spectral_color(inverse_image)

# rescale inv_outs to be between 0.45 and 0.55

# inv_outs = inv_outs_unscale - inv_outs_unscale.min(axis=0)
# inv_outs = inv_outs / inv_outs.max(axis=0)
# inv_outs = 0.3 + 0.0 * inv_outs

# print(out_image)

inv_ii_mask = np.zeros_like(image)
inv_ii_mask[ii[:, 0], ii[:, 1]] = 1
inv_ii = np.argwhere(inv_ii_mask == 0)

out_images = np.zeros((*image.shape, outs.shape[1]))
out_images[ii[:, 0], ii[:, 1]] = outs.get()
out_images[inv_ii[:, 0], inv_ii[:, 1]] = 0.0

num_evecs = outs.shape[1]


# cm = plt.get_cmap("winter")
# cm = plt.get_cmap("RdYlGn")
cm = matplotlib.colors.LinearSegmentedColormap.from_list(
    "",
    [
        "#66CF66",
        "#205f20",
        "#FF6464",
    ],
)

# make ts an array of floats from 0 to num_evecs, with every integer appearing.
# additionally add equally spaced points between each integer equal to count
count = 15
ts = np.concatenate(
    [np.linspace(i, i + 1, count, endpoint=False) for i in range(num_evecs - 1)]
)
# add the last integer
ts = np.concatenate([ts, [num_evecs - 1]])
ts = ts[count * 10 :]
imgs = []
for i, t in enumerate(ts):
    if i != len((ts)) - 1:
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

imageio.mimsave("out.gif", imgs[:], loop=0, fps=10)

# static logo

# logo = out_images[:, :, 33] + out_images[:, :, 32]
# logo = logo - logo.min()
# logo = logo / logo.max()


# clip to 1
# logo = 1-np.clip(logo, 0, 1.0)

cm2 = matplotlib.colors.LinearSegmentedColormap.from_list(
    "",
    [
        "#66CF66",
        "#205f20",
        "#FF6464",
    ],
)

logo = out_images[:, :, 5]
logo = cm2(1 - logo)
logo[inv_ii[:, 0], inv_ii[:, 1], :] = 0.0
# logo = logo[100:-200, 300:-300]
logo = (logo * 255).astype("uint8")
# logo = imgs[200]
imageio.imwrite("logo.png", logo)
