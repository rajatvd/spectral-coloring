# input binary image (for now binary, later grayscale or color somehow)

# get graph by treating each 1 as a node and connecting it to its neighbors
# color each node by eigenvectors of the laplacian


import numpy as np
import networkx as nx
from scipy.sparse import linalg

from cupyx.scipy.sparse import linalg as cupy_linalg
import cupyx.scipy.sparse


def spectral_color(image):
    G = nx.Graph()

    # add nodes
    lnp = np.argwhere(image == 1)
    l = [tuple(x) for x in lnp]
    G.add_nodes_from(l)

    print(len(G.nodes()))

    # add edges
    for node in G.nodes():
        x, y = node
        for i in range(-1, 2):
            for j in range(-1, 2):
                if (i, j) != (0, 0):
                    if (x + i, y + j) in G.nodes():
                        G.add_edge((x, y), (x + i, y + j), weight=1.0)

    S = nx.laplacian_matrix(G)
    print(S.shape)

    S_cu = cupyx.scipy.sparse.csr_matrix(S)

    # w, v = linalg.eigsh(S, k=3, which="SM")
    w, v = cupy_linalg.eigsh(S_cu, k=10, which="SA")

    # rescale all eigenvectors to be between 0.1 and 1

    v = (v - v.min(axis=0)) / (v.max(axis=0) - v.min(axis=0))
    v = 0.1 + 0.9 * v

    return lnp, v.get()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from gifify import gifify
    import imageio

    # image = np.array(
    #     [
    #         [0, 0, 1, 0, 0],
    #         [0, 1, 1, 1, 0],
    #         [1, 1, 1, 1, 1],
    #         [0, 1, 1, 1, 0],
    #         [0, 0, 1, 0, 0],
    #     ]
    # ).astype(np.float32)

    image = imageio.imread("e.jpg")[:, :, 0]
    image = (image < 128).astype(np.float32)

    plt.figure()
    plt.axis("off")
    plt.imshow(image, cmap="Greens")
    plt.savefig("e_plt.png")

    ii, outs = spectral_color(image)
    # print(out_image)

    out_images = np.zeros((*image.shape, outs.shape[1]))
    out_images[ii[:, 0], ii[:, 1]] = outs

    # interpolate through all eigenvectors
    for t in gifify(np.linspace(0, outs.shape[1] - 1, 100)[:-1]):
        a = out_images[:, :, int(t)]
        b = out_images[:, :, int(t) + 1]
        frac = t - int(t)
        print(t, frac)
        interp = frac * b + (1 - frac) * a
        plt.figure()
        plt.axis("off")
        plt.imshow(interp, cmap="Greens")
        plt.show()
