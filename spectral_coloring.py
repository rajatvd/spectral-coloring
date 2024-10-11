# input binary image (for now binary, later grayscale or color somehow)

# get graph by treating each 1 as a node and connecting it to its neighbors
# color each node by eigenvectors of the laplacian


import numpy as np
import networkx as nx
from cupyx.scipy.sparse import linalg as cupy_linalg
import cupyx.scipy.sparse


def graph_spectrum(image, num_evecs=100):
    # assumes image is binary and returns eigenvectors of laplacian of image graph
    # returns 3 things:
    # - the list of tuples which correspond to positions of the nodes in the image
    # - the num_evecs smallest eigenvectors of the laplacian
    # - the eigenvalues corresponding to the eigenvectors

    G = nx.Graph()

    # add nodes
    lnp = np.argwhere(image == 1)
    l = [tuple(x) for x in lnp]
    G.add_nodes_from(l)

    print(f"{len(G.nodes())} nodes in graph")

    # add edges
    for node in G.nodes():
        x, y = node
        for i, j in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
            if (x + i, y + j) in G.nodes():
                G.add_edge((x, y), (x + i, y + j), weight=1.0)

    S = nx.laplacian_matrix(G)
    print(f"shape of laplacian: {S.shape}")

    S_cu = cupyx.scipy.sparse.csr_matrix(S)

    print(f"computing {num_evecs} eigenvectors")
    w, v = cupy_linalg.eigsh(S_cu, k=num_evecs, which="SA")

    return lnp, v, w
