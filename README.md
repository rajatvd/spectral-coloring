# Spectral Coloring

Input: binary image

![Input](h.png)

* Construct graph of image pixels 
    - Each active pixel is a node
    - Each node is connected to active pixels in 4 orthogonal directions
* Compute Laplacian of graph
* Compute eigenvectors of Laplacian
* Color active pixels based on eigenvectors
* Interpolate between eigenvectors

Output: `python example.py`

![Output](h.gif)

