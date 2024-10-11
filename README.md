# Spectral Coloring

Input: binary image

![Input](h.png)

* Construct graph of image pixels
 - Each 1 pixel is a node
 - Each node is connected to 1 pixels in 4 orthogonal directions
* Compute Laplacian of graph
* Compute eigenvectors of Laplacian
* Color 1 pixels based on eigenvectors
* Interpolate between eigenvectors

Output: `python example.py`

![Output](h.gif)

