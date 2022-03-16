# Style Sleuths

## Datasets

data:

- Cropped, 256x256 resized dataset of all ~8000 images

mini_data:

- Cropped, 256x256 resized dataset of ~200 random images from data
- No artist information

data_resized:

- data, resized to 64x64

mini_data_resized:

- mini_data, resized to 64x64

## Scripts

img_to_bin.py

- Converts all n images in directory to numpy array, saved as a binary file.
- Generates RGB array with size (n, 256, 256, 3), and Grayscale array with size (n, 256, 256)
- Use binary file as array with,
  > array = np.load("file.npy")

pca_manual.ipynb

 - Reduces the raw, (N, d, d, 3) RGB data into Kx3 features, by performing PCA on each channel individually and recombining. 
 - Returns an (n, K, 3) numpy array of features, and a (K, d^2, 3) numpy array to reform each image. 

img_to_bin_sums.py (not used in final results)

- Generates the sums of each RGB channel in each image, divided by its total opacity.
- Contains a `convolute` function, which convolutes an image with either a Sobel or Roberts Cross edge-detecting kernel and returns an array of its edges.

pca_rgb.ipynb (not used in final results)

- Reduces the mini data RGB images to 115 components, using scikit-learn
- Generates a numpy array in the main directory 
- > PLEASE RUN this notebook in order to generate the reduced data in your directory. 
