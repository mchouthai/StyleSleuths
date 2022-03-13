# Style Sleuths

## DATASETS

data:

- Cropped, resized dataset of all ~8000 images

mini_data:

- Cropped, resized dataset of ~200 random images from data
- No artist information

## SCRIPTS

img_to_bin.py

- > python img_to_bin.py path/to/dir
- Converts all _n_ images in directory to numpy array, saved as a binary file.
- Generates RGB array with size (_n_, 256, 256, 3), and Grayscale array with size (_n_, 256, 256)
- Use binary file as array with,
  > array = np.load("file.npy")

pca_rgb.ipynb

- Reduces the mini data RGB images to 115 components 
- Generates a numpy array in the main directory 
- > PLEASE RUN this notebook in order to generate the reduced data in your directory. 
