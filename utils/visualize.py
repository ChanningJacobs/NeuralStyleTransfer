import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import os
from pathlib import Path


def plot_images_rowwise(images, save_path=None):
    """Creates a figure of images side by side. Shows plot or saves out according to save_path's existence."""
    fig, axs = plt.subplots(1, len(images))
    for i, image in enumerate(images):
        axs[i].imshow(tf.cast(image[0], tf.uint8))
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


def images_to_gif(dir_in, path_out, duration=30, loop=0, remove_originals=False):
    """Create a GIF87a from image files in a directory ordered by their file creation time."""
    # Get images in order of creation
    image_paths = [Path(dir_in, file) for file in os.listdir(dir_in)]
    image_paths.sort(key=lambda path: os.path.getctime(path))
    # Use PIL to create GIF
    image, *images = [Image.open(image) for image in image_paths]
    image.save(fp=path_out, format='GIF', append_images=images, save_all=True, duration=duration, loop=loop)
    # Optionally delete individual frames
    if remove_originals:
        for path in image_paths:
            path.unlink()
