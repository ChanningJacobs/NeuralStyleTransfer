import sys
import tensorflow as tf
from utils.visualize import plot_images_rowwise, images_to_gif
from pathlib import Path
import os

def build_vgg_model():
    """Creates a VGG16 model with outputs at each convolutional layer.
    :return:
    model -- tf.keras.Model with all convolutional outputs exposed
    """
    vgg16 = tf.keras.applications.vgg16.VGG16()
    model = tf.keras.Model(
        inputs=vgg16.inputs, outputs=[layer.output for layer in vgg16.layers if 'conv' in layer.name])
    return model


def reconstruct_content(model, path_content):
    """Show results of gradient descent reconstruction of content image from its activations.
    :return:
    None
    """

    # Load the image whose content we want to preserve
    with open(path_content, 'rb') as image:
        content = tf.io.decode_image(image.read())
    content = tf.expand_dims(tf.cast(tf.image.resize(content, (224, 224)), tf.float32), 0)
    # Generate the activation maps belonging to the content image
    content_activation_maps = model(content)
    # Create a uniform random tf.Variable image that will be used for gradient descent reconstruction of content
    random_image = tf.cast(tf.random.uniform(shape=content.shape, maxval=256, dtype=tf.int32), tf.float32)
    reconstruction = tf.Variable(random_image)
    # Show both starting images
    plot_images_rowwise([content, random_image])
    # Begin gradient descent and update the reconstruction image each time
    optimizer = tf.keras.optimizers.Adam()
    iterations = 9999999999
    for i in range(iterations):
        loss = 0
        with tf.GradientTape() as g:
            activation_maps = model(reconstruction)
            for j in range(len(activation_maps)):
                loss += tf.reduce_mean((content_activation_maps[j] - activation_maps[j]) ** 2)
        grads = g.gradient(loss, reconstruction)
        optimizer.apply_gradients(zip([grads], [reconstruction]))
        if i % 9999 == 0:
            print(f'Completed iteration {i} out of {iterations}.')
            print(f'Current loss: {loss}')
            plot_images_rowwise(
                [content, random_image, reconstruction, random_image-reconstruction],
                f'imagery/reconstruction/iteration_{i}')
    '''
    results = Path('imagery/reconstruction')
    output = Path(results, Path(os.getcwd(), results, 'test.gif'))
    images_to_gif(dir_in=results, path_out=output, duration=50, loop=0, remove_originals=False)
    '''


def main():
    """
    Performs neural style transfer based on paper: https://arxiv.org/pdf/1508.06576.pdf
    :return: None
    """
    print(tf.config.list_physical_devices())
    print(sys.argv)
    model = build_vgg_model()
    reconstruct_content(model, 'imagery/photo/kitten.jpg')


if __name__ == '__main__':
    main()