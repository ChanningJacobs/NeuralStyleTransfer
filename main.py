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


def reconstruct(model, path_style, path_content, alpha, beta, style_layer_weights, content_layer):
    # Load the image for style extraction and the image for content preservation
    with open(path_style, 'rb') as style_file, open(path_content, 'rb') as content_file:
        style = tf.io.decode_image(style_file.read())
        content = tf.io.decode_image(content_file.read())
    # Reformat to expected model format
    #TODO Use model input shape and create functions to reduce redudancy
    #TODO Check expected format of input (may not effect results, but consider zscorin/etc...)
    style = tf.expand_dims(tf.cast(tf.image.resize(style, (224, 224)), tf.float32), 0)
    content = tf.expand_dims(tf.cast(tf.image.resize(content, (224, 224)), tf.float32), 0)
    # Generate and activation maps from the content and style
    content_activation_maps = model(content)
    style_activation_maps = model(style)
    # Create a uniform random tf.Variable image that will be used for gradient descent
    random_image = tf.cast(tf.random.uniform(shape=content.shape, maxval=256, dtype=tf.int32), tf.float32)
    reconstruction = tf.Variable(random_image)
    # Show input images and starting random image
    plot_images_rowwise([content, style, random_image])
    # Begin gradient descent and update the reconstruction image each iteration
    optimizer = tf.keras.optimizers.Adam()
    iterations = 99999999
    epsilon = 1e-2
    loss = 0
    for i in range(iterations):
        prev_loss = loss
        loss = 0
        with tf.GradientTape() as g:
            reconstruction_activation_maps = model(reconstruction)
            loss = (alpha * content_loss(content_activation_maps, reconstruction_activation_maps, content_layer) +
                    beta * style_loss(style_activation_maps, reconstruction_activation_maps, style_layer_weights))
        grads = g.gradient(loss, reconstruction)
        optimizer.apply_gradients(zip([grads], [reconstruction]))
        if i % 9999 == 0:
            print(f'Completed iteration {i} out of {iterations}.')
            print(f'Current loss: {loss}')
            plot_images_rowwise(
                [style, content, random_image, reconstruction,
                 style-reconstruction, content-reconstruction, random_image-reconstruction],
                f'imagery/reconstruction/iteration_{i}')
        if tf.abs(loss - prev_loss) < epsilon:
            break
    # Create GIF from intermediate reconstruction images
    results = Path('imagery/reconstruction')
    output = Path(results, Path(os.getcwd(), results, 'reconstruction.gif'))
    images_to_gif(dir_in=results, path_out=output, duration=50, loop=0, remove_originals=False)


def style_loss(style_maps, reconstruction_maps, layer_weights):

    def gram_matrix(activation_map):
        shape = tf.squeeze(activation_map).shape
        flattened_map = tf.reshape(activation_map, (shape[0] * shape[1], shape[2]))
        return tf.matmul(tf.transpose(flattened_map), flattened_map)

    loss = 0
    style_gram_matrices = [gram_matrix(act_map) for act_map in style_maps]
    reconstruction_gram_matrices = [gram_matrix(act_map)for act_map in reconstruction_maps]

    for i, act_maps in enumerate(zip(style_gram_matrices, reconstruction_gram_matrices)):
        shape = style_maps[i].shape
        dividend = 4 * shape[1] ** 2 * shape[3] ** 2
        loss += layer_weights[i] * tf.reduce_sum((act_maps[0] - act_maps[1]) ** 2) / dividend
    return loss


def content_loss(content_maps, reconstruction_maps, layer):
    return 0.5 * tf.reduce_sum((content_maps[layer] - reconstruction_maps[layer]) ** 2)


def main():
    """
    Performs neural style transfer based on paper: https://arxiv.org/pdf/1508.06576.pdf
    :return: None
    """
    print(tf.config.list_physical_devices())
    # print(sys.argv)
    model = build_vgg_model()
    # layer_weights = [0] * len(model.outputs)
    layer_weights = [0.2, 0, 0.2, 0, 0.2, 0, 0, 0.2, 0, 0, 0.2, 0, 0]
    # TODO Choose layers script to assign fractions where regex matches model output layer names
    reconstruct(model=model,
                path_style='imagery/photo/starry_night.jpg', path_content='imagery/photo/kitten.jpg',
                alpha=1e-3, beta=1,
                style_layer_weights=layer_weights, content_layer=8)


if __name__ == '__main__':
    main()
