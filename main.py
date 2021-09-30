import sys
import tensorflow as tf
import matplotlib.pyplot as plt


def build_vgg_model():
    """Creates a VGG16 model with outputs at each convolutional layer.
    :return:
    """
    vgg16 = tf.keras.applications.vgg16.VGG16()
    model = tf.keras.Model(
        inputs=vgg16.inputs, outputs=[layer.output for layer in vgg16.layers if 'conv' in layer.name])
    return model


def reconstruct_content(model, path_content):
    """Show results of gradient descent reconstruction of content image from activations.
    :return:
    """

    with open(path_content, 'rb') as image:
        content = tf.io.decode_image(image.read())
    content = tf.expand_dims(tf.cast(tf.image.resize(content, (224, 224)), dtype=tf.float32), 0)
    reconstruction = tf.Variable(tf.cast(tf.random.uniform(shape=content.shape, maxval=256, dtype=tf.int32), tf.float32))

    content_activation_maps = model(content)

    plt.imshow(tf.cast(content[0], tf.uint8))
    plt.show()
    plt.imshow(tf.cast(reconstruction[0], tf.uint8))
    plt.show()

    optimizer = tf.keras.optimizers.Adam()
    for i in range(9999):
        loss = 0
        with tf.GradientTape() as g:
            g.watch(reconstruction) # Must call watch on a tf.Tensor
            activation_maps = model(reconstruction)
            for j in range(len(activation_maps)):
                loss += tf.reduce_sum((content_activation_maps[j] - activation_maps[j]) ** 2)
        grads = g.gradient(loss, reconstruction)
        optimizer.apply_gradients(zip([grads], [reconstruction]))
        print(loss)
    plt.imshow(tf.cast(reconstruction[0], tf.uint8))
    plt.show()


def main():
    """
    Performs neural style transfer based on paper: https://arxiv.org/pdf/1508.06576.pdf
    :return: None
    """
    print(sys.argv)
    model = build_vgg_model()
    reconstruct_content(model, 'imagery/photo/kitten.jpg')


if __name__ == '__main__':
    main()
