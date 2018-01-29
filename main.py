import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
plt.interactive(True)

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))
trainable_variables = []
# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

global x

def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    graph = tf.get_default_graph()
    w1 = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return w1, keep, layer3, layer4, layer7
#tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    # conv_1x1_7 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding='same',
    #                               kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-2), name='new_conv_1x1_7')
    # conv_1x1_7 = tf.layers.batch_normalization(conv_1x1_7)
    # conv_1x1_7_up = tf.layers.conv2d_transpose(conv_1x1_7, num_classes, 4, padding='same',strides=(2, 2), name ='new_conv_1x1_7_up',
    #                                            kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-2))
    #
    # conv_1x1_4 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, padding='same',
    #                               kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-2), name='new_onv_1x1_4')
    # conv_1x1_4 = tf.layers.batch_normalization(conv_1x1_4)
    # add7_4 = tf.add(conv_1x1_7_up, conv_1x1_4, name='new_add7_4')
    #
    # add7_4_up = tf.layers.conv2d_transpose(add7_4, num_classes, 4, padding='same',strides=(2, 2), name='new_add7_4_up',
    # kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-2))
    #
    # conv_1x1_3 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, padding='same',
    #                               kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-2), name='new_conv_1x1_3')
    # conv_1x1_3 = tf.layers.batch_normalization(conv_1x1_3)
    # add7_4_3 = tf.add(conv_1x1_3, add7_4_up,name='new_add7_4_3')
    #
    # add7_4_3image = tf.layers.conv2d_transpose(add7_4_3, num_classes, 16, padding='same', strides=(8, 8), name='new_add7_4_3image',
    #                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-2))
    #
    # vgg_layer7_out = tf.stop_gradient(vgg_layer7_out)
    # vgg_layer4_out = tf.stop_gradient(vgg_layer4_out)
    # vgg_layer3_out = tf.stop_gradient(vgg_layer3_out)

    with tf.variable_scope('FCN'):
        conv_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, strides=(1, 1), padding='same',
                                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(.01),
                                    name="conv_1x1")
        # tf.Print(conv_1x1, [tf.shape(conv_1x1)[:]])

        upsample1 = tf.layers.conv2d_transpose(conv_1x1, num_classes, 4, strides=(2, 2), padding='same',
                                               kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                               kernel_regularizer=tf.contrib.layers.l2_regularizer(.01),
                                               name="upsample1")
        # tf.Print(upsample1, [tf.shape(upsample1)[:]])

        vgg_layer4_out = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, strides=(1, 1), padding='same',
                                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                          kernel_regularizer=tf.contrib.layers.l2_regularizer(.01),
                                          name="vgg_layer4_out")
        skip1 = tf.add(upsample1, vgg_layer4_out, name="skip1")
        # tf.Print(skip1, [tf.shape(skip1)[:]])

        upsample2 = tf.layers.conv2d_transpose(skip1, num_classes, 4, strides=(2, 2), padding='same',
                                               kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                               kernel_regularizer=tf.contrib.layers.l2_regularizer(.01),
                                               name="upsample2")
        # tf.Print(upsample2, [tf.shape(upsample2)[:]])

        vgg_layer3_out = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, strides=(1, 1), padding='same',
                                          kernel_regularizer=tf.contrib.layers.l2_regularizer(.01),
                                          name="vgg_layer_3out")
        skip2 = tf.add(upsample2, vgg_layer3_out, name="skip2")
        # tf.Print(skip2, [tf.shape(skip2)[:]])

        final = tf.layers.conv2d_transpose(skip2, num_classes, 16, strides=(8, 8), padding='same',
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.001),
                                           kernel_regularizer=tf.contrib.layers.l2_regularizer(.01),
                                           name="final")

        global x
        x = tf.Print(vgg_layer4_out, [tf.shape(vgg_layer4_out)[:]], message="Shape of vgg4 after conv:", summarize=10, first_n=1)

        return final





#tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label = tf.reshape(correct_label, (-1,num_classes))
    tf.Print(correct_label, [tf.shape(correct_label)[:]])
    #correct_label = tf.reshape(correct_label, [-1])
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))
    tf.summary.scalar('cross_entropy', cross_entropy)
    #iou, iou_op = tf.metrics.mean_iou(correct_label, nn_last_layer, 2)
    pred = tf.argmax(nn_last_layer, axis=3)

    probabilities = tf.nn.softmax(nn_last_layer)


    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
    return logits, train_op, cross_entropy,pred,probabilities
#tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate, pred, probabilities):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    merged = tf.summary.merge_all()
    #train_writer = tf.summary.FileWriter('/home/marcelo/CarND-Semantic-Segmentation/train_viz', sess.graph)
    sess.run(tf.global_variables_initializer())

    i = 0
    for epoch in range(epochs):
        rate = np.float(0.0001)
        prob = np.float(0.1)

        for image, label in get_batches_fn(batch_size):

            loss, _, summary ,m = sess.run([cross_entropy_loss, train_op,merged, x],
                      feed_dict={input_image: image, correct_label: label, keep_prob: prob, learning_rate: rate})
            print('epoch:' + str(epoch))
            print('loss:' + str(loss))
            #print(m)
            i+=1
            if (i % 200) == 0:

              f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(20,5))
              ax1.imshow(image[0, :, :, :])
              ax1.set_title('Input image')
                         #plt.imshow(image[0, :, :, :])
                         #plt.imsave(fname= 'im'+str(i) +'-loss:'+str(loss)+'.png', arr=image[0, :, :, :])
              probability_graph = ax2.imshow(label[0, :, :, 0])
              ax2.set_title('Input Ground-Truth Annotation')
                     #plt.imshow(label[0, :, :, 0])
                     #plt.imsave(fname='groundt' + str(i) + '-loss:' + str(loss) + '.png', arr=label[0, :, :, 0])
              #plt.show()
              plt.savefig('gt'+ str(i) +'-loss:'+ str(loss)+'.png', bbox_inches='tight')
             # train_writer.add_summary(summary, i)

              cmap = plt.get_cmap('bwr')

              pred_np, probabilities_np = sess.run([pred, probabilities],
                                                 feed_dict={input_image: image, correct_label: label, keep_prob: prob,
                                                            learning_rate: rate})
              #plt.imshow(np.uint8(pred_np[0, :, :]), vmax=1.5, vmin=-0.4, cmap=cmap)
              #plt.imsave(fname='class' + str(i) + '-loss:' + str(loss) + '.png', arr=np.uint8(pred_np[0, :, :]))
              f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(20,5))
              ax1.imshow(np.uint8(pred_np[0, :, :]), vmax=1.5, vmin=-0.4, cmap=cmap)
              ax1.set_title('Argmax. Iteration # ' + str(i))
              probability_graph = ax2.imshow(probabilities_np.squeeze()[0, :, :, 1])
              ax2.set_title('Probability of the Class. Iteration # ' + str(i))
              plt.imshow(probabilities_np.squeeze()[0, :, :, 1])
              plt.colorbar()
              plt.savefig(fname='prob' + str(i) + '-loss:' + str(loss) + '.png', arr=probabilities_np.squeeze()[0, :, :, 1])

              #fig = plt.figure(figsize=(18, 5))  # Your image (W)idth and (H)eight in inches
              # Stretch image to full figure, removing "grey region"
              #plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
              #im = plt.imshow(probabilities_np.squeeze()[0, :, :, 1])  # Show the image
             # pos = fig.add_axes([0.93, 0.1, 0.02, 0.35])  # Set colorbar position in fig
              #fig.colorbar(im)  # Create the colorbar
              #plt.savefig(fname='prob' + str(i) + '-loss:' + str(loss) + '.png', arr=probabilities_np.squeeze()[0, :, :, 1])

              #plt.savefig('prob' + str(i) + '-loss:' + str(loss) + '.png', bbox_inches='tight')
              #plt.show()
            i = i



#tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:

        # sess.run(tf.global_variables_initializer())
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)
        input, keep, layer3, layer4, layer7 = load_vgg(sess, vgg_path)
        layers_output = layers(layer3, layer4, layer7, num_classes)

        #image_placeholder = tf.placeholder(tf.float32, shape=(None, image_shape[0], image_shape[1], 3))
        label_placeholder = tf.placeholder(tf.bool, shape=(None, image_shape[0], image_shape[1], num_classes))
        #keep_prob = tf.placeholder(tf.float32, name ='keep')
        learning_rate_placeholder = tf.placeholder(tf.float32, name='rate')

        logits, train_op, cross_entropy,pred,probabilities= optimize(layers_output, label_placeholder, 0.001, num_classes)

        train_nn(sess, 100, 10, get_batches_fn, train_op, cross_entropy, input, label_placeholder,
                 keep, learning_rate_placeholder,pred,probabilities)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network


        # TODO: Build NN using load_vgg, layers, and optimize function

        # TODO: Train NN using the train_nn function

        # TODO: Save inference data using helper.save_inference_samples


        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep, input)

        # OPTIONAL: Apply the trained model to a video




if __name__ == '__main__':
    run()
