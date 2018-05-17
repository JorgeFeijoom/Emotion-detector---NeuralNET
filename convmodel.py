# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    o_h = np.zeros(n)
    o_h[x] = 1
    return o_h

# --------------------------------------------------
#
#       DATA SOURCE
#
# --------------------------------------------------
num_classes = 3
batch_size = 4


def dataSource(paths, batch_size):
    min_after_dequeue = 10
    capacity = min_after_dequeue + 3 * batch_size

    example_batch_list = []
    label_batch_list = []

    for i, p in enumerate(paths):
        filename = tf.train.match_filenames_once(p)
        filename_queue = tf.train.string_input_producer(filename, shuffle=False)
        reader = tf.WholeFileReader()

        _, file_image = reader.read(filename_queue)

        image, label = tf.image.decode_jpeg(file_image), one_hot(int(i), num_classes)
        image = tf.image.resize_image_with_crop_or_pad(image, 80, 140)
        image = tf.image.rgb_to_grayscale(image)
        image = tf.reshape(image, [80, 140, 1])
        image = tf.to_float(image) / 256. - 0.5

        example_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=capacity,
                                                            min_after_dequeue=min_after_dequeue)
        example_batch_list.append(example_batch)
        label_batch_list.append(label_batch)

    example_batch = tf.concat(values=example_batch_list, axis=0)
    label_batch = tf.concat(values=label_batch_list, axis=0)

    return example_batch, label_batch

# --------------------------------------------------
#
#       MODEL
#
# --------------------------------------------------


def myModel(X, reuse=False):
    with tf.variable_scope('ConvNet', reuse=reuse):
        o1 = tf.layers.conv2d(inputs=X, filters=32, kernel_size=3, activation=tf.nn.relu)
        o2 = tf.layers.max_pooling2d(inputs=o1, pool_size=2, strides=2)
        o3 = tf.layers.conv2d(inputs=o2, filters=64, kernel_size=3, activation=tf.nn.relu)
        o4 = tf.layers.max_pooling2d(inputs=o3, pool_size=2, strides=2)

        h = tf.layers.dense(inputs=tf.reshape(o4, [batch_size * 3, 33 * 18 * 64]), units=5, activation=tf.nn.relu)
        h2 = tf.layers.dense(inputs=h, units=3, activation=tf.nn.relu)

        y = tf.layers.dense(inputs=h2, units=3, activation=tf.nn.softmax)
    return y


example_batch_train, label_batch_train = dataSource(["train/sad/*.jpg", "train/happy/*.jpg", "train/surprised/*.jpg"],
                                                    batch_size=batch_size)
example_batch_valid, label_batch_valid = dataSource(["valid/sad/*.jpg", "valid/happy/*.jpg", "valid/surprised/*.jpg"],
                                                    batch_size=batch_size)
example_batch_test, label_batch_test = dataSource(["test/sad/*.jpg", "test/happy/*.jpg", "test/surprised/*.jpg"],
                                                    batch_size=batch_size)


example_batch_train_predicted = myModel(example_batch_train, reuse=False)
example_batch_valid_predicted = myModel(example_batch_valid, reuse=True)
example_batch_test_predicted = myModel(example_batch_test, reuse=True)

train_cost = tf.reduce_sum(tf.square(example_batch_train_predicted - tf.cast(label_batch_train, dtype=tf.float32)))
valid_cost = tf.reduce_sum(tf.square(example_batch_valid_predicted - tf.cast(label_batch_valid, dtype=tf.float32)))
test_cost = tf.reduce_sum(tf.square(example_batch_test_predicted - tf.cast(label_batch_test, dtype=tf.float32)))

# cost = tf.reduce_mean(-tf.reduce_sum(label_batch * tf.log(y), reduction_indices=[1]))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.007).minimize(train_cost)


# --------------------------------------------------
#
#       TRAINING
#
# --------------------------------------------------

saver = tf.train.Saver()

with tf.Session() as sess:

    print "---------------------------"
    print " Start training process... "
    print "---------------------------"

    file_writer = tf.summary.FileWriter('./tmp', sess.graph)

    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    error_train = []
    error_valid = []

    for _ in range(300):
        sess.run(optimizer)
        if _ % 20 == 0:
            print "Iter:", _, "|| Error:", sess.run(train_cost)

            error_train.append(sess.run(train_cost))
            error_valid.append(sess.run(valid_cost))

    print "---------------------------------------------"

    # --------------------------------------------------
    #
    #       TESTING
    #
    # --------------------------------------------------

    print "---------------------------"
    print " Start testing process...  "
    print "---------------------------"

    total = 0
    error_count = 0
    correct = 0

    test_data = sess.run(label_batch_test)
    result = sess.run(example_batch_test_predicted)

    for test, result in zip(test_data, result):
        if np.argmax(test) != np.argmax(result):
            error_count += 1
            total += 1
        else:
            correct += 1
            total += 1

    print "Correct elements: ", correct, " | of ", total, " in total.", " |", error_count, " errors."


    save_path = saver.save(sess, "./tmp/model.ckpt")
    print "Model saved in file: %s" % save_path

    print "---------------------------------------------"
    coord.request_stop()
    coord.join(threads)

    plt.figure(1)
    plt.plot(error_train)
    plt.xlabel("Iterations")
    plt.ylabel("Error")
    plt.grid(True)
    plt.show()

