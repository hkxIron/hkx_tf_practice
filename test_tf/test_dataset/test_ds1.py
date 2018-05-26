import tensorflow as tf
import numpy as np

sess = tf.Session()

def t1():
    dataset = tf.data.Dataset.range(100)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    with tf.Session() as sess:
        for i in range(10):
            value = sess.run(next_element)
            print(value)


def t2():
    max_value = tf.placeholder(tf.int64, shape=[])
    dataset = tf.data.Dataset.range(max_value)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    # Initialize an iterator over a dataset with 10 elements.
    sess.run(iterator.initializer, feed_dict={max_value: 10})
    for i in range(10):
        value = sess.run(next_element)
        # print(value)
        # assert i == value

    # Initialize the same iterator over a dataset with 100 elements.
    MAX = 100000
    sess.run(iterator.initializer, feed_dict={max_value: MAX})
    import time
    start = time.time()
    for i in range(MAX):
        value = sess.run(next_element)
        # print(value)
    end = time.time()
    print("speed:", (MAX + 0.0) / (end - start))


def t3():
    # Define training and validation datasets with the same structure.
    training_dataset = tf.data.Dataset.range(100).map(
        lambda x: x + tf.random_uniform([], -10, 10, tf.int64))
    validation_dataset = tf.data.Dataset.range(50)

    # A reinitializable iterator is defined by its structure. We could use the
    # `output_types` and `output_shapes` properties of either `training_dataset`
    # or `validation_dataset` here, because they are compatible.
    iterator = tf.data.Iterator.from_structure(training_dataset.output_types,
                                               training_dataset.output_shapes)
    next_element = iterator.get_next()

    training_init_op = iterator.make_initializer(training_dataset)
    validation_init_op = iterator.make_initializer(validation_dataset)

    # Run 20 epochs in which the training dataset is traversed, followed by the
    # validation dataset.
    for _ in range(20):
        # Initialize an iterator over the training dataset.
        sess.run(training_init_op)
        for _ in range(10):
            print(sess.run(next_element))

        # Initialize an iterator over the validation dataset.
        sess.run(validation_init_op)
        for _ in range(50):
            print(sess.run(next_element))


def t4():
    # Define training and validation datasets with the same structure.
    training_dataset = tf.data.Dataset.range(100).map(
        lambda x: x + tf.random_uniform([], -10, 10, tf.int64)).repeat()
    validation_dataset = tf.data.Dataset.range(50)

    # A feedable iterator is defined by a handle placeholder and its structure. We
    # could use the `output_types` and `output_shapes` properties of either
    # `training_dataset` or `validation_dataset` here, because they have
    # identical structure.
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(
        handle, training_dataset.output_types, training_dataset.output_shapes)
    next_element = iterator.get_next()

    # You can use feedable iterators with a variety of different kinds of iterator
    # (such as one-shot and initializable iterators).
    training_iterator = training_dataset.make_one_shot_iterator()
    validation_iterator = validation_dataset.make_initializable_iterator()

    # The `Iterator.string_handle()` method returns a tensor that can be evaluated
    # and used to feed the `handle` placeholder.
    training_handle = sess.run(training_iterator.string_handle())
    validation_handle = sess.run(validation_iterator.string_handle())

    # Loop forever, alternating between training and validation.
    while True:
        # Run 200 steps using the training dataset. Note that the training dataset is
        # infinite, and we resume from where we left off in the previous `while` loop
        # iteration.
        for _ in range(200):
            sess.run(next_element, feed_dict={handle: training_handle})

        # Run one pass over the validation dataset.
        sess.run(validation_iterator.initializer)
        for _ in range(50):
            sess.run(next_element, feed_dict={handle: validation_handle})


def t5():
    dataset = tf.data.Dataset.range(5)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    # Typically `result` will be the output of a model, or an optimizer's
    # training operation.
    result = tf.add(next_element, next_element)  # 将两个元素相加
    sess.run(iterator.initializer)
    print(sess.run(result))  # ==> "0"
    print(sess.run(result))  # ==> "2"
    print(sess.run(result))  # ==> "4"
    print(sess.run(result))  # ==> "6"
    print(sess.run(result))  # ==> "8"
    try:
        sess.run(result)
    except tf.errors.OutOfRangeError:
        print("End of dataset")  # ==> "End of dataset"


def t6():
    dataset1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))
    dataset2 = tf.data.Dataset.from_tensor_slices((tf.random_uniform([4]), tf.random_uniform([4, 100])))
    dataset3 = tf.data.Dataset.zip((dataset1, dataset2))
    iterator = dataset3.make_initializable_iterator()
    sess.run(iterator.initializer)
    next1, (next2, next3) = iterator.get_next()
    print(sess.run([next1, next2, next3]))


def t7():
    # Define training and validation datasets with the same structure.
    training_dataset = tf.data.Dataset.range(100).map(
        lambda x: x + tf.random_uniform([], -10, 10, tf.int64)).repeat()
    validation_dataset = tf.data.Dataset.range(50)

    # A feedable iterator is defined by a handle placeholder and its structure. We
    # could use the `output_types` and `output_shapes` properties of either
    # `training_dataset` or `validation_dataset` here, because they have
    # identical structure.
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(
        handle, training_dataset.output_types, training_dataset.output_shapes)
    next_element = iterator.get_next()

    # You can use feedable iterators with a variety of different kinds of iterator
    # (such as one-shot and initializable iterators).
    training_iterator = training_dataset.make_one_shot_iterator()
    validation_iterator = validation_dataset.make_initializable_iterator()

    # The `Iterator.string_handle()` method returns a tensor that can be evaluated
    # and used to feed the `handle` placeholder.
    training_handle = sess.run(training_iterator.string_handle())
    validation_handle = sess.run(validation_iterator.string_handle())

    # Loop forever, alternating between training and validation.
    while True:
        # Run 200 steps using the training dataset. Note that the training dataset is
        # infinite, and we resume from where we left off in the previous `while` loop
        # iteration.
        for _ in range(200):
            sess.run(next_element, feed_dict={handle: training_handle})

        # Run one pass over the validation dataset.
        sess.run(validation_iterator.initializer)
        for _ in range(50):
            sess.run(next_element, feed_dict={handle: validation_handle})


def t8():
    # Load the training data into two NumPy arrays, for example using `np.load()`.
    with np.load("/var/data/training_data.npy") as data:
        features = data["features"]
        labels = data["labels"]

    # Assume that each row of `features` corresponds to the same row as `labels`.
    assert features.shape[0] == labels.shape[0]
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))


def t9():
    # Load the training data into two NumPy arrays, for example using `np.load()`.
    with np.load("/var/data/training_data.npy") as data:
        features = data["features"]
        labels = data["labels"]
    # Assume that each row of `features` corresponds to the same row as `labels`.
    assert features.shape[0] == labels.shape[0]

    features_placeholder = tf.placeholder(features.dtype, features.shape)
    labels_placeholder = tf.placeholder(labels.dtype, labels.shape)

    dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
    # [Other transformations on `dataset`...]
    # dataset = ...
    iterator = dataset.make_initializable_iterator()
    sess.run(iterator.initializer, feed_dict={features_placeholder: features,
                                              labels_placeholder: labels})


def t10():
    filenames = tf.placeholder(tf.string, shape=[None])
    dataset = tf.data.TFRecordDataset(filenames)
    # dataset = dataset.map(...)  # Parse the record into tensors.
    dataset = dataset.repeat()  # Repeat the input indefinitely.
    dataset = dataset.batch(32)
    iterator = dataset.make_initializable_iterator()

    # You can feed the initializer with the appropriate filenames for the current
    # phase of execution, e.g. training vs. validation.

    # Initialize `iterator` with training data.
    training_filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
    sess.run(iterator.initializer, feed_dict={filenames: training_filenames})

    # Initialize `iterator` with validation data.
    validation_filenames = ["/var/data/validation1.tfrecord", ...]
    sess.run(iterator.initializer, feed_dict={filenames: validation_filenames})


def t11():
    filenames = ["/var/data/file1.txt", "/var/data/file2.txt"]

    dataset = tf.data.Dataset.from_tensor_slices(filenames)

    # Use `Dataset.flat_map()` to transform each file as a separate nested dataset,
    # and then concatenate their contents sequentially into a single "flat" dataset.
    # * Skip the first line (header row).
    # * Filter out lines beginning with "#" (comments).
    dataset = dataset.flat_map(
        lambda filename: (
            tf.data
                .TextLineDataset(filename)
                .skip(1)
                .filter(lambda line: tf.not_equal(tf.substr(line, 0, 1), "#")))
    )

def t12():
    # Transforms a scalar string `example_proto` into a pair of a scalar string and
    # a scalar integer, representing an image and its label, respectively.
    def _parse_function(example_proto):
        features = {"image": tf.FixedLenFeature((), tf.string, default_value=""),
                    "label": tf.FixedLenFeature((), tf.int32, default_value=0)}
        parsed_features = tf.parse_single_example(example_proto, features)
        return parsed_features["image"], parsed_features["label"]

    # Creates a dataset that reads all of the examples from two files, and extracts
    # the image and label features.
    filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(_parse_function)

def t13():
    # Reads an image from a file, decodes it into a dense tensor, and resizes it
    # to a fixed shape.
    def _parse_function(filename, label):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_image(image_string)
        image_resized = tf.image.resize_images(image_decoded, [28, 28])
        return image_resized, label

    # A vector of filenames.
    filenames = tf.constant(["/var/data/image1.jpg", "/var/data/image2.jpg", ...])

    # `labels[i]` is the label for the image in `filenames[i].
    labels = tf.constant([0, 37, ...])

    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(_parse_function)

def use_tf_py_func():
    import cv2
    # Use a custom OpenCV function to read the image, instead of the standard
    # TensorFlow `tf.read_file()` operation.
    def _read_py_function(filename, label):
        image_decoded = cv2.imread(image_string, cv2.IMREAD_GRAYSCALE)
        return image_decoded, label

    # Use standard TensorFlow operations to resize the image to a fixed shape.
    def _resize_function(image_decoded, label):
        image_decoded.set_shape([None, None, None])
        image_resized = tf.image.resize_images(image_decoded, [28, 28])
        return image_resized, label

    filenames = ["/var/data/image1.jpg", "/var/data/image2.jpg", ...]
    labels = [0, 37, 29, 1, ...]

    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(
        lambda filename, label: tuple(tf.py_func(
            _read_py_function, [filename, label], [tf.uint8, label.dtype]))) # py_func的作用是延迟python function的执行
    dataset = dataset.map(_resize_function)

def test_batch():
    inc_dataset = tf.data.Dataset.range(100)
    dec_dataset = tf.data.Dataset.range(0, -100, -1)
    dataset = tf.data.Dataset.zip((inc_dataset, dec_dataset))
    batched_dataset = dataset.batch(4)

    iterator = batched_dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    print(sess.run(next_element))  # ==> ([0, 1, 2,   3],   [ 0, -1,  -2,  -3])
    print(sess.run(next_element))  # ==> ([4, 5, 6,   7],   [-4, -5,  -6,  -7])
    print(sess.run(next_element))  # ==> ([8, 9, 10, 11],   [-8, -9, -10, -11])

def test_padded_batch():
    dataset = tf.data.Dataset.range(100)
    dataset = dataset.map(lambda x: tf.fill([tf.cast(x, tf.int32)], x))
    dataset = dataset.padded_batch(4, padded_shapes=[None])

    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    print(sess.run(next_element))  # ==> [[0, 0, 0], [1, 0, 0], [2, 2, 0], [3, 3, 3]]
    print("\n")
    print(sess.run(next_element))  # ==> [[4, 4, 4, 4, 0, 0, 0],
                                    #      [5, 5, 5, 5, 5, 0, 0],
                                    #      [6, 6, 6, 6, 6, 6, 0],
                                    #      [7, 7, 7, 7, 7, 7, 7]]


def input_fn():
    def dataset_input_fn():
        filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
        dataset = tf.data.TFRecordDataset(filenames)

        # Use `tf.parse_single_example()` to extract data from a `tf.Example`
        # protocol buffer, and perform any additional per-record preprocessing.
        def parser(record):
            keys_to_features = {
                "image_data": tf.FixedLenFeature((), tf.string, default_value=""),
                "date_time": tf.FixedLenFeature((), tf.int64, default_value=""),
                "label": tf.FixedLenFeature((), tf.int64,
                                            default_value=tf.zeros([], dtype=tf.int64)),
            }
            parsed = tf.parse_single_example(record, keys_to_features)

            # Perform additional preprocessing on the parsed data.
            image = tf.image.decode_jpeg(parsed["image_data"])
            image = tf.reshape(image, [299, 299, 1])
            label = tf.cast(parsed["label"], tf.int32)
            return {"image_data": image, "date_time": parsed["date_time"]}, label

        # Use `Dataset.map()` to build a pair of a feature dictionary and a label
        # tensor for each example.
        num_epochs =1
        dataset = dataset.map(parser)
        dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.batch(32)
        dataset = dataset.repeat(num_epochs)
        iterator = dataset.make_one_shot_iterator()

        # `features` is a dictionary in which each value is a batch of values for
        # that feature; `labels` is a batch of labels.
        features, labels = iterator.get_next()
        return features, labels

def t14():
    dataset = tf.data.Dataset.range(100).map(lambda x: x+1).batch(10)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    # Typically `result` will be the output of a model, or an optimizer's
    # training operation.
    sess.run(iterator.initializer)
    while True:
        try:
            print(sess.run(next_element))  # ==> "0"
        except tf.errors.OutOfRangeError:
            print("End of dataset")  # ==> "End of dataset"
            break

if __name__ == "__main__":
    # t3()
    # t4()
    # t5()
    # t6()
    #t7()
    #test_batch()
    #test_padded_batch()
    t14()
    sess.close()
