"""
This module holds sources for CIFAR100 dataset. Only `TFSource` type source are
implemented for now.
"""
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
import os

import six.moves.cPickle as pickle
import numpy as np
import tensorflow as tf

from ..core.sources import ClassificationTFSource
from .datasets import DataSet
from six.moves import range


SUPER_CLASS_NUM = 20
SUB_CLASS_NUM = 5
OLD_LABEL_ORDER_LIST = ['apple',
                        'aquarium_fish',
                        'baby',
                        'bear',
                        'beaver',
                        'bed',
                        'bee',
                        'beetle',
                        'bicycle',
                        'bottle',
                        'bowl',
                        'boy',
                        'bridge',
                        'bus',
                        'butterfly',
                        'camel',
                        'can',
                        'castle',
                        'caterpillar',
                        'cattle',
                        'chair',
                        'chimpanzee',
                        'clock',
                        'cloud',
                        'cockroach',
                        'couch',
                        'crab',
                        'crocodile',
                        'cup',
                        'dinosaur',
                        'dolphin',
                        'elephant',
                        'flatfish',
                        'forest',
                        'fox',
                        'girl',
                        'hamster',
                        'house',
                        'kangaroo',
                        'keyboard',
                        'lamp',
                        'lawn_mower',
                        'leopard',
                        'lion',
                        'lizard',
                        'lobster',
                        'man',
                        'maple_tree',
                        'motorcycle',
                        'mountain',
                        'mouse',
                        'mushroom',
                        'oak_tree',
                        'orange',
                        'orchid',
                        'otter',
                        'palm_tree',
                        'pear',
                        'pickup_truck',
                        'pine_tree',
                        'plain',
                        'plate',
                        'poppy',
                        'porcupine',
                        'possum',
                        'rabbit',
                        'raccoon',
                        'ray',
                        'road',
                        'rocket',
                        'rose',
                        'sea',
                        'seal',
                        'shark',
                        'shrew',
                        'skunk',
                        'skyscraper',
                        'snail',
                        'snake',
                        'spider',
                        'squirrel',
                        'streetcar',
                        'sunflower',
                        'sweet_pepper',
                        'table',
                        'tank',
                        'telephone',
                        'television',
                        'tiger',
                        'tractor',
                        'train',
                        'trout',
                        'tulip',
                        'turtle',
                        'wardrobe',
                        'whale',
                        'willow_tree',
                        'wolf',
                        'woman',
                        'worm']

# Five classes in a line is a super class. The order is by the sequences shown
# given in the dataset [website](https://www.cs.toronto.edu/~kriz/cifar.html),
# and is copied in the class comment of `HCifar100TFSource`.
NEW_LABEL_ORDER_LIST = [
    'beaver', 'dolphin', 'otter', 'seal', 'whale',
    'aquarium_fish', 'flatfish', 'ray', 'shark', 'trout',
    'orchid', 'poppy', 'rose', 'sunflower', 'tulip',
    'bottle', 'bowl', 'can', 'cup', 'plate',
    'apple', 'mushroom', 'orange', 'pear', 'sweet_pepper',
    'clock', 'keyboard', 'lamp', 'telephone', 'television',
    'bed', 'chair', 'couch', 'table', 'wardrobe',
    'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach',
    'bear', 'leopard', 'lion', 'tiger', 'wolf',
    'bridge', 'castle', 'house', 'road', 'skyscraper',
    'cloud', 'forest', 'mountain', 'plain', 'sea',
    'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo',
    'fox', 'porcupine', 'possum', 'raccoon', 'skunk',
    'crab', 'lobster', 'snail', 'spider', 'worm',
    'baby', 'boy', 'girl', 'man', 'woman',
    'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle',
    'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel',
    'maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree',
    'bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train',
    'lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor'
]


class Cifar100TFSource(ClassificationTFSource):
    """
    A concrete `Source` for Cifar100 dataset. This class provides global
    contrast normalized, then ZCA whitened images using tfrecords.
    """
    SAMPLE_NUM = 50000

    def _load_cifar100_python(self, filenames):
        """
        Load python version of Cifar100 dataset.
        """
        # Load the first batch of data to get shape info.
        filename = filenames[0]
        with open(filename, "rb") as f:
            tmp = pickle.load(f)
            data = tmp["data"]
            labels = np.array(tmp["fine_labels"])

        # Load the rest.
        for filename in filenames[1:]:
            with open(filename, "rb") as f:
                tmp = pickle.load(f)
                data = np.append(data, tmp["data"], 1)
                labels = np.append(labels, tmp["fine_labels"])

        data = np.reshape(data, [-1, 3, 32, 32])
        data = np.einsum("nchw->nhwc", data)

        return DataSet(data, labels)

    def _forward(self):
        """
        Construct input for CIFAR100 evaluation using the Reader ops.
        """
        self._maybe_convert_to_tf()
        self._read_from_tfrecord()

    def _read_from_tfrecord(self):
        # Read and set up data tensors.
        filename = os.path.join(self.work_dir, 'cifar100_training.tfrecords')
        with tf.name_scope('input'):
            filename_queue = tf.train.string_input_producer([filename])
        self._training_datum, self._training_label \
            = self._get_sample_tensors_from_tfrecords(filename_queue)

        filename = os.path.join(self.work_dir, 'cifar100_test.tfrecords')
        with tf.name_scope('input'):
            filename_queue = tf.train.string_input_producer([filename])
        self._val_datum, self._val_label \
            = self._get_sample_tensors_from_tfrecords(filename_queue)

    def _maybe_convert_to_tf(self):
        """
        If tfrecords data are not available, convert it from the ZCA whitened
        data and downloaded labels.

        TODO: write the data pre-processing code so I won't need to rely on
        pylearn2.
        """
        TRAINING_TF_FILENAME = "cifar100_training"
        if not os.path.exists(
                os.path.join(self.work_dir,
                             TRAINING_TF_FILENAME + ".tfrecords")):
            # Read the numpy data in and convert it to TFRecord.
            imgs = np.load(os.path.join(self.work_dir,
                                        "pylearn2_gcn_whitened",
                                        "train.npy"))
            imgs = imgs.reshape([Cifar100TFSource.SAMPLE_NUM, 3, 32, 32])
            imgs = np.einsum("nchw->nhwc", imgs)

            # Load training set into memory.
            train_filenames = [os.path.join(self.work_dir,
                                            'cifar-100-python',
                                            "train")]
            training_labels = self._load_cifar100_python(
                train_filenames).labels
            self._convert_to_tf(imgs, training_labels, TRAINING_TF_FILENAME)

        TEST_TF_FILENAME = "cifar100_test"
        if not os.path.exists(
                os.path.join(self.work_dir, TEST_TF_FILENAME + ".tfrecords")):
            imgs = np.load(os.path.join(self.work_dir,
                                        "pylearn2_gcn_whitened",
                                        "test.npy"))
            imgs = imgs.reshape([10000, 3, 32, 32])
            imgs = np.einsum("nchw->nhwc", imgs)
            test_filenames = [os.path.join(self.work_dir,
                                           'cifar-100-python',
                                           'test')]
            test_labels = self._load_cifar100_python(test_filenames).labels
            self._convert_to_tf(imgs, test_labels, TEST_TF_FILENAME)

    def _get_sample_tensors_from_tfrecords(self, filename_queue):
        """
        Read from tfrecord file and return data tensors.

        Args:
            filename_queue: tf.train.string_input_producer
                A file name queue that gives string tensor for tfrecord names.

        Returns:
            (image, label): tuple of (rank-4 tf.float32 tensor and rank-1
                            tf.int32 tensor)
                individual sample that may be later put into a batch.
        """
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            # Defaults are not specified since both keys are required.
            features={
                'image_raw': tf.FixedLenFeature(
                    [32, 32, 3],
                    tf.float32),
                'label': tf.FixedLenFeature([], tf.int64),
            })

        # Convert label from a scalar uint8 tensor to an int32 scalar.
        label = tf.cast(features['label'], tf.int32)
        image = features["image_raw"]

        return image, label


class HCifar100TFSource(ClassificationTFSource):
    """
    A concrete `Source` for Cifar100 dataset to provide both fine labels and
    coarse labels. It remaps fine labels to let classes in a super class has
    consecutive labels. This class provides global contrast normalized, then
    ZCA whitened images using tfrecords.

    Label property of this source will point to a list of three labels, being
    coarse label, fine label and expanded fine label respectively.

    The logic to remap fine labels is as following:

    Alphabetically, coarse labels are assigned to super class (which is
    already done in the original dataset). Starting from super class 0,
    sub-class in this super class is assigned label alphabetically.

    +------------------------------+------------------------------+
    |Superclass                    |Classes                       |
    +------------------------------+------------------------------+
    |aquatic mammals               |beaver, dolphin,  otter, seal,|
    |                              |whale                         |
    +------------------------------+------------------------------+
    |fish                          |aquarium fish,  flatfish, ray,|
    |                              |shark, trout                  |
    +------------------------------+------------------------------+
    |flowers                       |orchids,    poppies,    roses,|
    |                              |sunflowers, tulips            |
    +------------------------------+------------------------------+
    |food containers               |bottles,  bowls,  cans,  cups,|
    |                              |plates                        |
    +------------------------------+------------------------------+
    |fruit and vegetables          |apples,   mushrooms,  oranges,|
    |                              |pears, sweet peppers          |
    +------------------------------+------------------------------+
    |household electrical devices  |clock,    computer   keyboard,|
    |                              |lamp, telephone, television   |
    +------------------------------+------------------------------+
    |household furniture           |bed,   chair,  couch,   table,|
    |                              |wardrobe                      |
    +------------------------------+------------------------------+
    |insects                       |bee,     beetle,    butterfly,|
    |                              |caterpillar, cockroach        |
    +------------------------------+------------------------------+
    |large carnivores              |bear,  leopard,  lion,  tiger,|
    |                              |wolf                          |
    +------------------------------+------------------------------+
    |large man-made outdoor things |bridge,  castle, house,  road,|
    |                              |skyscraper                    |
    +------------------------------+------------------------------+
    |large natural outdoor scenes  |cloud,    forest,    mountain,|
    |                              |plain, sea                    |
    +------------------------------+------------------------------+
    |large omnivores and herbivores|camel,   cattle,   chimpanzee,|
    |                              |elephant, kangaroo            |
    +------------------------------+------------------------------+
    |medium-sized mammals          |fox,     porcupine,    possum,|
    |                              |raccoon, skunk                |
    +------------------------------+------------------------------+
    |non-insect invertebrates      |crab, lobster,  snail, spider,|
    |                              |worm                          |
    +------------------------------+------------------------------+
    |people                        |baby, boy, girl, man, woman   |
    +------------------------------+------------------------------+
    |reptiles                      |crocodile,  dinosaur,  lizard,|
    |                              |snake, turtle                 |
    +------------------------------+------------------------------+
    |small mammals                 |hamster, mouse, rabbit, shrew,|
    |                              |squirrel                      |
    +------------------------------+------------------------------+
    |trees                         |maple, oak, palm, pine, willow|
    +------------------------------+------------------------------+
    |vehicles 1                    |bicycle,    bus,   motorcycle,|
    |                              |pickup truck, train           |
    +------------------------------+------------------------------+
    |vehicles 2                    |lawn-mower, rocket, streetcar,|
    |                              |tank, tractor                 |
    +------------------------------+------------------------------+
    """
    SAMPLE_NUM = 50000
    TRAINING_TF_FILENAME = "hierarchical_cifar100_training.tfrecords"
    TEST_TF_FILENAME = "hierarchical_cifar100_test.tfrecords"

    def __init__(self, **kwargs):
        super(HCifar100TFSource, self).__init__(**kwargs)
        # Construct remapping table.
        remap_list = [-1] * 100
        for i, label in enumerate(OLD_LABEL_ORDER_LIST):
            for j, new_label in enumerate(NEW_LABEL_ORDER_LIST):
                if new_label == label:
                    remap_list[i] = j

        for l in remap_list:
            assert l is not -1, "All label should have a mapped value."

        self.remap_list = remap_list

    def _load_cifar100_python(self, filenames):
        """
        Load python version of Cifar100 dataset.
        """
        # Load the first batch of data to get shape info.
        filename = filenames[0]
        with open(filename, "rb") as f:
            tmp = pickle.load(f)
            fine_labels = np.array(tmp["fine_labels"])
            coarse_labels = np.array(tmp["coarse_labels"])

        # Load the rest.
        for filename in filenames[1:]:
            with open(filename, "rb") as f:
                tmp = pickle.load(f)
                fine_labels = np.append(fine_labels, tmp["fine_labels"])
                coarse_labels = np.append(coarse_labels, tmp["coarse_labels"])

        return fine_labels, coarse_labels

    def _forward(self):
        """
        Construct input for CIFAR100 evaluation using the Reader ops.
        """
        self._maybe_convert_to_tf()
        self._read_from_tfrecord()

    def _read_from_tfrecord(self):
        # Read and set up data tensors.
        filename = os.path.join(self.work_dir,
                                HCifar100TFSource.TRAINING_TF_FILENAME)
        with tf.name_scope('input'):
            filename_queue = tf.train.string_input_producer([filename])
        self._training_datum, self._training_label \
            = self._get_sample_tensors_from_tfrecords(filename_queue)

        filename = os.path.join(self.work_dir,
                                HCifar100TFSource.TEST_TF_FILENAME)
        with tf.name_scope('input'):
            filename_queue = tf.train.string_input_producer([filename])
        self._val_datum, self._val_label \
            = self._get_sample_tensors_from_tfrecords(filename_queue)

    def _maybe_convert_to_tf(self):
        """
        If tfrecords data are not available, convert it from the ZCA whitened
        data and downloaded labels.

        TODO: write the data pre-processing code so I won't need to rely on
        pylearn2.
        """
        if not os.path.exists(
                os.path.join(
                    self.work_dir,
                    HCifar100TFSource.TRAINING_TF_FILENAME)):
            # Read the numpy data in and convert it to TFRecord.
            imgs = np.load(os.path.join(self.work_dir,
                                        "pylearn2_gcn_whitened",
                                        "train.npy"))
            imgs = imgs.reshape([Cifar100TFSource.SAMPLE_NUM, 3, 32, 32])
            imgs = np.einsum("nchw->nhwc", imgs)

            # Load training set into memory.
            train_filenames = [os.path.join(self.work_dir,
                                            'cifar-100-python',
                                            "train")]
            training_fine_labels, training_coarse_labels \
                = self._load_cifar100_python(train_filenames)
            self._remap_fine_labels(training_fine_labels)
            self._convert_to_tf(imgs,
                                training_fine_labels,
                                training_coarse_labels,
                                HCifar100TFSource.TRAINING_TF_FILENAME)

        if not os.path.exists(
                os.path.join(
                    self.work_dir,
                    HCifar100TFSource.TEST_TF_FILENAME)):
            imgs = np.load(os.path.join(self.work_dir,
                                        "pylearn2_gcn_whitened",
                                        "test.npy"))
            imgs = imgs.reshape([10000, 3, 32, 32])
            imgs = np.einsum("nchw->nhwc", imgs)
            test_filenames = [os.path.join(self.work_dir,
                                           'cifar-100-python',
                                           'test')]

            test_fine_labels, test_coarse_labels \
                = self._load_cifar100_python(test_filenames)
            self._remap_fine_labels(test_fine_labels)
            self._convert_to_tf(imgs,
                                test_fine_labels,
                                test_coarse_labels,
                                HCifar100TFSource.TEST_TF_FILENAME)

    def _remap_fine_labels(self, labels):
        """
        Remap labels of sub-class in place.

        Args:
            labels: numpy array of int of shape [N].
        """
        for i, l in enumerate(labels):
            labels[i] = self.remap_list[l]

    def _convert_to_tf(self, images, fine_labels, coarse_labels, name):
        """
        Convert images and labels to tfrecord format.
        """
        num_examples = fine_labels.shape[0]
        if images.shape[0] != num_examples:
            raise ValueError("Images size %d does not match fine label size"
                             "%d." % (images.shape[0], num_examples))
        if coarse_labels.shape[0] != num_examples:
            raise ValueError("Coarse label size %d does not match fine label"
                             " size %d" % (images.shape[0], num_examples))

        expanded_fine_labels = self._expand_fine_labels(fine_labels)

        row = images.shape[1]
        col = images.shape[2]
        depth = images.shape[3]

        filename = os.path.join(self.work_dir, name)
        print(('Writing', filename))
        writer = tf.python_io.TFRecordWriter(filename)
        for index in range(num_examples):
            image_raw = np.reshape(images[index], -1).tolist()
            expanded_fine_label = expanded_fine_labels[index].tolist()
            example = tf.train.Example(features=tf.train.Features(
                feature={
                    'height': self._int_feature([row]),
                    'width': self._int_feature([col]),
                    'depth': self._int_feature([depth]),
                    'coarse_label': self._int_feature(
                        [int(coarse_labels[index])]),
                    'expanded_fine_label': self._int_feature(
                        expanded_fine_label),
                    'fine_label': self._int_feature(
                        [int(fine_labels[index])]),
                    'image_raw': self._float_feature(image_raw)}))
            writer.write(example.SerializeToString())
        writer.close()

    def _expand_fine_labels(self, labels):
        """
        A fine label will be converted to a `(SUB_CLASS_NUM+1) *
        SUPER_CLASS_NUM` vector, where in the super class group the sub-class
        belongs to, the corresponding dimension will be 1, and in other super
        class groups, the non-existence dimension will be 1. In all, in total,
        there will be super-class number of 1 in the label vector whatever.
        """
        num_examples = labels.shape[0]
        fine_label_vectors = np.zeros(
            [num_examples, (SUB_CLASS_NUM+1)*SUPER_CLASS_NUM],
            dtype=np.int)
        # Set default non-existence labels.
        fine_label_vectors[:, SUB_CLASS_NUM::(SUB_CLASS_NUM+1)] = 1
        # Set real labels.
        batch_indices = list(range(0, num_examples))
        augmented_labels = labels // SUB_CLASS_NUM * (SUB_CLASS_NUM+1) \
            + labels % SUB_CLASS_NUM
        fine_label_vectors[batch_indices, augmented_labels] = 1
        # Unset non-existence labels corresponding to real labels.
        labels_to_unset = labels // SUB_CLASS_NUM * (SUB_CLASS_NUM+1) + \
            SUB_CLASS_NUM
        fine_label_vectors[batch_indices, labels_to_unset] = 0

        return fine_label_vectors

    def _get_sample_tensors_from_tfrecords(self, filename_queue):
        """
        Read from tfrecord file and return data tensors.

        Args:
            filename_queue: tf.train.string_input_producer
                A file name queue that gives string tensor for tfrecord names.

        Returns:
            A tuple of a tensor and a list. The tensor is the image, the list
            contains tensors of coarse label, fine label and expanded fine
            labels respectively.
        """
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            # Defaults are not specified since both keys are required.
            features={
                'image_raw': tf.FixedLenFeature(
                    [32, 32, 3],
                    tf.float32),
                'coarse_label': tf.FixedLenFeature([], tf.int64),
                'fine_label': tf.FixedLenFeature([], tf.int64),
                'expanded_fine_label': tf.FixedLenFeature(
                    [(SUB_CLASS_NUM+1)*SUPER_CLASS_NUM],
                    tf.int64)
            })

        # Convert label from a scalar uint8 tensor to an int32 scalar.
        coarse_label = tf.cast(features['coarse_label'], tf.int32)
        fine_label = tf.cast(features['fine_label'], tf.int32)
        expanded_fine_label = tf.cast(
            features['expanded_fine_label'], tf.int32)
        image = features["image_raw"]

        return image, [coarse_label, fine_label, expanded_fine_label]
