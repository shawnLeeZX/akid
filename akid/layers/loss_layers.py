import inspect

import tensorflow as tf

from ..core.blocks import ProcessingLayer
from .activation_layers import GroupSoftmaxLayer
from .. import backend as A


class LossLayer(ProcessingLayer):
    """
    An abstract top level loss layer.

    It is nothing here now. However, it has already been used to check whether
    a layer is a loss layer or not.
    """
    def __init__(self, multiplier=1, **kwargs):
        super(LossLayer, self).__init__(**kwargs)
        self.multiplier = multiplier

    def _setup(self):
        self.log("Using multiplier {}".format(self.multiplier))

    def _post_forward(self, *args, **kwargs):
        super(LossLayer, self)._post_forward(self, *args, **kwargs)
        self._loss = self._loss * self.multiplier


class SoftmaxWithLossLayer(LossLayer):
    """
    This layer supports two types of labels --- scalar label, or dense vector
    labels. Passing either one type of them will work out of the box.
    """
    def __init__(self, class_num, **kwargs):
        super(SoftmaxWithLossLayer, self).__init__(**kwargs)
        self.class_num = class_num

    def _forward(self, data_in):
        logits = data_in[0]
        labels = data_in[1]

        self._loss = A.nn.cross_entropy_loss(logits, labels, name="xentropy_mean")
        self._eval = A.nn.class_acccuracy(logits, labels, name='accuracy')


class GroupSoftmaxWithLossLayer(SoftmaxWithLossLayer, GroupSoftmaxLayer):
    """
    A loss layer that makes use of the group structure in the target classes.
    Group structure here means hidden units in a group is mutually exclusive
    classes, for instance, sub-classes in a super class, not the mathematical
    group, yet.

    The input labels should be processed as a multi-hot vector augmented for
    non-existence labels in groups (it is not convenient to fine granularity
    processing of labels with tensors). For example, suppose overall there are
    4 classes, each consecutive 2 classes make up a group. A label 3 (starting
    with 0) corresponds to a vector `100001`, where the first 2 stands for
    non-existence of the first super class (so are two sub-classes in the
    group).
    """
    NAME = "GSMax_Loss"

    def __init__(self, augment_label=False, **kwargs):
        """
        Args:
            augment_label: bool
                Augment label with non-existence label or not.
        """
        super(GroupSoftmaxWithLossLayer, self).__init__(**kwargs)
        self.augment_label = augment_label

    def _forward(self, data_in):
        input = data_in[0]
        labels = data_in[1]
        label_vectors = data_in[2]

        # Compute group softmax.
        shape = input.get_shape().as_list()
        assert len(shape) is 2, "Input should be rank 2."
        if shape[-1] % self.group_size is not 0:
            raise Exception("Group size {} should evenly divide output channel"
                      " number {}".format(self.group_size, shape[-1]))

        out_channel_num = shape[-1]
        num_split = out_channel_num // self.group_size
        self.log("Feature maps of layer {} is divided into {} group".format(
            self.name, num_split))
        data = tf.reshape(input, [-1, shape[-1]])
        if self.use_temperature:
            T = self._get_variable(
                "T",
                [1],
                initializer=tf.constant_initializer(10.0))
            data /= T
        data_split = tf.split(axis=1, num_or_size_splits=num_split, value=data)
        data_split = list(data_split)
        # Augment each split with a constant 1.
        # Get dim of non-channel shape
        nc_shape = 1
        for d in shape[0:-1]:
            nc_shape *= d
        self.ground_state = tf.constant(1.0, shape=[nc_shape, 1])
        for i in xrange(0, len(data_split)):
            data_split[i] = tf.concat(axis=1,
                                      values=[data_split[i],
                                       self.ground_state])
        for i in xrange(0, len(data_split)):
            data_split[i] = tf.nn.softmax(
                data_split[i])

        if self.augment_label:
            # All labels eval graph.
            # Note we need access to hidden units before augmented dimensions
            # for non-existence have been dropped, so this eval graph
            # constructor has not been put in the end.
            group_label_vectors = tf.split(axis=1, num_or_size_splits=num_split, value=label_vectors)
            group_label_vectors = list(group_label_vectors)

            group_label_argmax_idx = []
            for group_label_vector in group_label_vectors:
                group_label_argmax_idx.append(tf.argmax(group_label_vector, 1))
            label_argmax_idx = tf.stack(group_label_argmax_idx)

            group_data_argmax_idx = []
            for group_data_vector in data_split:
                group_data_argmax_idx.append(tf.argmax(group_data_vector, 1))
            data_argmax_idx = tf.stack(group_data_argmax_idx)

            all_label_eval = tf.reduce_mean(
                tf.cast(tf.equal(data_argmax_idx, label_argmax_idx),
                        tf.float32),
                name="augmented_acc")

            # Compute cross entropy with labels.
            # I could choose to split label vector and do cross entropy one by
            # one, or merge the splitted probability vectors and do cross
            # entropy once. The latter was chosen.
            data = tf.concat(axis=1, values=data_split,)
            aug_shape = list(shape)
            aug_shape[-1] = (self.group_size + 1) * num_split
            logits = tf.reshape(data,
                                aug_shape,
                                GroupSoftmaxWithLossLayer.NAME)
            _ = - tf.cast(label_vectors, tf.float32) * tf.log(logits)
            cross_entropy_mean = tf.reduce_mean(_, name='xentropy_mean')

        # Drop the augmented dimension.
        for i in xrange(0, len(data_split)):
            data_split[i] = data_split[i][:, 0:self.group_size]
        data = tf.concat(axis=1, values=data_split,)
        output = tf.reshape(data, shape, GroupSoftmaxWithLossLayer.NAME)

        self._data = output

        if not self.augment_label:
            batch_size = tf.size(labels)
            _labels = tf.expand_dims(labels, 1)
            indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
            concated = tf.concat(axis=1, values=[indices, _labels])
            onehot_labels = tf.sparse_to_dense(
                concated, tf.stack([batch_size, self.class_num]), 1.0, 0.0)
            cross_entropy \
                = tf.nn.softmax_cross_entropy_with_logits(logits=output,
                                                          labels=onehot_labels,
                                                          name='xentropy')
            cross_entropy_mean = tf.reduce_mean(cross_entropy,
                                                name='xentropy_mean')

        self._loss = cross_entropy_mean

        # Real label eval graph.
        # We check the dimension of the real label has the largest value
        # (probability).
        correct = tf.nn.in_top_k(output, labels, 1)
        # Return the number of true entries.
        real_label_eval = tf.reduce_mean(
            tf.cast(correct, tf.float32),
            name="real_acc")

        if self.augment_label:
            # Since we have augmented labels, two eval graph have been created,
            # one for the real label, and one for real and augmented labels. A
            # list will be saved to `_eval`.
            self._eval = [all_label_eval, real_label_eval]
        else:
            self._eval = real_label_eval


class MSELossLayer(LossLayer):
    NAME = "MSELoss"

    def __init__(self, size_average=True, **kwargs):
        super(MSELossLayer, self).__init__(**kwargs)

        self.size_average = size_average

    def _forward(self, data):
        self._loss = A.nn.mse_loss(data[0], data[1], self.size_average)
        self._data = self._loss
        return self._loss


class BCELossLayer(LossLayer):
    NAME = "BCELoss"

    def _forward(self, x):
        self._loss = A.nn.binary_cross_entropy_loss_with_logits(x[0], x[1])
        self._data = self._loss
        return self._loss


class WeightDecayLayer(LossLayer):
    NAME = "WD"

    def _forward(self, variable_list):
        sum = 0
        for v in variable_list:
            v = A.reshape(v, -1)
            v_square = v.dot(v)
            sum += v_square

        sum *= self.multiplier

        self._loss = sum
        self._data = sum

        return sum



__all__ = [name for name, x in locals().items() if
           not inspect.ismodule(x) and not inspect.isabstract(x)]
