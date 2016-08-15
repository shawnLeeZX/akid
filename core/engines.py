"""
Collections `Engines` to implement the scheme to actually do the computation.
"""
import abc

import tensorflow as tf

from .common import TRAINING_DYNAMICS_COLLECTION
from .common import GLOBAL_STEP, global_var_scope
from ..utils import glog as log


class Engine(object):
    """
    The class that abstracts parallel scheme of network training.

    An `Engine` is responsible for setting up computational graph of a `Brain`
    on proper devices, and the coordination between devices (if there is any).

    More specifically, an `Engine` will take an already set up `Sensor`, unset
    up `Brain` and `KongFu`. It splits the data provide by `Sensor`, feeds them
    to devices according to the parallel scheme, gathers and processes the
    results, and provides the end result as if no parallelism exists.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, sensor, brain, kongfu):
        self.sensor = sensor
        self.brain = brain
        self.kongfu = kongfu

        with tf.variable_scope(global_var_scope):
            self.global_step_tensor = tf.get_variable(
                name=GLOBAL_STEP,
                shape=[],
                initializer=tf.constant_initializer(0),
                trainable=False)

    @abc.abstractmethod
    def loss(self, get_val=False):
        """
        Return the loss of a `Brain`. If `get_val` is True, return the
        validation loss, otherwise, return training loss.
        """
        raise NotImplementedError("Each engine should implement the interface"
                                  " to provide loss.")

    @abc.abstractmethod
    def eval(self, get_val=False):
        """
        Return the evaluation list of a `Brain`. If `get_val` is True, return
        that of validation brain, otherwise, that of training brain.
        """
        raise NotImplementedError("Each engine should implement the interface"
                                  " to provide evaluation.")

    def setup(self):
        grads = self._setup_train_towers()
        self._post_setup_train(grads)
        self._setup_val_towers()

    def _post_setup_train(self, grads):
        apply_grad_op = self.kongfu.opt.apply_gradients(
            grads, global_step=self.global_step_tensor)

        with tf.control_dependencies([apply_grad_op]):
            self.brain.on_batch_finishes()
            if self.brain.max_norm_clip_op:
                self.train_op = self.brain.max_norm_clip_op
            else:
                self.train_op = apply_grad_op

        for grad, var in grads:
            if grad is not None:
                tf.histogram_summary(
                    var.op.name + '/gradients',
                    grad,
                    collections=[TRAINING_DYNAMICS_COLLECTION])


class SingleGPUEngine(Engine):
    def _setup_train_towers(self):
        # TODO(Shuai): here is a temporary solution. Since sensor is actually
        # just a system with two outputs, `GraphSystem` could handle it, but it
        # would make many changes, for now, I just settle with the not-so
        # elegant solution here. Note that if here is going to be changed,
        # `_setup_val_brain` should also be changed.
        # Note that a cascade change is needed for all engines if this logic
        # needs changes.
        data = self.sensor.data()
        label = self.sensor.labels()
        system_in = [data]
        system_in.extend(label) if type(label) is list \
            else system_in.append(label)
        self.brain.setup(system_in)
        self.kongfu.setup(self, self.brain.loss)

        return self.kongfu.data

    def _setup_val_towers(self):
        self.val_brain = self.brain.get_val_copy()
        data = self.sensor.data(get_val=True)
        label = self.sensor.labels(get_val=True)
        system_in = [data]
        system_in.extend(label) if type(label) is list \
            else system_in.append(label)
        self.val_brain.setup(system_in)

    def loss(self, get_val=False):
        if not get_val:
            return self.brain.loss
        else:
            return self.val_brain.loss

    def eval(self, get_val=False):
        if not get_val:
            return self.brain.eval
        else:
            return self.val_brain.eval

    def get_layer_data(self, name, get_val=False):
        """
        Given the `name` of the layer, return the tensor of the data of this
        layer. `get_val` has similar meaning with `loss`.
        """
        layer = None
        for b in self.val_brain.blocks:
            if b.name == name:
                layer = b
                break
        if layer is None:
            raise Exception("Layer {} is not found.".format(name))

        return layer.data


class DataParallelEngine(Engine):
    """
    This engine will implement typical parallelism in training neural
    network. It splits the batch, and train a fraction of them in an individual
    computing devices.

    Due to the known fact that communication between GPUs are slow, the average
    of gradient is done on CPU.
    """
    def __init__(self, num_gpu, **kwargs):
        super(DataParallelEngine, self).__init__(**kwargs)
        self.num_gpu = num_gpu

    def _split_input(self, data, label):
        """
        Given data and labels, split them and return.
        """
        with tf.variable_scope("data_split"):
            splitted_data = tf.split(0, self.num_gpu, data)
            if type(label) is list:
                for i in xrange(0, len(label)):
                    label[i] = tf.split(0, self.num_gpu, label[i])
                splitted_labels = zip(*label)
            else:
                splitted_labels = tf.split(0, self.num_gpu, label)

        return splitted_data, splitted_labels

    def _setup_system_in(self, data, label):
        """
        Given data and label, create the list of input for the brain.

        Args:
            data: tf.Tensor
                The data tensor.
            label: list of tf.Tensor or tf.Tensor
                List of label tensors or a single label tensor.

        Return:
            system_in: list
                A list of tensors merge `data` and `label`.
        """
        system_in = [data]
        system_in.extend(list(label)) \
            if type(label) in (list, tuple)\
            else system_in.append(label)
        return system_in

    def _average_loss(self, towers):
        """
        Given a list of computing towers, average their loss, and return.
        """
        with tf.variable_scope("loss_average"):
            sum_of_loss = tf.add_n([t.loss for t in towers])
            loss = tf.div(sum_of_loss, self.num_gpu, name="avg")

        return loss

    def _average_eval(self, towers):
        """
        Given a list of computing towers, average their evaluation metrics and
        return.
        """
        with tf.variable_scope("eval_average"):
            eval_list = []
            for i in xrange(0, len(towers[0].eval)):
                sum_of_eval = tf.add_n([t.eval[i] for t in towers])
                eval = tf.div(sum_of_eval,
                        self.num_gpu,
                        name="{}_avg".format(towers[0].eval[i].op.name))
                eval_list.append(eval)

        return eval_list

    def _setup_train_towers(self):
        # Split the data.
        data = self.sensor.data()
        label = self.sensor.labels()
        splitted_data, splitted_labels = self._split_input(data, label)

        # Set up brains according to the number of gpus used.

        # Since we need reduce operations on replicas of brains, we need to
        # keep track of different brains, denoted as towers of the
        # brain. Similar with validation towers.
        self.train_towers = []
        tower_grads = []
        tower = self.brain
        kongfu = self.kongfu
        for i in xrange(0, self.num_gpu):
            log.info("Setting up tower {} for training".format(i))
            with tf.device('/gpu:{}'.format(i)):
                # Set up a tower
                system_in = self._setup_system_in(splitted_data[i],
                                                  splitted_labels[i])
                tower.setup(system_in)

                # Keep track of the new tower.
                self.train_towers.append(tower)

                # Set up KongFu (optimizer).
                # For now, we do not need to keep track of Kongfu, so just set
                # it up multiple times.
                kongfu.setup(self, tower.loss)

                # Create the next tower.
                # Do not do copy at the last tower.
                if i is not self.num_gpu - 1:
                    tower = tower.get_shadow_copy()
                    kongfu = kongfu.get_shadow_copy()

                tower_grads.append(kongfu.data)

        # Gather and reduce.
        with tf.device('/cpu:0'):
            grads = self._average_grads(tower_grads)
            self._train_loss = self._average_loss(self.train_towers)
            self._train_eval = self._average_eval(self.train_towers)

        return grads

    def _setup_val_towers(self):
        self.val_brain = self.brain.get_val_copy()
        data = self.sensor.data(get_val=True)
        label = self.sensor.labels(get_val=True)
        splitted_data, splitted_labels = self._split_input(data, label)

        # Set up val brains according to the number of gpus used.
        self.val_towers = []
        tower = self.val_brain
        for i in xrange(0, self.num_gpu):
            log.info("Setting up tower {} for validation".format(i))
            with tf.device('/gpu:{}'.format(i)):
                system_in = self._setup_system_in(splitted_data[i],
                                                  splitted_labels[i])
                tower.setup(system_in)
                self.val_towers.append(tower)
                if i is not self.num_gpu - 1:
                    tower = tower.get_shadow_copy()

        with tf.device('/cpu:0'):
            self._val_loss = self._average_loss(self.val_towers)
            self._val_eval = self._average_eval(self.val_towers)

    def loss(self, get_val=False):
        if not get_val:
            return self._train_loss
        else:
            return self._val_loss

    def eval(self, get_val=False):
        if not get_val:
            return self._train_eval
        else:
            return self._val_eval

    def _average_grads(self, tower_grads):
        """
        Calculate the average gradient for each shared variable across all
        towers.

        Note that this function provides a synchronization point across all
        towers.

        Args:
            tower_grads: List of lists of (gradient, variable) tuples.
                The outer list is over individual gradients. The inner list is
                over the gradient calculation for each tower.
        Returns:
            List of pairs of (gradient, variable), where the gradient has been
            averaged across all towers.
        """
        with tf.variable_scope("gradient_average"):
            average_grads = []
            for grad_and_vars in zip(*tower_grads):
                # Note that each grad_and_vars looks like the following:
                #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
                grads = []
                for g, _ in grad_and_vars:
                    # Add 0 dimension to the gradients to represent the tower.
                    expanded_g = tf.expand_dims(g, 0)

                    # Append on a 'tower' dimension which we will average over
                    # below.
                    grads.append(expanded_g)

                # Average over the 'tower' dimension.
                grad = tf.concat(0, grads)
                grad = tf.reduce_mean(grad, 0)

                # Keep in mind that the Variables are redundant because they
                # are shared across towers. So .. we will just return the first
                # tower's pointer to the Variable.
                v = grad_and_vars[0][1]
                grad_and_var = (grad, v)
                average_grads.append(grad_and_var)

        return average_grads
