"""
The distributed computing stack is responsible to handle concurrency and
communication between different computing nodes, so the end user only needs to
deal with how to build a power network. All complexity has been hidden in the
class `Engine`. The usage of `Engine` is just to pick and use.

More specifically, `akid` offers built-in data parallel scheme in form of class
`Engine`. Currently, the engine mainly works with neural network training,
which is be used with `Kid` by specifying the engine at the construction of the
kid.

As an example, we could do data parallelism on multiple computing towers using::

    kid = kids.Kid(
        sensor,
        brain,
        MomentumKongFu(lr_scheme={"name": LearningRateScheme.placeholder}),
        engine={"name": "data_parallel", "num_gpu": 2},
        log_dir="log",
        max_epoch=200)
"""
import abc

import tensorflow as tf

from .common import TRAINING_DYNAMICS_COLLECTION
from . import common
from .blocks import ProcessingBlock


class Engine(ProcessingBlock):
    """
    The class that abstracts parallel scheme of network training.

    TODO: A better way of abstraction is to pass in already built sensor and
    brain, then offer a property called loss. The parallelism should be done at
    batch dimension.

    An `Engine` is responsible for setting up computational graph of a `Brain`
    on proper devices, and the coordination between devices (if there is any).

    More specifically, an `Engine` will take an already set up `Sensor`, unset
    up `Brain` and `KongFu`. It splits the data provide by `Sensor`, feeds them
    to devices according to the parallel scheme, gathers and processes the
    results, and provides the end result as if no parallelism exists.
    """
    def __init__(self, kid, **kwargs):
        super(Engine, self).__init__(**kwargs)
        self.kid = kid

        self.sensor = kid.sensor
        self.brain = kid.brain
        self.kongfu = kid.kongfu

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

    @property
    def data(self):
        """
        Placeholder. Do not return anything. Ideally, it should return whatever
        data the brain it contains. But it does not of use for now. So just
        skip.
        """
        return None

    def _post_forward(self):
        apply_grad_op = self.kongfu.opt.apply_gradients(
            self.grads, global_step=common.global_step_tensor)

        self.train_op_list.append(apply_grad_op)
        train_op = tf.group(*self.train_op_list)

        with tf.control_dependencies([train_op]):
            update_op = self.brain.on_para_update()
            if update_op:
                self.train_op = tf.group(*update_op)
            else:
                self.train_op = train_op

        for grad, var in self.grads:
            if grad is not None:
                tf.summary.histogram(
                    var.op.name + '/gradients',
                    grad,
                    collections=[TRAINING_DYNAMICS_COLLECTION])

    def _forward(self):
        self.log("Build training phase ...")
        self._forward_train()
        self.log("Build validation phase ...")
        self._forward_val()

    def _setup(self):
        self.brain.setup()
        self.val_brain = self.brain.get_val_copy()


class SingleGPUEngine(Engine):
    def _forward_train(self):
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
        self.brain.forward(system_in)

        if self.brain.train_op is not None:
            if type(self.brain.train_op) is list:
                self.train_op_list = list(self.brain.train_op)
            else:
                self.train_op_list = [self.brain.train_op]
        else:
            self.train_op_list = []

        self.grads = self.kongfu.forward(self.brain.loss)

        return self.grads

    def _forward_val(self):
        data = self.sensor.data(get_val=True)
        label = self.sensor.labels(get_val=True)
        system_in = [data]
        system_in.extend(label) if type(label) is list \
            else system_in.append(label)
        self.val_brain.forward(system_in)

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
        if get_val:
            brain = self.val_brain
        else:
            brain = self.brain
        layer = None
        for b in brain.blocks:
            if b.name == name:
                layer = b
                break
        if layer is None:
            raise Exception("Layer {} is not found.".format(name))

        return layer.data

    def get_layer(self, name, get_val=False):
        """
        Given the `name` of the layer, return the tensor of the data of this
        layer. `get_val` has similar meaning with `loss`.
        """
        if get_val:
            brain = self.val_brain
        else:
            brain = self.brain

        layer = None
        for b in brain.blocks:
            if b.name == name:
                layer = b
                break
        if layer is None:
            raise Exception("Layer {} is not found.".format(name))

        return layer


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

    def get_layer_data(self, name, get_val=False):
        if get_val:
            towers = self.val_towers
        else:
            towers = self.train_towers

        data = []
        for t in towers:
            l = t.get_layer_by_name(name)
            data.append(l.data)

        # Check whether the output is a list, if it is merge by list members
        if type(data[0]) is list:
            data = zip(*data)
            for i, l in enumerate(data):
                data[i] = tf.concat(axis=0, values=l, name=name)
        # Otherwise, concat.
        else:
            data = tf.concat(axis=0, values=data, name=name)

        return data

    def _split_input(self, data, label):
        """
        Given data and labels, split them and return.
        """
        with tf.variable_scope("data_split"):
            splitted_data = tf.split(axis=0, num_or_size_splits=self.num_gpu, value=data)
            if type(label) is list:
                splitted_labels = []
                for i in xrange(0, len(label)):
                    splitted_labels.append(tf.split(axis=0, num_or_size_splits=self.num_gpu, value=label[i]))
                splitted_labels = zip(*splitted_labels)
            else:
                splitted_labels = tf.split(axis=0, num_or_size_splits=self.num_gpu, value=label)

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

    def _forward_train(self):
        # Split the data.
        data = self.sensor.data()
        label = self.sensor.labels()

        # TODO: A description of how the data parallelism is done should be
        # provided. Otherwise, the logging information only has the one of the
        # first built tower.

        splitted_data, splitted_labels = self._split_input(data, label)

        # Set up brains according to the number of gpus used.

        # Since we need reduce operations on replicas of brains, we need to
        # keep track of different brains, denoted as towers of the
        # brain. Similar with validation towers.
        self.train_towers = []
        tower_grads = []
        tower = self.brain
        kongfu = self.kongfu
        self.train_op_list = []
        for i in xrange(0, self.num_gpu):
            self.log("Setting up tower {} for training".format(i))
            with tf.device('/gpu:{}'.format(i)):
                # Set up a tower
                system_in = self._setup_system_in(splitted_data[i],
                                                  splitted_labels[i])
                tower.forward(system_in)
                if type(tower.train_op) is list:
                     self.train_op_list.extend(tower.train_op)
                else:
                    self.train_op_list.append(self.brain.train_op)

                # Keep track of the new tower.
                self.train_towers.append(tower)

                # Set up KongFu (optimizer).
                # For now, we do not need to keep track of Kongfu, so just set
                # it up multiple times.
                kongfu.forward(tower.loss)

                # Create the next tower.
                # Do not do copy at the last tower.
                if i is not self.num_gpu - 1:
                    tower = tower.get_shadow_copy()
                    kongfu = kongfu.get_shadow_copy()

                tower_grads.append(kongfu.data)

        # Gather and reduce.
        with tf.device('/cpu:0'):
            self.grads = self._average_grads(tower_grads)
            self._train_loss = self._average_loss(self.train_towers)
            self._train_eval = self._average_eval(self.train_towers)

        return self.grads

    def _forward_val(self):
        data = self.sensor.data(get_val=True)
        label = self.sensor.labels(get_val=True)
        splitted_data, splitted_labels = self._split_input(data, label)

        # Set up val brains according to the number of gpus used.
        self.val_towers = []
        tower = self.val_brain
        for i in xrange(0, self.num_gpu):
            self.log("Setting up tower {} for validation".format(i))
            with tf.device('/gpu:{}'.format(i)):
                system_in = self._setup_system_in(splitted_data[i],
                                                  splitted_labels[i])
                tower.forward(system_in)
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
                grad = tf.concat(axis=0, values=grads)
                grad = tf.reduce_mean(grad, 0)

                # Keep in mind that the Variables are redundant because they
                # are shared across towers. So .. we will just return the first
                # tower's pointer to the Variable.
                v = grad_and_vars[0][1]
                grad_and_var = (grad, v)
                average_grads.append(grad_and_var)

        return average_grads
