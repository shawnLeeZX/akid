"""
Collections `Engines` to implement the scheme to actually do the computation.
"""
import tensorflow as tf

from .common import TRAINING_DYNAMICS_COLLECTION
from .common import GLOBAL_STEP, global_var_scope


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


class SingleGPUEngine(Engine):
    def setup(self):
        self._setup_train()
        self._setup_val()

    def _setup_train(self):
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
        self.kongfu.setup(self)

        grads = self.kongfu.opt.compute_gradients(self.brain.loss_graph)
        # Add histograms for gradients.
        for grad, var in grads:
            if grad is not None:
                tf.histogram_summary(
                    var.op.name + '/gradients',
                    grad,
                    collections=[TRAINING_DYNAMICS_COLLECTION])

        apply_grad_op = self.kongfu.opt.apply_gradients(
            grads, global_step=self.global_step_tensor)

        with tf.control_dependencies([apply_grad_op]):
            self.brain.on_batch_finishes()
            if self.brain.max_norm_clip_op:
                self.train_op = self.brain.max_norm_clip_op
            else:
                self.train_op = apply_grad_op

    def _setup_val(self):
        self.val_brain = self.brain.get_val_copy()
        data = self.sensor.data(get_val=True)
        label = self.sensor.labels(get_val=True)
        system_in = [data]
        system_in.extend(label) if type(label) is list \
            else system_in.append(label)
        self.val_brain.setup(system_in)

    def loss(self, get_val=False):
        """
        Return the loss of a `Brain`. If `get_val` is True, return the
        validation loss, otherwise, return training loss.
        """
        if not get_val:
            return self.brain.loss_graph
        else:
            return self.val_brain.loss_graph

    def eval(self, get_val=False):
        """
        Return the evaluation list of a `Brain`. If `get_val` is True, return
        that of validation brain, otherwise, that of training brain.
        """
        if not get_val:
            return self.brain.eval_graph_list
        else:
            return self.val_brain.eval_graph_list

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
