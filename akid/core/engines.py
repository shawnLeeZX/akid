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
from __future__ import absolute_import
import abc

import tensorflow as tf

from .blocks import ValidatableProcessingBlock
from .interface_blocks import UpdateBlock
from .. import backend as A
from .. import parallel as P
from ..utils.tools import is_tuple_or_list
from six.moves import range
from six.moves import zip


class Engine(ValidatableProcessingBlock, UpdateBlock):
    """
    The class that abstracts parallel scheme of network training.

    An `Engine` is responsible for setting up computational graph of a `Brain`
    on proper devices, and the coordination between devices (if there is any).

    More specifically, an `Engine` will take the data from `Sensor`, which
    means it has already been set up, and set up `Brain` and `KongFu`. It
    splits the data provide by `Sensor`, feeds them to devices according to the
    parallel scheme, gathers and processes the results, and provides the end
    result as if no parallelism exists.
    """
    def __init__(self, brain, kongfu, **kwargs):
        if "name" not in kwargs:
            kwargs["name"] = 'engine'
        kwargs["call_on_update"] = True

        super(Engine, self).__init__(**kwargs)

        self.brain = brain
        self.kongfu = kongfu

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

    # def _post_forward(self):
    #     pass
        # self.train_op_list.append(apply_grad_op)
        # train_op = tf.group(*self.train_op_list)

        # with tf.control_dependencies([train_op]):
        #     update_op = self.brain.on_para_update()
        #     if update_op:
        #         self.train_op = tf.group(*update_op)
        #     else:
        #         self.train_op = train_op

        # for grad, var in self.grads:
        #     if grad is not None:
        #         tf.summary.histogram(
        #             var.op.name + '/gradients',
        #             grad,
        #             collections=[TRAINING_DYNAMICS_COLLECTION])


class SingleGPUEngine(Engine):
    def _setup(self):
        if not self.brain.is_setup:
            self.brain.setup()
        if A.backend() == A.TF:
            self.val_brain = self.brain.get_val_copy()
        elif A.backend() == A.TORCH:
            self.val_brain = self.brain

        self.kongfu.set_var_list(self.brain.get_filters())
        self.kongfu.setup()

    def _forward(self, data, val=False):
        if val:
            if A.backend() == A.TORCH:
                # For torch, val brain and brain are the same.
                pushed_brain_state = None
                if self.val_brain.mode != A.Mode.VAL:
                    pushed_brain_state = self.val_brain.mode
                    self.val_brain.set_val(True)
                pushed_name = self.brain.name
                self.val_brain.name = self.brain.name + "_val"
                d = self.val_brain.forward(data)
                if pushed_brain_state is not None:
                    # Right now, we only have train and val for a brain, thus
                    # the saved state is not used actually.
                    self.val_brain.set_val(False)
                self.val_brain.name = pushed_name
            elif A.backend() == A.TF:
                d = self.val_brain.forward(data)
            else:
                raise ValueError("Backend not supported.")
        else:
            pushed_brain_state = None
            if self.brain.mode != A.Mode.TRAIN:
                pushed_brain_state = self.brain.mode
                self.brain.set_val(False)

            d = self.brain.forward(data)

            if pushed_brain_state != None:
                self.brain.set_val(True)
        return d

    def _update(self):
        grads = self.kongfu.forward(self.brain.loss)
        self.train_op = self.kongfu.update(grads)
        return self.train_op

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

    def verbose_eval(self, get_val=False):
        if not get_val:
            return self.brain.verbose_eval
        else:
            return self.val_brain.verbose_eval

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


class DataParallelEngine(SingleGPUEngine):
    """
    This engine will implement typical parallelism in training neural
    network. It splits the batch, and train a fraction of them in an individual
    computing devices.

    Due to the known fact that communication between GPUs are slow, the average
    of gradient is done on CPU.
    """
    def __init__(self, gpu_num=2, val_gpu_num=None, **kwargs):
        super(DataParallelEngine, self).__init__(**kwargs)
        # TODO: if num_gpu is None, use all available ones.
        self.train_gpu_num = gpu_num
        self.train_devices = [i for i in range(gpu_num)]
        if val_gpu_num is not None:
            self.val_gpu_num = val_gpu_num
            self.val_devices = [i for i in range(val_gpu_num)]
        else:
            self.val_gpu_num = gpu_num
            self.val_devices = self.train_devices

        self.gpu_num = self.train_gpu_num
        self.devices = self.train_devices

        if A.backend() != A.TORCH:
            raise ValueError("Backend other torch has not been implemented yet.")
            # TODO: note that the weight decay computation in the old code of
            # tensorflow may be wrong, which makes each replica creates a copy
            # of the weight decay. It may make the weight decay 8 or 4 larger
            # than the reported value, which may be the reason why the
            # reproduced resnet has a lower accuracy. Check when implement
            # tensorflow data parallel.

    def set_val(self, val):
        super(DataParallelEngine, self).set_val(val)

        if val:
            self.devices = self.val_devices
            self.gpu_num = self.val_gpu_num
        else:
            self.devices = self.train_devices
            self.gpu_num = self.train_gpu_num

    def _setup(self):
        if self.gpu_num == 1:
            super(DataParallelEngine, self)._setup()
            return

        # Set up the primary computational graph, and replicate.
        if not self.brain.is_setup:
            self.brain.setup()
        A.get_variable_scope().reuse_variables()
        # Replicate the parameters.
        self._sync(self.brain.get_filters(), self.devices)

        self.towers = self._shadow_copy(self.brain)
        if A.backend() == A.TF:
            self.val_brain = self.brain.get_val_copy()
            self.val_towers = self._shadow_copy(self.val_brain)
        else:
            self.val_brain = self.brain
            self.val_towers = self.towers

        self.kongfu.set_var_list(self.brain.get_filters())
        self.kongfu.setup()

    def _shadow_copy(self, brain):
        """
        Make `self.gpu_num` number of shadow copy of `brain`. A list of shadow
        copies will be returned. The original brain will be put at the first of
        the list.
        """
        # Set up shadow copies.
        towers = []
        towers.append(brain)
        for i in range(1, self.gpu_num):
            with P.device('/gpu:{}'.format(i)):
                tower = brain.get_shadow_copy()
                tower.setup()
                towers.append(tower)

        return towers

    def _forward(self, data, val=False):
        if self.gpu_num == 1:
            d = super(DataParallelEngine, self)._forward(data, val)
            return d

        # Else, do data parallel.
        # Scatter the data to shadow devices.
        data = P.scatter(data, self.devices)

        # Forward propagate the scattered data.
        if A.backend() == A.TORCH:
            if val:
                pushed_name = self.brain.name
                for t in self.towers:
                    t.set_val(True)
                    t.name = t.name + "_val"

        # results here is a general mechanism to gather result, which is
        # another use here though, since loss and evaluation metrics is
        # gathered through loss and eval in towers' attributes.
        results = []
        threads = []
        for i in range(self.gpu_num):
            with P.device("/gpu:{}".format(i)):
                # TODO: think how to handle tensorflow
                ret, t = P.thread_run(self.towers[i].forward, data[i])
                results.append(ret)
                threads.append(t)
        for t in threads:
            t.join()
        # Gather the result, and compute average loss
        losses = [t.loss for t in self.towers]

        loss = self._average(losses)
        evals = [t.eval for t in self.towers]
        eval = self._average(evals)
        # To keep return values, a list is created in `thread_run` to get the
        # results. It only has one element, so we retrieve it.
        results = [r[0] for r in results]
        if None not in results:
            ret = self._average(results)
        else:
            ret = None

        if val:
            self._val_loss = loss
            self._val_eval = eval
            for t in self.towers:
                t.set_val(False)
                t.name = pushed_name
        else:
            self._train_loss = loss
            self._train_eval = eval

        return ret

    def _update(self, *args, **kwargs):
        # TODO: test the behavior of tensorflow, given that the loss is gather
        # from different GPU
        grads = self.kongfu.forward(self._train_loss)
        self.train_op = self.kongfu.update(grads)
        return self.train_op

    def _sync(self, variables, devices):
        """
        Synchronize parameters to devices specified.

        Args:
            variables: a list of variables
            devices: a list of id for GPUs.
        """
        if A.backend() == A.TORCH:
            # Tensorflow handles all communication, no need for this. Do this
            # for torch.
            P.broadcast(variables, devices)

    def _on_update(self, *args, **kwargs):
        # If the backend is torch, broadcast the parameters from primary device
        # to shadow copies.
        if A.backend() == A.TORCH:
            self._sync(self.brain.get_filters(), self.devices)
            # Let the shadow copy retried its parameters again.
            with A.variable_scope(self.name):
                A.get_variable_scope().reuse_variables()
                self.towers = self._shadow_copy(self.brain)

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
            data = list(zip(*data))
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
        with A.variable_scope("data_split"):
            splitted_data = tf.split(axis=0, num_or_size_splits=self.gpu_num, value=data)
            if type(label) is list:
                splitted_labels = []
                for i in range(0, len(label)):
                    splitted_labels.append(tf.split(axis=0, num_or_size_splits=self.gpu_num, value=label[i]))
                splitted_labels = list(zip(*splitted_labels))
            else:
                splitted_labels = tf.split(axis=0, num_or_size_splits=self.gpu_num, value=label)

        return splitted_data, splitted_labels

    def _average(self, data):
        """
        Given a list of computing towers, average their loss, and return.
        """
        # TODO: think how the gather works in tensorflow
        data_gathered = P.gather(data)
        def mean_inner(d, name=None):
            if is_tuple_or_list(d):
                return list(map(mean_inner, d))
            else:
                d_reduced = A.mean(d)
                return d_reduced

        data_reduced = mean_inner(data_gathered)

        if is_tuple_or_list(data_reduced):
            # Rename all the results. The results are a list of tensors, or named
            # tuples. So an iteration would do.
            for i, r in enumerate(data_reduced):
                name = A.get_name(data[0][i], with_device_id=False)
                if name:
                    new_name = name # NOTE: Use the same name for now.
                    if A.is_tensor(r):
                        A.cache_tensor_auto_scope(r, new_name)
                    elif isinstance(data[0][i], A.NamedValue):
                        r = type(data[0][i])(new_name, r)
                        # TODO: cache the tuple results separately
                        data_reduced[i] = r
        else:
            name = A.get_name(data[0])
            if name:
                new_name = name
                A.cache_tensor_auto_scope(data_reduced, new_name)

        return data_reduced

    def _average_eval(self, towers):
        """
        Given a list of computing towers, average their evaluation metrics and
        return.
        """
        with A.variable_scope("eval_average"):
            eval_list = []
            for i in range(0, len(towers[0].eval)):
                sum_of_eval = tf.add_n([t.eval[i] for t in towers])
                eval = tf.div(sum_of_eval,
                              self.gpu_num,
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
        for i in range(0, self.gpu_num):
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
                if i is not self.gpu_num - 1:
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
        for i in range(0, self.gpu_num):
            self.log("Setting up tower {} for validation".format(i))
            with tf.device('/gpu:{}'.format(i)):
                system_in = self._setup_system_in(splitted_data[i],
                                                  splitted_labels[i])
                tower.forward(system_in)
                self.val_towers.append(tower)
                if i is not self.gpu_num - 1:
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
        with A.variable_scope("gradient_average"):
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


engines = {}


class EngineRegistry(object):
    def __init__(self,
                 name,
                 cls,
                 message=None,
                 required_fields=(),
                 default_paras={}):
        self.message = message
        self.cls = cls
        self.name = name
        self.default_paras = default_paras
        self.required_fields = required_fields
        engines[name] = self


def get(name, **kwargs):
    entry = engines[name]
    for f in entry.required_fields:
        if f not in kwargs:
            raise KeyError("Required field {} not found in the engine parameters.".format(entry.required_fields))

    for k in entry.default_paras:
        if k not in kwargs:
            kwargs[k] = entry.default_paras[k]

    return engines[name].cls(**kwargs)


EngineRegistry('single',
               SingleGPUEngine,
               "Single GPU engine")
EngineRegistry('data_parallel',
               DataParallelEngine,
               "Data Parallel Engine. Specify the number of GPU with field `gpu_num` (2 GPU is used by default).",
               default_paras={"gpu_num": 2})
