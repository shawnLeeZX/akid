"""
This module contains `Kid` class to play the survival game.
"""
from __future__ import absolute_import, division, print_function

import os
import time
import sys
import inspect

import tensorflow as tf

from ..utils import glog as log
from . import sensors
from . import engines
from .blocks import Block
from .kongfus import LearningRateScheme
from . import common
from .common import (
    TRAIN_SUMMARY_COLLECTION,
    VALID_SUMMARY_COLLECTION,
    TRAINING_DYNAMICS_COLLECTION,
)


class Kid(Block):
    """
    Kid is a class to assemble a `Sensor`, for supplying data, a
    `Brain`, for data processing, and a genre of `KongFu`, for algorithms or
    polices to train.

    It supports parallelism by specifying different engines.

    It has the following hooks:

        * `on_train_log_step`.
        * `on_val_log_step`
        * `on_train_begin`
        * `on_batch_begin`

    Refer to function that calls functions on hooks for detailed explanation on
    what does those hooks do. For example, to refer to method
    `on_train_log_step` for more on the hook `on_train_log_step`.

    To add a function to one of those hooks, append the function to
    `hooks.hook_name`. For example, to add a hook to `on_train_log_step`, call

    ```
    def func(kid):
        ...

    hooks.on_train_log_step.append(func)
    ```

    where `func` is the function you want it to be called. Functions added to
    hooks are supposed to take a `Kid` instance, which serves to provide
    information needed. That is also to say, no more information is available.
    """
    def __init__(self,
                 sensor_in,
                 brain_in,
                 kongfu_in,
                 engine="single",
                 sess=None,
                 max_steps=None,
                 max_epoch=None,
                 log_dir=None,
                 log_to_file=True,
                 val_log_step=1000,
                 train_log_step=100,
                 graph=None,
                 save_chk_point=True,
                 do_summary=True,
                 summary_on_val=False,
                 debug=False):
        """
        Assemble a sensor, a brain, and a KongFu to start the survival game.

        Args:
            sensor: Sensor
                A `Sensor` class to supply data.
            brain_in: Brain
                A `Brain` class to process data.
            kongfu_in: KongFu
                A `KongFu` class for training.
            engine: a str or a dict
                When it is a str, it should be the name of the `Engine` class
                to use, which implements parallel scheme. Available engines
                are:

                    'single', 'data_parallel'

                Default parameters of that scheme will be used.

                When it is a dict, it should be of the form:

                    {"name": "single"}
                    {"name": "data_parallel", "num_gpu": 2}

               where the `name` key indicates the parallel scheme while other
               keys are parameters of that scheme. If parameters are not
               provided, again default ones will be used.
            sess: tf.Session
                The session to use. If None, one will be created.
            max_steps: int
            max_epoch: int
                You can only specify either max epoch to train or max
                steps. Not both.
            log_dir: str
                The folder to hold tensorboard event, training logs and trained
                models. If not given, first a folder named `log` will be
                created, then a folder named by current time stamp will be the
                folder to keep all the mentioned files.
            log_to_file: Boolean
                Whether to save log to file.
            graph: tf.Graph()
                The computational graph this kid is in. If not given, a
                new graph will be created.
            val_log_step: int
                After how many steps evaluation on the validation dataset
                should be taken.
            train_log_step: int
                After how many steps training statistics should be logged.
            do_summary: Boolean
                If False, no tensorboard summaries will be saved at all. Note
                that if `Brain` or `Sensor`'s `do_summary` option is True, they
                will not be unset. Though during training, they would not be
                used. This makes possible that objects other than this kid
                could use the summary ops created by the brain or sensor.
            summary_on_val: Boolean
                Whether to collect summary on the validation brain. This option
                is for debugging purpose. It is useful to see how summary in
                activation of validation brain is different from training
                brain. However, if such option is used, accuracy on validation
                set will become inaccurate. This is because input from
                validation source is needed when doing summaries on validation
                brain, which would make validation data be used. Since
                validation source will reshuffle data after one epoch is
                finished, some validation may be reused and some may not be
                seen at all when doing the actual validation.
            Other args are self-evident.
        """
        self.sensor = sensor_in
        self.brain = brain_in
        self.kongfu = kongfu_in
        self.engine_para = engine
        self.sess = sess
        self.debug = debug

        # Set up logging facilities.
        if log_dir is None:
            # Naming log dir according to time if not specified.
            log_dir = "log/" + time.ctime()
            # As ':' is widely used in network protocols, replace it with '_'
            # to avoid conflict.
            self.log_dir = log_dir.replace(':', '_')
        else:
            self.log_dir = os.path.normpath(log_dir)
        self.log_filepath = self.log_dir + "/training.log"
        self.model_dir = self.log_dir + "/model"
        self.log_to_file = log_to_file

        self.max_steps = max_steps
        self.max_epoch = max_epoch
        assert self.max_steps is None or self.max_epoch is None,\
            "Only one of `max_steps` and `max_epoch` could be used."
        assert self.max_steps is not None or self.max_epoch is not None,\
            "At least one `max_steps` and `max_epoch` is needed."

        self.train_log_step = train_log_step
        self.val_log_step = val_log_step
        self.summary_on_val = summary_on_val
        self.do_summary = do_summary
        self.save_chk_point = save_chk_point

        # A tensorflow computational graph to hold training and validating
        # graphs.
        if not graph:
            self.graph = tf.Graph()
        else:
            self.graph = graph

        # Flag to indicate the input queue runner has been started or not,
        # since both training and validation may start this, while it should
        # only be started once.
        self.initialized = False

        # Set up hooks.
        class hooks(object):
            def __init__(self):
                self.on_training_log = []
                self.on_val_log = []
                self.on_train_begin = []
                self.on_batch_begin = []
                self.on_epoch_end = []
                self.add_default_hooks()

            def add_default_hooks(self):
                from .callbacks import on_train_log_step
                self.on_training_log.append(on_train_log_step)

                from .callbacks import on_val_log_step
                self.on_val_log.append(on_val_log_step)

                from .callbacks import on_train_begin
                self.on_train_begin.append(on_train_begin)

                from .callbacks import on_batch_begin
                self.on_batch_begin.append(on_batch_begin)

        self.hooks = hooks()

        # Class members whose value depends on the state of the class.
        self.feed_dict = None
        self.loss_value = None
        self.evals = None
        self.best_val_evals = None

    def validate(self):
        """Evaluating on validation set.

        Return:
            loss: float
                The validation loss. Note that though only loss is returned,
                `Kid`'s reference to evaluation metrics and loss are both
                updated.
        """
        self.log('Validation Data Eval:')

        if not self.initialized:
            self.init(continue_from_chk_point=True)

        # Run one epoch of eval.
        eval_metric_values = [0] * len(self.engine.eval(get_val=True))
        loss = 0
        steps_per_epoch = self.sensor.num_batches_per_epoch_val

        for step in xrange(steps_per_epoch):
            if type(self.sensor) is sensors.FeedSensor:
                self.feed_dict = self.sensor.fill_feed_dict(get_val=True)

            fetch = [self.engine.loss(get_val=True)]
            fetch.extend(self.engine.eval(get_val=True))
            result = self.sess.run(fetch, feed_dict=self.feed_dict)

            loss += result[0]
            for i, v in enumerate(result[1:]):
                eval_metric_values[i] += v

        loss /= steps_per_epoch
        for i, v in enumerate(eval_metric_values):
            eval_metric_values[i] = v / steps_per_epoch

        self.loss_value = loss
        self.evals = eval_metric_values
        self.on_val_log_step()

        return loss, eval_metric_values

    def setup(self):
        """
        Set up logging and the computation graph.
        """
        with self.graph.as_default():
            common.init()
            self.global_step_tensor = common.global_step_tensor
            self._setup_log()
            self._setup_sensor()
            self._setup_engine()
            self._setup_summary()
            self.saver = tf.train.Saver(tf.global_variables())
            if self.sess is None:
                config = tf.ConfigProto(allow_soft_placement=True)
                config.gpu_options.allow_growth = True
                self.sess = tf.Session(graph=self.graph, config=config)

    def teardown(self):
        """
        Close sessions.

        This method has not been tested whether it works or not. It stays here
        to remind that any session created by kid may cause memory leak.
        """
        self.sess.close()
        self.sess.reset()

    def practice(self, continue_from_chk_point=False, return_eval=False):
        """
        Improve the performance of the kid's brain by practicing, aka
        applying back propagation to train neural network.

        Args:
            continue_from_chk_point: Boolean
                Setup configuration. Passed to `setup`
            return_eval: bool
                Return evaluation metric after trainning is finished.
        Return:
            loss, [eval]: float, float
                Final validation loss and optional evaluation metric.
        """
        try:
            self.init(continue_from_chk_point)
            # And then after everything is built, start the training loop.
            self.log("Begin training brain: " + self.brain.name)
            previous_step = tf.train.global_step(self.sess,
                                                 self.global_step_tensor)
            self.step = previous_step
            # Note the epoch estimation is not accurate if the batch size
            # cannot divide total number of training samples.
            self.epoch = previous_step \
                // self.sensor.num_batches_per_epoch_train

            self.on_train_begin()

            while self.step < self.max_steps + 1:
                if self.step % self.val_log_step == 0 or\
                   self.step == self.max_steps:
                    if self.save_chk_point:
                        self.save_to_ckpt()
                    loss, eval_ = self.validate()

                self.forward_backward()

                self.step += 1

                if self.step % self.sensor.num_batches_per_epoch_train is 0:
                    self.epoch += 1
                    self.on_epoch_end()

                if self.step % self.train_log_step == 0:
                    self.on_train_log_step()

            if return_eval:
                return loss, eval_
            else:
                return loss

        except tf.OpError as e:
            self.log("Tensorflow error when running: {}".format(e.message))
            sys.exit(0)

    def _setup_log(self):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        # TODO: Think how to add a check that if log has initialized, do not do
        # this any more.
        log.init("stdout")
        if self.log_to_file:
            log.init(self.log_filepath)
        if self.debug:
            log.setLevel(log.DEBUG)
        self.log("Logs will be save to: {}".format(self.log_dir))

    def _setup_summary(self):
        if self.do_summary:
            # SummaryWriter to output summaries and the Graph.
            self.summary_writer = tf.summary.FileWriter(self.log_dir)
            self.log("Summary event file will be saved to {}".format(
                self.log_dir))
            # Build the summary operation based on the TF collection of
            # Summaries.
            summary_ops = tf.get_collection(TRAIN_SUMMARY_COLLECTION)
            summary_ops.extend(tf.get_collection(
                TRAINING_DYNAMICS_COLLECTION))
            if self.summary_on_val:
                val_summary_ops = tf.get_collection(
                    VALID_SUMMARY_COLLECTION)
                summary_ops.extend(val_summary_ops)
            self.summary_op = tf.summary.merge(summary_ops)
            # Write the brain to tensorflow event file.
            self.summary_writer.add_graph(self.graph)

    def _setup_sensor(self):
        # Build training graph.
        self.sensor.forward()

    def _setup_engine(self):
        if type(self.engine_para) is str:
            engine_name = self.engine_para
        else:
            try:
                engine_name = self.engine_para["name"]
            except KeyError as e:
                e.message = "Engine {} name is not found.".format(engine_name)
                raise e

        if engine_name == "single":
            self.engine = engines.SingleGPUEngine(self, name="SEngine")
        elif engine_name == "data_parallel":
            if type(self.engine_para) is str:
                # TODO: automatically use the maximal even number of gpus.
                num_gpu = 2
            else:
                num_gpu = self.engine_para["num_gpu"]
            self.engine = engines.DataParallelEngine(num_gpu, kid=self, name="DEngine")
        else:
            raise Exception('No engine "{}". Perhaps you have a typo.'.format(
                engine_name))

        self.engine.forward()

    def init(self, continue_from_chk_point=None):
        """
        Initialize computational graph for training. It initializes or restores
        variables, starts queues and so on.

        Args:
            continue_from_chk_point: Boolean
                Continue from a previous training or not. If it is True, a
                folder named `model` must exist under `Kid`'s `log_dir`
                with saved models.
        """
        # Initialization.
        if continue_from_chk_point:
            # Train from pre-trained model.
            self.restore_from_ckpt()
        else:
            with self.graph.as_default():
                init = tf.global_variables_initializer()
            self.sess.run(init)

        # Start queue runner if needed.
        if type(self.sensor) is sensors.IntegratedSensor:
            if not self.initialized:
                tf.train.start_queue_runners(sess=self.sess)

        self.initialized = True

        if self.max_epoch:
            # Convert the max epoch number to max steps.
            self.max_steps \
                = self.sensor.num_batches_per_epoch_train * self.max_epoch

    def save_to_ckpt(self):
        step = tf.train.global_step(self.sess, self.global_step_tensor)
        self.saver.save(self.sess,
                        self.model_dir + "/checkpoint",
                        global_step=step)
        self.log("Checkpoint at step {} saved to folder:"
                 " {}".format(step, self.model_dir))

    def restore_from_ckpt(self):
        """
        Restore variables of this net from the latest checkpoint of
        `model_dir`.

        Return:
            Training step of the checkpoint the net are recovering from.
        """
        checkpoint = tf.train.get_checkpoint_state(self.model_dir)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.log("Recovering net from checkpoint %s."
                     % checkpoint.model_checkpoint_path)
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            filename = checkpoint.model_checkpoint_path.split('/')[-1]
            step = int(filename.split('-')[-1])
            return step
        else:
            self.error("No checkpoint found under %s!" % self.model_dir)
            sys.exit()

    def fill_train_feed_dict(self):
        if type(self.sensor) is sensors.FeedSensor:
            # Placeholder of `FeedSensor` should be filled.
            self.feed_dict = self.sensor.fill_feed_dict()
            if self.summary_on_val:
                # Validation data is also needed, so add them in.
                val_feed_dict = self.sensor.fill_feed_dict(True)
                self.feed_dict.update(val_feed_dict)

        if self.kongfu.lr_scheme["name"] is LearningRateScheme.placeholder:
            if not self.kongfu.lr_value:
                raise Exception("You should feed a learning rate to"
                                " `Kongfu.lr_value`")
            lr_dict = {self.kongfu.learning_rate: self.kongfu.lr_value}
            if self.feed_dict:
                self.feed_dict.update(lr_dict)
            else:
                self.feed_dict = lr_dict

    def forward_backward(self):
        """
        Train for one step.
        """
        # Run one step.
        self.on_batch_begin()

        self.fill_train_feed_dict()

        fetch = [self.engine.train_op, self.engine.loss()]
        fetch.extend(self.engine.eval())
        start_time = time.time()
        result = self.sess.run(fetch, feed_dict=self.feed_dict)
        self.forward_backward_time = time.time() - start_time
        self.loss_value = result[1]
        self.evals = result[2:]

    def on_train_log_step(self):
        """
        Call hooks at the time when the kid should do logging for training.
        """
        for func in self.hooks.on_training_log:
            func(self)

    def on_val_log_step(self):
        """
        Call hooks at the time when the kid should do logging for validation.
        """
        # Update the best validation evaluation results.
        if self.best_val_evals:
            for i, e in enumerate(self.evals):
                if self.best_val_evals[i] < e:
                    self.best_val_evals[i] = e
        else:
            self.best_val_evals = list(self.evals)

        for func in self.hooks.on_val_log:
            func(self)

    def on_train_begin(self):
        for func in self.hooks.on_batch_begin:
            func(self)
        for func in self.hooks.on_train_begin:
            func(self)

    def on_batch_begin(self):
        """
        NOTE: all functions attached to this hook will be called in the hook
        `on_train_begin` as well, with a higher priority.
        """
        for func in self.hooks.on_batch_begin:
            func(self)

    def on_epoch_end(self):
        for func in self.hooks.on_epoch_end:
            func(self)

__all__ = [name for name, x in locals().items() if
           not inspect.ismodule(x) and not inspect.isabstract(x)]
