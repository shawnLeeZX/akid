"""
This module contains `Survivor` class to play the survival game.
"""
from __future__ import absolute_import, division, print_function

import os
import time
import sys
import inspect

import tensorflow as tf

from ..utils import glog as log
from . import sensors
from .common import GLOBAL_STEP, global_var_scope
from .common import (
    TRAIN_SUMMARY_COLLECTION,
    VALID_SUMMARY_COLLECTION,
    TRAINING_DYNAMICS_COLLECTION,
    LEARNING_RATE_TAG
)


class Survivor(object):
    """
    Survivor is a class to assemble a `Sensor`, for supplying data, a
    `Brain`, for data processing, and a genre of `KongFu`, for algorithms or
    polices to train.

    # TODO(Shuai): I guess when the time comes to run on multiple GPUs or
    # machines, an class need to be abstracted out to deal with the
    # computational unit, compared with `Block`. `Survivor` is the start point.
    """
    def __init__(self,
                 sensor_in,
                 brain_in,
                 kongfu_in,
                 log_dir=None,
                 log_to_file=True,
                 max_steps=20000,
                 val_step=1000,
                 train_step=100,
                 graph=None,
                 do_summary=True,
                 summary_on_val=False):
        """
        Assemble a sensor, a brain, and a KongFu to start the survival game.

        Args:
            sensor: Sensor
                A `Sensor` class to supply data.
            brain: Brain
                A `Brain` class to process data.
            kongfu: KongFu
                A `KongFu` class for training.
            log_dir: str
                The folder to hold tensorboard event, training logs and trained
                models. If not given, first a folder named `log` will be
                created, then a folder named by current time stamp will be the
                folder to keep all the mentioned files.
            log_to_file: Boolean
                Whether to save log to file.
            graph: tf.Graph()
                The computational graph this survivor is in. If not given, a
                new graph will be created.
            val_step: int
                After how many steps evaluation on the validation dataset
                should be taken.
            train_step: int
                After how many steps training statistics should be logged.
            do_summary: Boolean
                If False, no tensorboard summaries will be saved at all. Note
                that if `Brain` or `Sensor`'s `do_summary` option is True, they
                will not be unset. Though during training, they would not be
                used. This makes possible that objects other than this survivor
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
        self.train_step = train_step
        self.val_step = val_step
        self.summary_on_val = summary_on_val
        self.do_summary = do_summary

        # A tensorflow computational graph to hold training and validating
        # graphs.
        if not graph:
            self.graph = tf.Graph()
        else:
            self.graph = graph

    def validate(self, sess):
        """Evaluating on validation set.

        Args:
            sess: The session where validation graph has been built.

        Return:
            precision: float
                The prediction accuracy.
        """
        log.info('Validation Data Eval:')
        # Run one epoch of eval.
        eval_metric_values = [0] * len(self.val_brain.eval_graph_list)
        loss = 0
        steps_per_epoch = self.sensor.num_batches_per_epoch_val
        num_examples = steps_per_epoch * self.sensor.val_batch_size
        for step in xrange(steps_per_epoch):
            if type(self.sensor) is sensors.FeedSensor:
                feed_dict = self.sensor.fill_feed_dict(get_val=True)
            else:
                feed_dict = None
            fetch = [self.val_brain.loss_graph]
            fetch.extend(self.val_brain.eval_graph_list)
            result = sess.run(fetch, feed_dict=feed_dict)
            loss += result[0]
            for i, v in enumerate(result[1:]):
                eval_metric_values[i] += v
        loss /= steps_per_epoch
        for i, v in enumerate(eval_metric_values):
            eval_metric_values[i] = v / steps_per_epoch

        current_step = tf.train.global_step(sess, self.global_step_tensor)
        if self.do_summary:
            # Add summary.
            summary = tf.Summary()
            summary.value.add(tag="Validation Loss", simple_value=loss)
            for i, v in enumerate(eval_metric_values):
                summary.value.add(
                    tag=self.val_brain.eval_graph_list[i].op.name,
                    simple_value=v)
            summary.value.add(
                tag=LEARNING_RATE_TAG,
                simple_value=float(self.kongfu.learning_rate.eval()))
            self.summary_writer.add_summary(summary, current_step)
        # Log.
        name_to_print = [g.op.name for g in self.val_brain.eval_graph_list]
        eval_value_to_print = ["%0.04f" % v for v in eval_metric_values]
        eval_to_print = dict(zip(name_to_print, eval_value_to_print))
        log.info('  Num examples: {}  Evals : {}'.format(
            num_examples, eval_to_print))
        log.info('  Step %d: Validation loss = %.2f' % (current_step, loss))

        return loss

    def setup(self):
        """
        Set up logging and the computation graph.
        """
        with self.graph.as_default():
            self._setup_log()
            self._setup_sensor()
            self._setup_brain()
            self._setup_val_brain()
            self._setup_summary()
            self.saver = tf.train.Saver(tf.all_variables())

    def practice(self, sess=None, continue_from_chk_point=False):
        """
        Improve the performance of the survivor's brain by practicing, aka
        applying back propagation to train neural network.

        Args:
            sess: tf.Session()
                A session to launch the training. If not given, a default one
                will be created inside.
            continue_from_chk_point: Boolean
                Setup configuration. Passed to `setup`
        Return:
            None
        """
        if not sess:
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth=True
            sess = tf.Session(graph=self.graph, config=config)
            with sess:
                return self._practice(sess, continue_from_chk_point)
        else:
            return self._practice(sess, continue_from_chk_point)

    def _practice(self, sess, continue_from_chk_point):
        try:
            self._init(sess, continue_from_chk_point)
            # And then after everything is built, start the training loop.
            log.info("Begin training brain: " + self.brain.name)
            previous_step = tf.train.global_step(sess,
                                                 self.global_step_tensor)
            # Do one validation before beginning.
            self.save_to_ckpt(sess)
            self.validate(sess)
            # Run ops once to show initial training loss and save initial
            # summaries.
            if type(self.sensor) is sensors.FeedSensor:
                # Placeholder of `FeedSensor` should be filled.
                feed_dict = self.sensor.fill_feed_dict()
            else:
                feed_dict = None

            fetch = [self.brain.loss_graph]
            fetch.extend(self.brain.eval_graph_list)
            result = sess.run(fetch, feed_dict=feed_dict)
            loss_value = result[0]
            if self.do_summary:
                summary = tf.Summary()
                summary.value.add(tag="Training Loss",
                                  simple_value=float(loss_value))
                self.summary_writer.add_summary(summary, previous_step)
                if type(self.sensor) is sensors.FeedSensor \
                   and self.summary_on_val:
                    val_feed_dict = self.sensor.fill_feed_dict(True)
                    feed_dict.update(val_feed_dict)
                summary_str = sess.run(self.summary_op, feed_dict=feed_dict)
                self.summary_writer.add_summary(summary_str, previous_step)

            name_to_print = [g.op.name for g in self.brain.eval_graph_list]
            eval_value_to_print = ["%0.04f" % v for v in result[1:]]
            eval_to_print = dict(zip(name_to_print, eval_value_to_print))
            log.info("Step {}: loss = {:.5f} eval = {}".format(
                previous_step, loss_value, eval_to_print))

            for step in xrange(previous_step + 1, self.max_steps + 1):
                self._step(sess, step)

                if step % self.val_step == 0 or step == self.max_steps:
                    self.save_to_ckpt(sess)
                    precision = self.validate(sess)

            return precision
        except tf.OpError as e:
            log.info("Tensorflow error when running: {}".format(e.message))
            sys.exit(0)

    def _setup_log(self):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        log.init("stdout")
        if self.log_to_file:
            log.init(self.log_filepath)
        # TODO(Shuai): Make this a switch instead of hard coded.
        log.setLevel(log.DEBUG)
        log.info("Logs will be save to: {}".format(self.log_dir))

    def _setup_summary(self):
        if self.do_summary:
            # SummaryWriter to output summaries and the Graph.
            self.summary_writer = tf.train.SummaryWriter(self.log_dir)
            log.info("Summary event file will be saved to {}".format(
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
            self.summary_op = tf.merge_summary(summary_ops)
            # Write the brain to tensorflow event file.
            self.summary_writer.add_graph(self.graph.as_graph_def())

    def _setup_sensor(self):
        # Build training graph.
        log.info("Setting up sensor ...")
        self.sensor.setup()

    def _setup_brain(self):
        # A tensor to track training step.
        with tf.variable_scope(global_var_scope):
            self.global_step_tensor = tf.get_variable(
                name=GLOBAL_STEP,
                shape=[],
                initializer=tf.constant_initializer(0),
                trainable=False)
        # TODO(Shuai): here is a temporary solution. Since sensor is actually
        # just a system with two outputs, `GraphSystem` could handle it, but it
        # would make many changes, for now, I just settle with the not-so
        # elegant solution here. Note that if here is going to be changed,
        # `_setup_val_brain` should also be changed.
        data = self.sensor.data()
        label = self.sensor.labels()
        system_in = [data]
        system_in.extend(label) if type(label) is list \
            else system_in.append(label)
        self.brain.setup(system_in)
        self.kongfu.setup(self)

    def _setup_val_brain(self):
        self.val_brain = self.brain.get_val_copy()
        data = self.sensor.data(get_val=True)
        label = self.sensor.labels(get_val=True)
        system_in = [data]
        system_in.extend(label) if type(label) is list \
            else system_in.append(label)
        self.val_brain.setup(system_in)

    def _init(self, sess, continue_from_chk_point=None):
        """
        Initialize computational graph for training. It initializes or restores
        variables, starts queues and so on.

        Args:
            sess: tf.Session
            continue_from_chk_point: Boolean
                Continue from a previous training or not. If it is True, a
                folder named `model` must exist under `Survivor`'s `log_dir`
                with saved models.
        """

        # Initialization.
        if continue_from_chk_point:
            # Train from pre-trained model.
            self.restore_from_ckpt(sess)
        else:
            init = tf.initialize_all_variables()
            sess.run(init)

        # Start queue runner if needed.
        if type(self.sensor) is sensors.IntegratedSensor:
            tf.train.start_queue_runners(sess=sess)

    def save_to_ckpt(self, sess):
        step = tf.train.global_step(sess, self.global_step_tensor)
        self.saver.save(sess,
                        self.model_dir + "/checkpoint",
                        global_step=step)
        log.info("Checkpoint at step {} saved to folder:"
                 " {}".format(step, self.model_dir))

    def restore_from_ckpt(self, sess):
        """
        Restore variables of this net from the latest checkpoint of
        `model_dir`.

        Args:
            sess: tf.Session
                Whatever session of the caller uses this net.

        Return:
            Training step of the checkpoint the net are recovering from.
        """
        checkpoint = tf.train.get_checkpoint_state(self.model_dir)
        if checkpoint and checkpoint.model_checkpoint_path:
            log.info("Recovering net from checkpoint %s."
                     % checkpoint.model_checkpoint_path)
            self.saver.restore(sess, checkpoint.model_checkpoint_path)
            filename = checkpoint.model_checkpoint_path.split('/')[-1]
            step = int(filename.split('-')[-1])
            return step
        else:
            log.error("No checkpoint found under %s!" % self.model_dir)
            sys.exit()

    def _step(self, sess, step):
        """
        Train for one step.

        Args:
            sess: tf.Session
            step: int

        Current training step. It is used for logistics purpose, such
                as display current step in logging.

        Returns:
            loss_value: a real number. The training loss of current step.
        """
        if type(self.sensor) is sensors.FeedSensor:
            # Placeholder of `FeedSensor` should be filled.
            feed_dict = self.sensor.fill_feed_dict()
        else:
            feed_dict = None

        # Run one step.
        fetch = [self.kongfu.train_op, self.brain.loss_graph]
        fetch.extend(self.brain.eval_graph_list)
        start_time = time.time()
        result = sess.run(fetch, feed_dict=feed_dict)
        if self.brain.max_norm_clip_op:
            sess.run([self.brain.max_norm_clip_op])
        duration = time.time() - start_time
        loss_value = result[1]

        # Write the summaries and print an overview fairly often.
        if step % self.train_step == 0:
            name_to_print = [g.op.name for g in self.brain.eval_graph_list]
            eval_value_to_print = ["%0.04f" % v for v in result[2:]]
            eval_to_print = dict(zip(name_to_print, eval_value_to_print))

            num_examples_per_step = self.sensor.batch_size
            examples_per_sec = num_examples_per_step / duration
            sec_per_batch = float(duration)

            log.info("Step {}: loss = {:.5f} lr = {:.8f} acc = {} ({:.1f}"
                     " examples/sec {:.3f} sec/batch)".format(
                         step,
                         loss_value,
                         self.kongfu.learning_rate.eval(),
                         eval_to_print,
                         examples_per_sec,
                         sec_per_batch))

            if self.do_summary:
                # Update the events file.
                summary = tf.Summary()
                summary.value.add(tag="Training Loss",
                                  simple_value=float(loss_value))
                self.summary_writer.add_summary(summary, step)

                if type(self.sensor) is sensors.FeedSensor \
                   and self.summary_on_val:
                    # Validation data is also needed, so add them in.
                    val_feed_dict = self.sensor.fill_feed_dict(True)
                    feed_dict.update(val_feed_dict)
                summary_str = sess.run(self.summary_op, feed_dict=feed_dict)
                self.summary_writer.add_summary(summary_str, step)

        return loss_value


__all__ = [name for name, x in locals().items() if
           not inspect.ismodule(x) and not inspect.isabstract(x)]
