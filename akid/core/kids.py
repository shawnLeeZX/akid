"""
This module contains `Kid` class to play the survival game.
"""
from __future__ import absolute_import, division, print_function

import os
import time
import sys
import inspect
from tqdm import tqdm
import akid as K

from ..utils import glog as log
from . import sensors
from . import engines
from .blocks import Block
from .. import backend as A
from .events import EarlyStoppingEvent
from .common import (
    TRAIN_SUMMARY_COLLECTION,
    VALID_SUMMARY_COLLECTION,
    TRAINING_DYNAMICS_COLLECTION,
    DEFAULT_COLLECTION,
    BATCH_MONITORING_COLLECTION
)
from .eval_blocks import BatchEvalBlock
from six.moves import range


def scavenger(f):
    """
    Decorator to clean up when exception occurs.
    """
    def robust_f(self, *args, **kwargs):
        try:
            return f(self, *args, **kwargs)
        except Exception as e:
            self.teardown()
            A.reset()
            raise e
    return robust_f


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
                 engine={"name": "single"},
                 max_steps=None,
                 max_epoch=None,
                 log_dir=None,
                 log_to_file=True,
                 val_log_step=1000,
                 train_log_step=100,
                 log_by_epoch=False,
                 num_epoch_per_log=1,
                 log_by_step=True,
                 save_chk_point=True,
                 continue_from_chk_point=False,
                 inference_mode=False,
                 do_summary=True,
                 do_summary_on_val=False,
                 skip_validation=False,
                 do_batch_monitoring=False,
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
            val_log_step: int
                After how many steps evaluation on the validation dataset
                should be taken.
            train_log_step: int
                After how many steps training statistics should be logged.
            log_by_epoch: bool
                Whether to log at the end of each epoch.
            num_epoch_per_log: int
                Log per `epoch_num_per_log` number of epochs.
            log_by_step: bool
                Whether to log at every `val_log_step`.
            inference_mode: bool
                If the kid is set up as inference mode, stuffs related to
                training will not be built.
            do_summary: Boolean
                If False, no tensorboard summaries will be saved at all.
            do_summary_on_val: Boolean
                Whether to collect summary on the validation brain. This flag
                has no effect when `do_summary` is False. This option is for
                debugging purpose. It is useful to see how summary in
                activation of validation brain is different from training
                brain. However, if such option is used, accuracy on validation
                set will become inaccurate. This is because input from
                validation source is needed when doing summaries on validation
                brain, which would make validation data be used. Since
                validation source will reshuffle data after one epoch is
                finished, some validation may be reused and some may not be
                seen at all when doing the actual validation.
            skip_validation: bool
                If True, would not do validation during training.
            do_batch_monitoring: bool
                Whether to do summary on a specific batch throughout
                training. If you decide to switch this option on, you also need
                to manually add summary ops when building the network to the
                collection `BATCH_MONITORING_COLLECTION`, otherwise, nothing
                would be done.
            Other args are self-evident.
        """
        self.sensor = sensor_in
        self.brain = brain_in
        self.kongfu = kongfu_in
        self.engine_para = engine
        self.debug = debug

        # Set up logging facilities.
        if log_dir is None:
            self.log_dir = log.get_random_log_dir()
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
        self.log_by_epoch = log_by_epoch
        self.num_epoch_per_log = num_epoch_per_log
        self.log_by_step = log_by_step
        self.inference_mode = inference_mode
        self.do_summary_on_val = do_summary_on_val
        self.do_summary = do_summary
        self.save_chk_point = save_chk_point
        self.continue_from_chk_point = continue_from_chk_point
        self.skip_validation = skip_validation
        self.do_batch_monitoring = do_batch_monitoring

        # Log training dynamics
        self.loss_data_train = []
        self.eval_data_train = []

        self.loss_data_val = []
        self.eval_data_val = []

        self.initialized = False

        # Set up hooks.
        class hooks(object):
            def __init__(self):
                self.on_training_log_step = []
                self.on_val_log_step = []
                self.on_train_begin = []
                self.on_batch_begin = []
                self.on_epoch_end = []
                self.add_default_hooks()

            def add_default_hooks(self):
                from .callbacks import on_train_log_step
                self.on_training_log_step.append(on_train_log_step)

                from .callbacks import on_val_log_step
                self.on_val_log_step.append(on_val_log_step)

                from .callbacks import on_train_begin
                self.on_train_begin.append(on_train_begin)

                from .callbacks import on_batch_begin
                self.on_batch_begin.append(on_batch_begin)

                from .callbacks import on_epoch_end
                self.on_epoch_end.append(on_epoch_end)

        self.hooks = hooks()

        # Class members whose value depends on the state of the class.
        self.loss = None
        self.evals = None
        self.best_val_evals = None

    @scavenger
    def validate(self, mode="val"):
        """Evaluating on validation set.

        Args:
            mode: str
                If "val", validate on validation set; if "test", validate on
                test set.

        Return:
            loss: float
                The validation loss. Note that though only loss is returned,
                `Kid`'s reference to evaluation metrics and loss are both
                updated.
        """
        self.init()

        A.check_mode(mode)

        if mode == A.Mode.VAL:
            self.log('Validation Data Eval:')
        elif mode == A.Mode.TEST:
            self.log('Test Data Eval:')
        elif mode == A.Mode.TRAIN:
            self.log('Train Data Eval:')
        else:
            assert False, "Program should not reach here."

        if not self.sensor.mode == mode:
            self.sensor.set_mode(mode)

        # We want to traverse the dataset, so reset to reset the sampler.
        self.sensor.reset()

        # Run one epoch of eval.
        self.log("A epoch of {} set contains {} batches. Batch size {}.".format(mode, self.sensor.num_batches_per_epoch, self.sensor.batch_size))
        eval_blocks = [BatchEvalBlock()
                       if K.get_eval_block(A.get_name(v, no_scope=True)) is None
                       else K.get_eval_block(A.get_name(v, no_scope=True))()
                       for v in self.engine.eval(get_val=True)]
        if self.engine.verbose_eval(get_val=True) is not None:
            verbose_eval_blocks = [BatchEvalBlock()
                                   if K.get_eval_block(A.get_name(v, no_scope=True)) is None
                                   else K.get_eval_block(A.get_name(v, no_scope=True))()
                                   for v in self.engine.verbose_eval(get_val=True)]
        else:
            verbose_eval_blocks = None
        loss_block = BatchEvalBlock()
        steps_per_epoch = self.sensor.num_batches_per_epoch

        for step in tqdm(list(range(1, steps_per_epoch+1))):
            if verbose_eval_blocks is None:
                loss, evals = self.run_step(update=False, val=True)
            else:
                loss, evals, verbose_evals = self.run_step(update=False, val=True)

            loss_block.add(loss)
            for i, v in enumerate(evals):
                eval_blocks[i].add(v)
            if verbose_eval_blocks is not None:
                for i, v in enumerate(verbose_evals):
                    verbose_eval_blocks[i].add(v)

        # Note that the EvalBlocks become numeric data now.
        loss_avg = loss_block.data
        for i, v in enumerate(eval_blocks):
            eval_blocks[i] = v.data
        if verbose_eval_blocks is not None:
            for i, v in enumerate(verbose_eval_blocks):
                verbose_eval_blocks[i] = v.data

        self.loss = loss_avg
        self.evals = eval_blocks
        self.verbose_evals = verbose_eval_blocks

        return loss_avg, eval_blocks

    def reset(self):
        self.sensor.reset()
        self.sensor.setup()

    @scavenger
    def setup(self):
        """
        Set up logging and the computation graph.
        """
        self._setup_log()

        if A.backend() == A.TORCH and self.continue_from_chk_point:
            log.info("Recovering net from checkpoint %s." % self.model_dir)
            self.restore()

        if self.sensor.do_summary is None:
            self.sensor.set_do_summary_flag(self.do_summary)
        if self.sensor.do_summary_on_val is None:
            self.sensor.set_do_summary_on_val_flag(self.do_summary_on_val)
        if self.brain.do_summary is None:
            self.brain.set_do_summary_flag(self.do_summary)
        if self.brain.do_summary_on_val is None:
            self.brain.set_do_summary_on_val_flag(self.do_summary_on_val)

        self.sensor.setup()
        self.engine = engines.get(brain=self.brain, kongfu=self.kongfu, **self.engine_para)
        self.engine.setup()

        # Do forward once to build ops.

        self.step(update=False)
        # Build validation ops
        self.sensor.set_mode("val")
        self.sensor.setup()
        self.step(update=False, val=True)

    def teardown(self):
        self.sensor.teardown()

    def restore(self):
        A.restore(self.model_dir)
        if A.backend() ==  A.TORCH and not self.inference_mode:
            self.kongfu.set_lr(A.retrieve_tensor('lr'))

    def batch_monitoring(self, data):
        """
        A method that creates summary ops to visualize a specific batch
        throughout training to monitor the progress of training.

        Args:
            data:
                The specific data batch to monitor.
        """
        # Propagate the sample through network.
        # NOTE: the following code is PyTorch specific.
        self.brain.switch_batch_monitoring_mode()
        self.step(update=False, val=False, data=data)

        # Create summary ops if not already, and run them.
        if not hasattr(self, "batch_monitoring_summary_ops"):
            self.batch_monitoring_summary_ops = A.summary.get_collection(BATCH_MONITORING_COLLECTION)
        A.summary.run_summary_op(self.batch_monitoring_summary_ops)

        # Clean up the monitoring tensors created to save memory.
        # A.remove_variable_contains_str(self.brain.name)

        self.brain.switch_batch_monitoring_mode()

        if not self.brain.done_first_batch_monitoring_pass:
            self.brain.set_flag("done_first_batch_monitoring_pass", True)


    @scavenger
    def practice(self, return_eval=False):
        """
        Improve the performance of the kid's brain by practicing, aka
        applying back propagation to train neural network.

        Args:
            return_eval: bool
                Return evaluation metric after trainning is finished.
        Return:
            loss, [eval]: float, float
                Final validation loss and optional evaluation metric.
        """
        self.log("Begin training brain: " + self.brain.name)
        self.sensor.set_mode("train")
        self.sensor.setup()
        self.log("A epoch of training set contains {} batches".format(self.sensor.num_batches_per_epoch))
        self.epoch = A.get_step() // self.sensor.num_batches_per_epoch

        self.init()
        self.on_train_begin()
        # Reset sensor, so to prevent any code from using some training batches
        # in advance, and making them missing from initial training.
        self.sensor.reset()

        val_loss, val_evals = None, None
        while A.get_step() < self.max_steps:
            try:
                val_loss, val_evals = self.step_with_logistics()
            except EarlyStoppingEvent as e:
                val_loss, val_evals = e.val_loss, e.val_evals
                break

        if return_eval:
            return val_loss, val_evals
        else:
            return val_loss

    @scavenger
    def predict(self, data):
        """
        Given a datum, do inference with a NN. Note that we do not have format
        requirement on `data`, since it should be taken care when building the
        NN. If input validation is needed, it should be done in the layer that
        calls this API.
        """
        self.init()
        self.engine.set_val(True)
        self.engine.forward(data, val=True)

        evals = [e for e in self.engine.eval(get_val=True)]
        if self.engine.verbose_eval(get_val=True) is not None:
            verbose_evals = [e for e in self.engine.verbose_eval(get_val=True)]
        else:
            verbose_evals = None

        return evals, verbose_evals

    def step_with_logistics(self):
        val_loss = None
        val_evals = None

        start_time = time.time()

        self.on_batch_begin()
        self.loss, self.evals = self.run_step()

        if self.debug:
            print("Loss: {}; Eval: {}".format(self.loss, self.evals))

        self.step_time = time.time() - start_time

        A.step()

        if A.get_step() % self.train_log_step == 0:
            # Since during logging, loss and evals are passed in as attributes
            # of Kid, training log is put ahead of validation log to prevent
            # validation loss and evals from overriding the ones of training,
            # when the validation logging step and training logging step
            # coincide.
            self.on_train_log_step()

        if self.log_by_step and \
            (A.get_step() % self.val_log_step == 0 or\
            A.get_step() == self.max_steps):
            if self.save_chk_point:
                self.save_to_ckpt()
            if not self.skip_validation:
                self.loss, self.evals = self.validate()
            val_loss, val_evals = self.loss, self.evals
            self.on_val_log_step()
            self.sensor.set_mode("train")
            self.sensor.setup()

        if A.get_step() % self.sensor.num_batches_per_epoch is 0:
            self.epoch += 1
            self.on_epoch_end()
            if self.log_by_epoch \
               and A.get_step() % (self.sensor.num_batches_per_epoch * self.num_epoch_per_log) == 0:
                if self.save_chk_point:
                    self.save_to_ckpt()
                if not self.skip_validation:
                    self.loss, self.evals = self.validate()
                self.on_val_log_step()
                val_loss, val_evals = self.loss, self.evals
                self.sensor.set_mode("train")
                self.sensor.setup()

        return val_loss, val_evals

    def tf_step(self, update=True, val=False):
        self.sensor.set_val(val)
        self.feed_dict = self.get_feed_dict(val)
        fetch = {"loss": self.engine.loss(val)}
        fetch["eval"] = self.engine.eval(val)
        if val:
            # TODO: may be broken. could dict element be list?
            fetch["verbose_eval"] = self.engine.verbose_eval(val)
        if update:
            fetch["train_op"] = self.engine.train_op
        result = A.run(fetch, feed_dict=self.feed_dict)

        return result["loss"], result["eval"], result["verbose_eval"] if val else None

    def run_step(self, update=True, val=False, data=None):
        """
        Computation wise, execute the computational graph for a step.
        """
        if A.backend() == A.TF:
            return self.tf_step(update, val)
        elif A.backend() == A.TORCH:
            loss, evals, verbose_evals = self.step(update, val, data=data)
            loss = A.eval(loss)
            evals = A.eval(evals)
            if verbose_evals is not None and val:
                verbose_evals = A.eval(verbose_evals)
                return loss, evals, verbose_evals
            else:
                return loss, evals

    def step(self, update=True, val=False, data=None):
        """
        Computational graph wise, how the tensor should be run in a step.

        If `data` is None, the caller is supposed to feed in data. In such a
        case, the control on the sensor is left to the caller; that is, kid
        would not modify it after setup.
        """
        if data is None:
            system_in = self.sensor.forward()
        else:
            system_in = data

        self.engine.set_val(val)
        self.engine.forward(system_in, val)
        if update:
            self.engine.update()

        loss = self.engine.loss(val)
        evals = [e for e in self.engine.eval(val)]
        if val and self.engine.verbose_eval(val) is not None:
            verbose_evals = [e for e in self.engine.verbose_eval(val)]
        else:
            verbose_evals = None

        return loss, evals, verbose_evals

    def _setup_log(self):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        A.init_log()
        if self.log_to_file:
            log.add(self.log_filepath)
        if self.debug:
            log.setLevel(log.DEBUG)
        else:
            log.setLevel(log.INFO)
        self.log("Logs will be save to: {}".format(self.log_dir))

    def init(self):
        """
        Initialize the backend if having not.
        """
        if not self.initialized:
            if A.backend() == A.TF:
                # The checkpoint recovery has to be here, given that it needs to be
                # after the computational graph being set up.
                A.init(self.continue_from_chk_point, self.model_dir)
                self.initialized = True
            elif A.backend() == A.TORCH:
                A.init() # This actually does nothing.
                self.initialized = True
            else:
                raise ValueError("Backend {} not supported".format(A.backend()))

    def setup_summary(self):
        if self.do_summary:
            # SummaryWriter to output summaries and the Graph.
            A.summary.init(self.log_dir)

            summary_ops = A.summary.get_collection(TRAIN_SUMMARY_COLLECTION)
            summary_ops.extend(A.summary.get_collection(
                TRAINING_DYNAMICS_COLLECTION))
            summary_ops.extend(A.summary.get_collection(
                DEFAULT_COLLECTION))
            if self.do_summary_on_val:
                val_summary_ops = A.summary.get_collection(
                    VALID_SUMMARY_COLLECTION)
                summary_ops.extend(val_summary_ops)
            self.summary_op = A.summary.merge(summary_ops)
            A.summary.add_graph()

    def save_to_ckpt(self):
        # For torch background, the learning rate will be used during setup
        # phase, so it should be saved.
        if A.backend() == A.TORCH:
            A.cache_tensor(self.kongfu.get_lr(), 'lr')
        A.save(self.model_dir)
        self.log("Checkpoint at step {} saved to folder:"
                 " {}".format(A.get_step(), self.model_dir))

    def get_feed_dict(self, val=False):
        feed_dict = None
        if type(self.sensor) is sensors.FeedSensor:
            # Placeholder of `FeedSensor` should be filled.
            feed_dict = self.sensor.fill_feed_dict()
            # If val is True, we have gotten val data already.
            if not val and self.do_summary and self.do_summary_on_val:
                # Validation data is also needed, so add them in.
                self.sensor.set_val(True)
                val_feed_dict = self.sensor.fill_feed_dict()
                self.sensor.set_val(False)
                feed_dict.update(val_feed_dict)

        lr_dict = self.kongfu.get_feed_dict()

        if feed_dict is not None:
            feed_dict.update(lr_dict)
        else:
            feed_dict = lr_dict

        return feed_dict

    def on_train_log_step(self):
        """
        Call hooks at the time when the kid should do logging for training.
        """
        for func in self.hooks.on_training_log_step:
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

        for func in self.hooks.on_val_log_step:
            func(self)

    def on_train_begin(self):
        if self.max_epoch:
            # Convert the max epoch number to max steps.
            self.max_steps \
                = self.sensor.num_batches_per_epoch * self.max_epoch

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
