"""
A collection of functions that may attach to hooks of `Kid`.

Each function is expected to take a `Kid` instance as an input that is
supposed to hold all information needed.
"""
from __future__ import absolute_import
from akid import backend as A
import akid as K
from akid import ops

from ..utils import glog as log
from ..utils.tools import is_tuple_or_list
from six.moves import zip


def _do_summary(kid):
    if kid.do_summary:
        if A.backend() == A.TF:
            feed_dict = kid.feed_dict
        elif A.backend() == A.TORCH:
            if kid.do_summary_on_val:
                kid.sensor.set_mode("val")
                kid.sensor.setup()
                # If doing summary on validation set, a step needs to be run to
                # update the feature maps. Tensorflow will do this
                # automatically, so forward once only for torch.
                kid.step(update=False, val=True)
                kid.sensor.set_mode("train")
                kid.sensor.setup()

            feed_dict = None

        A.summary.run_summary_op(kid.summary_op, feed_dict=feed_dict)


def v_to_text(name_list, v_list):
    """
    Convert named tensors to text to print. v is a list of tensors.
    """
    return [K.get_eval_block(name_list[i]).__str__(None, v) for i, v in enumerate(v_list)]


def on_train_log_step(kid):
    loss = kid.loss
    evals = kid.evals
    duration = kid.step_time
    step = A.get_step()

    kid.loss_data_train.append(loss)
    kid.eval_data_train.append(evals)

    name_to_print = [A.get_name(g, no_scope=True) for g in kid.engine.eval()]
    eval_value_to_print = v_to_text(name_to_print, evals)
    eval_to_print = dict(list(zip(name_to_print, eval_value_to_print)))

    num_examples_per_step = kid.sensor.batch_size
    examples_per_sec = num_examples_per_step / duration
    sec_per_batch = float(duration)

    lr = kid.kongfu.get_lr()

    log.info("Step {}: loss = {:.5f} lr = {:.8f} acc = {} ({:.1f}"
             " examples/sec {:.3f} sec/batch)".format(
                 step,
                 loss,
                 lr,
                 eval_to_print,
                 examples_per_sec,
                 sec_per_batch))

    _do_summary(kid)

    if kid.do_batch_monitoring:
        kid.batch_monitoring(kid.cached_data)


def on_val_log_step(kid):
    if kid.do_summary:
        # Add summary.
        A.summary.add_scalar(name="Validation Loss",
                             value=kid.loss,
                             step=A.get_step())
        for i, v in enumerate(kid.evals):
            if A.is_numerical(v):
                A.summary.add_scalar(
                    name=A.append_suffix(A.get_name(kid.engine.eval(get_val=True)[i]), "val"),
                    value=v,
                    step=A.get_step())
            elif is_tuple_or_list(v):
                for j, u in enumerate(v):
                    A.summary.add_scalar(
                        name=A.append_suffix(A.get_name(kid.engine.eval(get_val=True)[i][j]), "val_{}".format(i)),
                        value=u,
                        step=A.get_step())
        if kid.verbose_evals is not None:
            for i, v in enumerate(kid.verbose_evals):
                if A.is_numerical(v):
                    A.summary.add_scalar(
                        name=A.append_suffix(A.get_name(kid.engine.verbose_eval(get_val=True)[i]), "vval"),
                        value=v,
                        step=A.get_step())
                elif is_tuple_or_list(v):
                    for j, u in enumerate(v):
                        A.summary.add_scalar(
                            name=A.append_suffix(A.get_name(kid.engine.verbose_eval(get_val=True)[i]), "vval_{}".format(j)),
                            value=u,
                            step=A.get_step())

    # Log current validation.
    name_to_print = [A.get_name(g, no_scope=True) for g in kid.engine.eval(get_val=True)]
    eval_value_to_print = v_to_text(name_to_print, kid.evals)
    eval_to_print = dict(list(zip(name_to_print, eval_value_to_print)))
    if kid.verbose_evals is not None:
        name_to_print = [A.get_name(g, no_scope=True) for g in kid.engine.verbose_eval(get_val=True)]
        veval_value_to_print =  v_to_text(name_to_print, kid.verbose_evals)
        veval_to_print = dict(list(zip(name_to_print, veval_value_to_print)))
    else:
        veval_to_print = None
    log.info('  Num examples: {}  Evals : {}  VEvals: {}'.format(
        kid.sensor.source.size, eval_to_print, veval_to_print))

    # Log current best validation.
    name_to_print = [A.get_name(g, no_scope=True) + '_best'
                     for g in kid.engine.eval(get_val=True)]
    eval_value_to_print = v_to_text([A.get_name(g, no_scope=True) for g in kid.engine.eval(get_val=True)],
                                    kid.best_val_evals)
    eval_to_print = dict(list(zip(name_to_print, eval_value_to_print)))
    log.info('Current best evals : {}'.format(eval_to_print))

    # Loss.
    log.info('  Step %d: Validation loss = %.6f' % (A.get_step(), kid.loss))

    kid.loss_data_val.append(kid.loss)
    kid.eval_data_val.append(kid.evals)


def on_train_begin(kid):
    # Calculate and log total parameter number.
    total_parameters = kid.brain.num_parameters()
    log.info("Total parameters: {}".format(total_parameters))

    # Set up summary ops
    kid.setup_summary()

    # Run ops once to show initial training loss and save initial
    # summaries.
    kid.loss, kid.evals = kid.run_step(update=False)

    _do_summary(kid)

    name_to_print = [A.get_name(g, no_scope=True) for g in kid.engine.eval()]
    eval_value_to_print = v_to_text(name_to_print, kid.evals)
    eval_to_print = dict(list(zip(name_to_print, eval_value_to_print)))
    log.info("Step {}: loss = {:.5f} eval = {}".format(
        A.get_step(), kid.loss, eval_to_print))

    kid.loss_data_train.append(kid.loss)
    kid.eval_data_train.append(kid.evals)

    # Cache the current batch as the monitoring batch if the flag is set.
    if kid.do_batch_monitoring:
        kid.cached_data = kid.sensor.data
        kid.batch_monitoring(kid.cached_data)

    # Initial validation
    if not kid.skip_validation:
        kid.loss, kid.evals = kid.validate()
        kid.on_val_log_step()
        kid.sensor.set_mode("train")
        kid.sensor.setup()


def on_batch_begin(kid):
    pass


def on_epoch_end(kid):
    log.info("Epoch {} finished.".format(kid.epoch))
