"""
A collection of functions that may attach to hooks of `Kid`.

Each function is expected to take a `Kid` instance as an input that is
supposed to hold all information needed.
"""
from akid import backend as A

from ..utils import glog as log


def _do_summary(kid):
    if kid.do_summary:
        if A.backend() == A.TF:
            feed_dict = kid.feed_dict
        elif A.backend() == A.TORCH:
            if kid.do_summary_on_val:
                # If doing summary on validation set, a step needs to be run to
                # update the feature maps. Tensorflow will do this
                # automatically, so forward once only for torch.
                kid.step(update=False, val=True)

            feed_dict = None

        A.summary.run_summary_op(kid.summary_op, feed_dict=feed_dict)


def on_train_log_step(kid):
    loss = kid.loss
    evals = kid.evals
    duration = kid.step_time
    step = A.get_step()

    name_to_print = [A.get_name(g) for g in kid.engine.eval()]
    eval_value_to_print = ["%0.04f" % v for v in evals]
    eval_to_print = dict(zip(name_to_print, eval_value_to_print))

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


def on_val_log_step(kid):
    if kid.do_summary:
        # Add summary.
        A.summary.add_scalar(name="Validation Loss",
                             value=kid.loss,
                             step=A.get_step())
        for i, v in enumerate(kid.evals):
            A.summary.add_scalar(
                name=A.append_suffix(A.get_name(kid.engine.eval(get_val=True)[i]), "val"),
                value=v,
                step=A.get_step())

    # Log current validation.
    name_to_print = [A.get_name(g) for g in kid.engine.eval(get_val=True)]
    eval_value_to_print = ["%0.04f" % v for v in kid.evals]
    eval_to_print = dict(zip(name_to_print, eval_value_to_print))
    log.info('  Num examples: {}  Evals : {}'.format(
        kid.sensor.source.num_val, eval_to_print))

    # Log current best validation.
    name_to_print = [A.get_name(g) + '_best'
                     for g in kid.engine.eval(get_val=True)]
    eval_value_to_print = ["%0.04f" % v for v in kid.best_val_evals]
    eval_to_print = dict(zip(name_to_print, eval_value_to_print))
    log.info('Current best evals : {}'.format(eval_to_print))

    # Loss.
    log.info('  Step %d: Validation loss = %.2f' % (A.get_step(), kid.loss))


def on_train_begin(kid):
    # Calculate and log total parameter number.
    total_parameters = 0
    for variable in kid.brain.get_filters():
        shape = A.get_shape(variable)
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim
        total_parameters += variable_parametes
    log.info("Total parameters: {}".format(total_parameters))

    # Set up summary ops
    kid.setup_summary()

    # Run ops once to show initial training loss and save initial
    # summaries.
    kid.loss, kid.evals = kid.run_step(update=False)

    _do_summary(kid)

    name_to_print = [A.get_name(g) for g in kid.engine.eval()]
    eval_value_to_print = ["%0.04f" % v for v in kid.evals]
    eval_to_print = dict(zip(name_to_print, eval_value_to_print))
    log.info("Step {}: loss = {:.5f} eval = {}".format(
        A.get_step(), kid.loss, eval_to_print))

    # Initial validation
    kid.loss, kid.evals = kid.validate()
    kid.on_val_log_step()


def on_batch_begin(kid):
    pass


def on_epoch_end(kid):
    log.info("Epoch {} finished.".format(kid.epoch))
