"""A simple Google-style logging wrapper.

Adopted from: https://github.com/benley/python-glog.

This library attempts to greatly simplify logging in Python applications.
Nobody wants to spend hours pouring over the PEP 282 logger documentation, and
almost nobody actually needs things like loggers that can be reconfigured over
the network.  We just want to get on with writing our apps.

## Core benefits

* You and your code don't need to care about how logging works. Unless you want
  to, of course.

* No more complicated setup boilerplate!

* Your apps and scripts will all have a consistent log format, and the same
  predictable behaviours.

* It is designed to work with `akid`, so akid can centralize control of
  logging, by indicating `akid_logger` as True when calling `init`.

## Behaviours

* No default behavior, since in python logging, once a handler is added, it
  cannot be removed.

* Messages are always written to `stderr` if `glog.init()` is used without any
  arugment.

* By calling `glog.init(FILE_NAME)`, where FILE_NAME is a `str`, logs will be
  saved to that file. Target files only need to be initialized once and could
  be shared anywhere. Repeated initialization is supported, and all logs will
  be added to that file only once.

* Calling `glog.init("stderr")` or `glog.init("stdout")` will make glog log to
  standard error or standard output.

* Lines are prefixed with a google-style log prefix, of the form

      E0924 22:19:15.123456 19552 filename.py:87] Log message blah blah

  Splitting on spaces, the fields are:

    1. The first character is the log level, followed by MMDD (month, day)
    2. HH:MM:SS.microseconds
    3. Process ID
    4. basename_of_sourcefile.py:linenumber]
    5. The body of the log message.


## Example use

    import glog as log

    log.info("It works.")
    log.warn("Something not ideal")
    log.error("Something went wrong")
    log.fatal("AAAAAAAAAAAAAAA!")

If your app uses gflags, it will automatically gain a --verbosity flag. In
order for that flag to be effective, you must call log.init() after parsing
flags, like so:

    import sys
    import gflags
    import glog as log

    FLAGS = gflags.FLAGS

    def main():
      log.debug('warble garble %s', FLAGS.verbosity)

    if __name__ == '__main__':
        posargs = FLAGS(sys.argv)
        log.init()
        main(posargs[1:])

Happy logging!
"""
from __future__ import absolute_import
import sys
import logging
import time
import gflags
import types
import os

from .tools import currentframe

gflags.DEFINE_integer('verbosity', logging.INFO, 'Logging verbosity.',
                      short_name='v')
FLAGS = gflags.FLAGS
file_names = []


def format_message(record):
    try:
        record_message = '%s' % (record.msg % record.args)
    except TypeError:
        record_message = record.msg
    return record_message


class GlogFormatter(logging.Formatter):
    LEVEL_MAP = {
        logging.FATAL: 'F',
        logging.ERROR: 'E',
        logging.WARN: 'W',
        logging.INFO: 'I',
        logging.DEBUG: 'D'
    }

    def __init__(self):
        logging.Formatter.__init__(self)

    def format(self, record):
        try:
            level = GlogFormatter.LEVEL_MAP[record.levelno]
        except:
            level = '?'
        date = time.localtime(record.created)
        date_usec = (record.created - int(record.created)) * 1e6
        record_message = '%c%02d%02d %02d:%02d:%02d.%06d %s %s:%d] %s' % (
            level, date.tm_mon, date.tm_mday, date.tm_hour, date.tm_min,
            date.tm_sec, date_usec,
            record.process if record.process is not None else '?????',
            record.filename,
            record.lineno,
            format_message(record))
        record.getMessage = lambda: record_message
        return logging.Formatter.format(self, record)

logger = None


def setLevel(newlevel):
    logger.setLevel(newlevel)
    FLAGS.verbosity = newlevel
    logger.debug('Log level set to %s', newlevel)


def init(filename=None, akid_logger=False):
    """
    If `akid_logger` is True, the logger is used for logging in `akid`, in
    which case the frame will be traced back to the caller of `Block.log` or
    any `log` of its subclasses.
    """
    global logger, debug, info, warning, warn, error, exception, fatal, log

    if logger:
        return

    logger = logging.getLogger("glog")

    if akid_logger:
        def findCaller(self, *args, **kwargs):
            """
            Find the stack frame of the caller so that we can note the source
            file name, line number and function name.
            """
            f = currentframe()
            #On some versions of IronPython, currentframe() returns None if
            #IronPython isn't run with -X:Frames.
            if f is not None:
                f = f.f_back
            rv = "(unknown file)", 0, "(unknown function)"
            while hasattr(f, "f_code"):
                co = f.f_code
                filename = os.path.normcase(co.co_filename)
                # Backtrack to akid
                if "akid" not in filename:
                    f = f.f_back
                    continue
                # Backtrack to methods that are not called `log`
                if co.co_name == "log":
                    f = f.f_back
                    continue
                rv = (co.co_filename, f.f_lineno, co.co_name, None)
                break
            return rv

        logger.findCaller = types.MethodType(findCaller, logger)

    logger.propagate = False

    add(filename)
    setLevel(FLAGS.verbosity)

    debug = logger.debug
    info = logger.info
    warning = logger.warning
    warn = logger.warning
    error = logger.error
    exception = logger.exception
    fatal = logger.fatal
    log = logger.log


def add(filename):
    """
    Add file to write the log for.
    """
    if filename is None:
        if "stderr" not in file_names:
            handler = logging.StreamHandler()
            filename = "stderr"
    elif filename in file_names:
        # Do not add files that already has been added.
        return
    elif filename == "stderr":
        handler = logging.StreamHandler(sys.stderr)
    elif filename == "stdout":
        handler = logging.StreamHandler(sys.stdout)
    else:
        handler = logging.FileHandler(filename)

    file_names.append(filename)
    handler.setFormatter(GlogFormatter())
    logger.addHandler(handler)


DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
WARN = logging.WARN
ERROR = logging.ERROR
FATAL = logging.FATAL

# basicConfig = logger.basicConfig

_level_names = {
    DEBUG: 'DEBUG',
    INFO: 'INFO',
    WARN: 'WARN',
    ERROR: 'ERROR',
    FATAL: 'FATAL'
}

_level_letters = [name[0] for name in _level_names.values()]

GLOG_PREFIX_REGEX = (
    r"""
    (?x) ^
    (?P<severity>[%s])
    (?P<month>\d\d)(?P<day>\d\d)\s
    (?P<hour>\d\d):(?P<minute>\d\d):(?P<second>\d\d)
    \.(?P<microsecond>\d{6})\s+
    (?P<process_id>-?\d+)\s
    (?P<filename>[a-zA-Z<_][\w._<>-]+):(?P<line>\d+)
    \]\s
    """) % ''.join(_level_letters)
"""Regex you can use to parse glog line prefixes."""


def get_random_log_dir():
    # Naming log dir according to time if not specified.
    log_dir = "log/" + time.ctime()
    # As ':' is widely used in network protocols, replace it with '_'
    # to avoid conflict.
    log_dir = log_dir.replace(':', '_')

    return log_dir
