# Download and Setup

Currently, `akid` only supports installing from the source.

## Dependency

`akid` depends on some regular numerical libraries. If you are using `pip`, you
could install them as the following:

```bash
pip install numpy, scipy, matplotlib, gflags
```

Follow the official installation
[guide](https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html) to
install tensorflow.

## Install from the source

### Clone the repository

```bash
git clone https://github.com/shawnLeeZX/akid
```

## Post installation setup

### Environment Variables

If you want to use the dataset automatically download feature, an environment
variable needs to be set. Add the following line to `.bashrc`, or other
configuration files of your favorite shell.

```bash
AKID_DATA_PATH=  # where you want to store data
```

Also, remember to make `akid` visible by adding the folder that contains `akid`
to `PYTHONPATH`.
