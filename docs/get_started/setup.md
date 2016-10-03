# Download and Setup

Currently, `akid` only supports installing from the source.

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
