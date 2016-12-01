# Introduction

## Why `akid`

Why another package on neural network?

Neural network, or in a more general term deep learning, has resurrected as a
vibrating field that may transform society. A range of libraries has emerged,
but the technology stacks of neural network is far from it maturity yet. Neural
network is still in the state of empirical science, and products using neural
network are in a state where research and production reinforces each other. In
this regards, we would like to explore technology stacks that enable fast
research prototyping and are production ready.

`akid` tries to provide a full stack of softwares (upon available open source
libraries) that provides abstraction to let researchers focus on research
instead of implementation, while at the same time the developed program can
also be put into production seamlessly in a distributed environment, and be
production ready.

```eval_rst
.. image:: ../images/akid_stack.png
   :scale: 50 %
   :alt: alternate text
   :align: left
```

At the top application stack, it provides out-of-box tools for neural network
applications. Lower down, `akid` provides programming paradigm that lets user
easily build customized model. The distributed computing stack handles the
concurrency and communication, thus letting models be trained or deployed to a
single GPU, multiple GPUs, or a distributed environment without affecting how a
model is specified in the programming paradigm stack. Lastly, the distributed
deployment stack handles how the distributed computing is deployed, thus
decoupling the research prototype environment with the actual production
environment, and is able to dynamically allocate computing resources, so
administration ops and development ops could be separated.

This paragraph is a brief summary of available packages, including
[theano](http://deeplearning.net/software/theano/), [keras](keras.io),
[torch](http://torch.ch/), [caffe](http://caffe.berkeleyvision.org/) which
could be safely skipped if you are not interested. The debugging information is
not very nice in theano. It takes too much time to compile programs. It needs
re-compilation after kernel update. Torch lacks communities and supporting
libraries, which may be good for research only, but for production ready
libraries neural network should not stop there. Caffe is built on C++, which is
too slow to develop. Keras seems to be perfect, but mostly it is a package like
sklearn, which takes neural network as a machine learning model, which is too
math oriented. I want something that feels like a growing kid. There are also
many subtle difference with Keras.

## A narrative description of `akid`

Briefly, `akid` is a kid who has the ability to keep practicing to improve
itself.

A living being (the kid) needs to use all kinds of *sensors* it equipped to
sense a *source*, parts of the world, by a certain *way*, to accumulate
experience and summarize knowledge in ones *brain* to fulfill a basic purpose,
to *play*.

The world is run by a clock. It represents how long the kid has been practices
in the world. The clock is the conventional training step.

The data supplier, such as the data generator in `keras`, or data layer in
`Caffe`, is abstracted as a class `Sensor`. It takes a `Source` which either
provides data in form of tensor of Tensorflow or array of numpy. Optionally, it
could make use of class `Joker` to make jokes to the data from `Source`,
meaning doing data augmentation.

The data processing engine, which is a deep neural network, is abstracted as a
class `Brain`. `Brain` is the name we give to the data processing system in
living beings. A `Brain` incarnates one of data processing system topology, or
in the terminology of neural network, network structure topology, such as a
sequentially linked together layers, to process data. Available topology is
defined in module `systems`.

The network training methods, which are first order iterative optimization
methods, is abstracted as a class `KongFu`. A living being needs to keep
practicing Kong Fu to get better at tasks needed to survive. The actual
computation is done by a class `Engine`, which implement a parallel scheme (or
one lack of parallelism) to actually do the computation.

A living being is abstracted as a `Kid` class, which assemblies all above
classes together to play the survival game. The metaphor means by sensing more
examples, with certain genre of Kong Fu(different training algorithms and
policies), the data processing engine of the Survivor, the brain, should get
better at doing whatever task it is doing, letting it be image classification
or something else.

Besides, an `Observer` class could open a brain and look into it, which is to
mean visualization.

## `akid` stack

### Application

At the top of the stack, `akid` could be used as a part of application without
knowing the underlying mechanism of neural networks.

`akid` provides full machinery from preparing the data, doing data
augmentation, specifying computation graph (neural network architecture),
choosing optimization algorithms, specifying training scheme (data parallelism
etc), and information logging.

#### Neural network training --- A holistic example

This case suits the ones who are product driven --- wants to get results
quickly. It aims to provide an easy to use flow to all elements of neural
network --- training, testing, logging, parallelism schemes, etc. That is to
say, it could make use of all possible blocks in `akid`.

From a perspective of design, this is what `akid` means, explained in the
following. Also, in this section, all available blocks in `akid` are briefly
introduced.


```eval_rst
.. image:: ../images/application_illustration.png
   :alt: alternate text
```

```python
from akid import AKID_DATA_PATH
from akid import FeedSensor
from akid import Kid
from akid import MomentumKongFu
from akid import MNISTFeedSource

from akid.models.brains import LeNet

brain = LeNet(name="Brain")
source = MNISTFeedSource(name="Source",
                         url='http://yann.lecun.com/exdb/mnist/',
                         work_dir=AKID_DATA_PATH + '/mnist',
                         center=True,
                         scale=True,
                         num_train=50000,
                         num_val=10000)

sensor = FeedSensor(name='Sensor', source_in=source)
s = Kid(sensor,
        brain,
        MomentumKongFu(name="Kongfu"),
        max_steps=100)
kid.setup()
kid.practice()
```

#### Parameter tuning

`akid` offers automatic parameter tuning through defining template using `tune`
function.

```eval_rst
.. autofunction:: akid.train.tuner.tune

```

#### Built-in summary

Tensorboard summaries are built in, and can be easily extended, and
distributedly gathered.

```eval_rst
.. image:: ../images/hist_summary.png
   :align: center

.. image:: ../images/scalar_summary.jpg
   :align: center
```

#### Visualization

It supports visualization of all feature maps and filters with control on the
layout.

```eval_rst
.. image:: ../images/gradual_sparse_fmap.png
   :align: center

.. image:: ../images/gsmax_conv1_filters.png
   :align: center
```

### Programming Paradigm

We have seen how to use functionality of `akid` without much programming in the
previous section. In this section, we would like to introduce the programming
paradigm underlying the previous example, and how to use `akid` as a research
package with such paradigm.

Best designs mimic nature. `akid` tries to reproduce how signals in nature
propagates. Information flow can be abstracted as propagating through
inter-connected blocks, each of which takes inputs and gives outputs. For
example, a vision classification system is a block that takes image inputs and
gives classification results.

```eval_rst
.. image:: ../images/akid_block.png
   :align: center
```

```eval_rst
.. automodule:: akid.core.blocks
```

`akid` offers various kinds of blocks, and it is also easy to build one's own
blocks. The `Kid` class is essentially an assembler that assemblies blocks
provided by `akid` to mainly fulfill the task to train neural networks. Here we
show how to build an arbitrary acyclic graph of blocks, to illustrate how to
use blocks in `akid`.

```eval_rst
.. automodule:: akid.core.brains
```

It is possible to build residual units using `Brain`.

```eval_rst
.. image:: ../images/residual_block.jpg
   :alt: alternate text
   :align: center
```

### Distributed Computation

```eval_rst
.. automodule:: akid.core.engines
```

The end computational graph is

```eval_rst
.. image:: ../images/data_parallelism.jpg
   :alt: data parallelism
   :align: center
```


### Distributed Deployment

```eval_rst
.. image:: https://3.bp.blogspot.com/-z1LvDFM7rKs/V1pTrqr265I/AAAAAAAAAPY/pMZWq_Fm3pMhV0GZ0VeKc4Md6DppM0xlwCLcB/s1600/Screen%2BShot%2B2015-01-26%2Bat%2B5.09.09%2BPM.png
   :scale: 50 %
   :alt: alternate text
   :align: right
```

