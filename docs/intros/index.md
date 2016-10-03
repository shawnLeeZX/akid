# Introduction

Best designs mimic nature. `akid` tries to reproduce how signals in nature
propagates. Information flow can be abstracted as propagating through
inter-connected blocks, each of which takes inputs and gives outputs. For
example, a vision classification system is a block that takes image inputs and
gives classification results.

This makes possible using `akid` in both low level abstraction and high level
abstraction. In the low level, one can just use the machinery of blocks, and
build computation graph as normal. In the high level, complex graph building
machinery, parallelism schemes, logging machinery and training utilities are
also provided.

In the following, two typical user case are introduced.

### Use Standalone Blocks

This case suits the ones who normally want to build things from scratch. It
aims to eases the burden to maintain the name scope or other logistics related
to a specific package, and lets people focus on the problem instead of the
code.

`akid` provides full machinery from preparing the data, doing data
augmentation, specifying computation graph (neural network architecture),
choosing optimization algorithms, specifying training scheme (data parallelism
etc), and information logging. However, it is possible to use those components
separately.

As an example, following the LEGO block design philosophy, it is possible to
`Brain` alone to just specify the computation graph, and use whatever the data
preparation machinery of your own. What you need is to pass the data tensors
and call `setup`.

```python
model = build_brain()
data, labels = build_input()
brain.setup([data, labels])
```

Similar scenarios hold for all classes.

### Use All Facilities Of Akid

This case suits the ones who are product driven --- wants to get results
quickly. It aims to provide an easy to use flow to all elements of neural
network --- training, testing, logging, parallelism schemes, etc. That is to
say, it could make use of all possible blocks in `akid`.

From a perspective of design, this is what `akid` means, explained in the
following. Also, in this section, all available blocks in `akid` are briefly
introduced.

It is designed by analogy to a living being. A living being needs to use all
kinds of *sensors* it equipped to sense a *source*, parts of the world, by a
certain *way*, to accumulate experience and summarize knowledge in ones *brain*
to fulfill a basic purpose, to *play*.

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

TODO: write a full example here.

## Architecture

TODO: the following sections may not belong here.

### Kick start clock

If any training is supposed to done, using the machinery provided `akid`, the
clock needs to be manually started using the following snippets.

```
from akid.common import init
init()
```

### Model Abstraction

All core classes a sub class of `Block`.

The philosophy is to make neural network come to its biological origin, the
brain, a data processing engine that is tailored to process hierarchical data
from our universe. The universe is built block by block, from micro world,
modules consisting of atoms, to macro world, buildings made up by bricks and
windows, and cosmic world, billions of stars creating galaxies.

To deal with how blocks composes a larger block, a module `systems` offers
different `System`s to model the mathematical topological structures how data
propagates. `System` is also a sub-class of `Block`, since it is just a larger
block.

Rephrase the scenario in the *Introduction* under the block universe: A
`Survivor` combines necessary `Block`s to survive. All previous classes
assembled by `Survivor` are sub-classes of `Block`.

### Model and Computation

Each object who actually needs to do computation has its own computational
components, a graph and a session. If no higher level one is given, the block
will create a suite of its own; otherwise, it will use what it is given.

#### Graph

A model's construction is separated from its execution environment.

To use most classes, its `setup` method should be called before anything
else. This is tied with Tensorflow's two-stage execution mechanism: first build
the computational graph in python, then run the graph in the back end. The
`setup` of most classes build and do necessary initialization for the first
stage of the computation. The caller is responsible for passing in the right
data for `setup`.

`setup` should be called under a `tf.Graph()` umbrella, which is in the
simplest case is a context manager that open a default graph:

```python
with self.graph.as_default():
    # Graph building codes here
```

That is to say if you are going to use certain class standalone, a graph
context manager is needed.

Each `System` takes a `graph` argument on construction. If no one is given, it
will create one internally. So no explicit graph umbrella is needed.

#### Session

A graph hold all created blocks. To actually run the computational graph, all
computational methods has an `sess` argument to take an opened `tf.Session()`
to run within, thus any upper level execution environment could be passed
down. The upper level code is responsible to set up a session. In such a way,
computational graph construction does not need to take care of
execution. However, for certain class, such as a Survivor, if an upper level
session does not exist, a default one will be created for the execution for
convenience.

This allows a model to be deployed on various execution environment.

### Parallelism

`akid` offers built-in data parallel scheme in form of class `Engine`, which
could be used with `Survivor`. Just specify the engine at the construction of
the survivor.


## Design Principles

### Compactness

The design principle is to make the number of concepts exist at the same time
as small as possible.

### LEGO Blocks

The coding process is to assembly various smaller blocks to form necessary
functional larger blocks.

The top level concept is a survivor. It models how an agent explore the world
by learning in order to survive, though the world has not been modeled yet. Up
to now, it could be certain virtual reality world that simulate the physical
world to provide environment to the survivor.

A `Survivor` assemblies together a `Sensor`, a `Brain` and a `KongFu`. A
`Sensor` assemblies together `Joker`s and data `Source`s. A `Brain` assemblies
together a number of `ProcessingLayer` to form a neural networking.

### Distributed Composition

Every sub-block of a large block should be self-contained. A large block only
needs minimum amount of information from a sub block. They communicate through
I/O interfaces. So the hierarchical composition scale up in a distributed way
and could goes arbitrary deep with manageable complexity.
