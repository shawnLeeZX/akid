# Architecture and Design Principles

TODO: finish this section

## Architecture

### Kick start clock

A centralized clock is available in `akid` to model the elapsing time in the
physical world. It is roughly the training step, or iteration step when
training a neural network.

If any training is supposed to done with the machinery provided `akid`, the
clock needs to be manually started using the following snippets. However, if
you are using `Kid` class, it will be called automatically.

```
from akid.common import init
init()
```

### Model Abstraction

All core classes a sub class of `Block`.

The philosophy is to bring neural network to its biological origin, the brain,
a data processing engine that is tailored to process hierarchical data of the
universe. The universe is built block by block, from micro world, modules
consisting of atoms, to macro world, buildings made up by bricks and windows,
and cosmic world, billions of stars creating galaxies.

Tentatively, there are two kinds of blocks --- simple blocks and complex
blocks. Simple blocks are traditional processing units such as convolutional
layers. While complex blocks holds sub-blocks with certain topology. Complex
blocks are `System`. A module `systems` offers different `System`s to model the
mathematical topological structures how data propagates. These two types of
blocks build up the recursive scheme to build arbitrarily complex blocks.

### Model and Computation

Blocks are responsible for easing building computational graph. Given two-phase
procedure that first builds computational graph, then executes that graph, each
object (including but not limiting to blocks) who actually needs to do
computation has its own computational components, a graph and a session. If no
higher level one is given, the block will create a suite of its own; otherwise,
it will use what it is given.

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

TODO: build the umbrella within.

```python
with self.graph.as_default():
    # Graph building codes here
```

That is to say if you are going to use certain class standalone, a graph
context manager is needed.

Each `System` takes a `graph` argument on construction. If no one is given, it
will create one internally. So no explicit graph umbrella is needed.

#### Session

TODO: abstract this within

A graph hold all created blocks. To actually run the computational graph, all
computational methods has an `sess` argument to take an opened `tf.Session()`
to run within, thus any upper level execution environment could be passed
down. The upper level code is responsible to set up a session. In such a way,
computational graph construction does not need to take care of
execution. However, for certain class, such as a Survivor, if an upper level
session does not exist, a default one will be created for the execution for
convenience.

This allows a model to be deployed on various execution environment.

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
