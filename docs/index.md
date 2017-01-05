# Welcome to akid's documentation!


* [Get Started](get_started/index.md)
* [Introduction](intros/index.md)
* [How To](how_tos/index.md)
* [Tutorials](tutorials/index.md)
* [Architecture](arch/index.md)
* [Model Zoo](model/index.md)

Fork me at https://github.com/shawnLeeZX/akid !

`akid` is a python package written for doing research in Neural Network. It
also aims to be production ready by taking care of concurrency and
communication in distributed computing. It is built on
[Tensorflow](tensorflow.org). If combining with
[GlusterFS](https://www.gluster.org/), [Docker](https://www.docker.com/) and
[Kubernetes](kubernetes.io), it is able to provide dynamic and elastic
scheduling, auto fault recovery and scalability.

It aims to enable fast prototyping and production ready at the same time. More
specifically, it

* supports fast prototyping
  * built-in data pipeline framework that standardizes data preparation and
    data augmentation.
  * arbitrary connectivity schemes (including multi-input and multi-output
    training), and easy retrieval of parameters and data in the network
  * meta-syntax to generate neural network structure before training
  * support for visualization of computation graph, weight filters, feature
    maps, and training dynamics statistics.
* be production ready
  * built-in support for distributed computing
  * compatibility to orchestrate with distributed file systems, docker
    containers, and distributed operating systems such as Kubernetes. (This
    feature mainly is a best-practice guide for K8s etc, which is under
    experimenting and not available yet.)

The name comes from the Kid saved by Neo in *Matrix*, and the metaphor to build
a learning agent, which we call *kid* in human culture.

It distinguish itself from an unique design, which is described in the following.

```eval_rst
.. figure:: images/akid_block.png
   :align: center

   Illustration of the arbitrary connectivity supported by `akid`. Forward
   connection, branching and mergine, and feedback connection are supported.
```

```eval_rst
.. automodule:: akid.core.blocks
```
