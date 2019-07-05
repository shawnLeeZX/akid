`akid` is a python package written for doing research in Neural Network (NN). It
also aims to be production ready by taking care of concurrency and
communication in distributed computing (which depends on the utilities provided
by [PyTorch](http://pytorch.org) and [Tensorflow](http://tensorflow.org)). It
could be seen as a user friendly front-end to torch, or tensorflow, like
[Keras](https://keras.io/). It grows out of the motivation to reuse my old
code, and in the long run, to explore alternative framework for building NN. It
supports two backends, i.e., Tensorflow and Pytorch. If combining with
[GlusterFS](https://www.gluster.org/), [Docker](https://www.docker.com/) and
[Kubernetes](kubernetes.io), it is able to provide dynamic and elastic
scheduling, auto fault recovery and scalability (which is not to brag
the capability of `akid`, since the features are not features of `akid` but
features thanks to open source (and libre software), but to mention the
possibilities that they can be combined.).

See http://akid.readthedocs.io/en/latest/index.html for documentation. The
document is dated, and has not been updated to include new changes e.g., the
PyTorch backend. But the backbone design is the same, and main features are
there.

NOTE: the PyTorch end support has been way ahead of Tensorflow support now ...
