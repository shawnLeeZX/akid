import seaborn as sns
from matplotlib import pyplot as plt
import pickle as pk
import numpy as np
import random

from torchvision import datasets, transforms

from akid import Source, ParallelSensor, AKID_DATA_PATH, Kid, MomentumKongFu
from akid import GraphBrain
from akid import ops
from akid.layers import (
    MaxPoolingLayer,
    ConvolutionLayer,
    ReLULayer,
    ReshapeLayer,
    InnerProductLayer,
    HingeLossLayer,
    BinaryAccuracy,
    PoolingLayer,
    SoftmaxWithLossLayer
)
from akid import backend as A
from akid.utils.test import debug_on


# MNIST
class MNISTBinarySource(Source):
    def __init__(self, example_per_class=None, *args, **kwargs):
        """
        If `example_per_class` is `None`, use the whole dataset.

        The source aims to create a binary dataset from the official MNIST
        dataset. It does so by taking digit 0 as the positive class, and the
        rest of the digits as negative samples. Note that to keep the number of
        positive and negative classes to be balanced, the bulk of the dataset
        is dropped.

        NOTE:

        We do not plan to use more than 1000 samples per class when we want to
        use that parameter, thus, for training and test dataset extracted, they
        have the same number of samples per class.
        """
        super(MNISTBinarySource, self).__init__(*args, **kwargs)
        self.example_per_class = example_per_class
        self.positive_class = 0

    def _setup(self):
        # with open("./mnist_binary.pk", 'rb') as f:
        #     self._data = pk.load(f)

            # Load MNIST data from torch source into memory, and according to
            # the data needed, build the dataset.

        self._train_data_o = datasets.MNIST(self.work_dir, train=True, download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))]))
        self._test_data_o = datasets.MNIST(self.work_dir, train=False,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))]))

        self._train_data = self.extract_subset(self._train_data_o)
        self._test_data = self.extract_subset(self._test_data_o)


    def extract_subset(self, dataset):
        subset = []
        other_class_candidates = []
        for i, d in enumerate(dataset):
            if self.example_per_class is not None and len(subset) >= self.example_per_class:
                break

            img, label = d
            if label == self.positive_class:
                subset.append(d)
            else:
                other_class_candidates.append(d)

        # Get equal number of samples from other classes to make up negative
        # samples.
        # NOTE: we assume negative samples are more (if the original dataset is
        # random), so we just pick the first part.
        N = len(subset)
        negative_subset = [other_class_candidates[i] for i in range(N)]

        # Mix the two sets.
        subset.extend(negative_subset)
        random.shuffle(subset)
        return subset

    @property
    def data(self):
        if self.mode == "train":
            return self._train_data
        elif self.mode == "val":
            return self._test_data
        else:
            raise ValueError("Wrong mode {}".format(self.mode))

    @property
    def size(self):
        return len(self.data)

    def _get(self, indices):
        data = [self.data[i] for i in indices]
        data = list(zip(*data))
        data = [np.stack(data[0]), np.stack(data[1]).astype(np.float32)]
        labels = np.ones_like(data[1], dtype=np.float32)
        for i, l in enumerate(data[1]):
            if l != 0:
                labels[i] = -1
        data = [data[0], labels]
        return data


# # VGG11 on MNIST with hinge loss.
class VGG11(GraphBrain):
    def __init__(self, *args, **kwargs):
        super(VGG11, self).__init__(*args, **kwargs)
        channel_num = [32, 32, 64, 64, 128, 128, 256, 256, 512, 512]
        for i in range(10):
            self.attach(ConvolutionLayer(3, 1, "SAME",
                                         in_channel_num=1 if i == 0 else channel_num[i-1],
                                         out_channel_num=channel_num[i]))
            self.attach(ReLULayer())
            if i != 0 and i % 2 == 0:
                self.attach(MaxPoolingLayer(ksize=2, strides=2, padding="VALID"))

        self.attach(ReshapeLayer())
        self.attach(InnerProductLayer(in_channel_num=512, out_channel_num=1, initial_bias_value=None))
        self.attach(ReshapeLayer(name="ip"))
        self.attach(HingeLossLayer(inputs=[{"name": "ip"}, {"name": "system_in", "idxs": [1]}]))
        self.attach(BinaryAccuracy(hinge_loss_label=True, inputs=[{"name": "ip"}, {"name": "system_in", "idxs": [1]}]))

class OneLayerBrain(GraphBrain):
    def __init__(self, **kwargs):
        super(OneLayerBrain, self).__init__(**kwargs)
        self.attach(
            ConvolutionLayer(ksize=[5, 5],
                             strides=[1, 1],
                             padding="SAME",
                             in_channel_num=1,
                             out_channel_num=32,
                             name="conv1")
        )
        self.attach(ReLULayer(name="relu1"))
        self.attach(
            PoolingLayer(ksize=[5, 5],
                         strides=[5, 5],
                         padding="SAME",
                         name="pool1")
        )
        # self.attach(InnerProductLayer(in_channel_num=1152, out_channel_num=10))
        # Since torch returns None when differentiating constant, we disable
        # the last layer's bias to avoid creating None.
        self.attach(InnerProductLayer(in_channel_num=1152, out_channel_num=1, initial_bias_value=None))
        self.attach(ReshapeLayer(name="ip"))
        self.attach(HingeLossLayer(
            inputs=[
                {"name": "ip", "idxs": [0]},
                {"name": "system_in", "idxs": [1]}],
        ))
        self.attach(BinaryAccuracy(hinge_loss_label=True, inputs=[{"name": "ip"}, {"name": "system_in", "idxs": [1]}]))
        # self.attach(SoftmaxWithLossLayer(
        #     class_num=10,
        #     inputs=[
        #         {"name": "ip", "idxs": [0]},
        #         {"name": "system_in", "idxs": [1]}],
        #     name="loss"))


class LeNet(GraphBrain):
    """
    A rough LeNet. It is supposed to copy the example from Caffe, but for the
    time being it has not been checked whether they are exactly the same.
    """
    def __init__(self, **kwargs):
        super(LeNet, self).__init__(**kwargs)
        self.attach(ConvolutionLayer(ksize=[5, 5],
                                     strides=[1, 1],
                                     padding="SAME",
                                     in_channel_num=1,
                                     out_channel_num=32,
                                     name="conv1"))
        self.attach(ReLULayer(name="relu1"))
        self.attach(PoolingLayer(ksize=[2, 2],
                                 strides=[2, 2],
                                 padding="SAME",
                                 name="pool1"))

        self.attach(ConvolutionLayer(ksize=[5, 5],
                                     strides=[1, 1],
                                     padding="SAME",
                                     in_channel_num=32,
                                     out_channel_num=64,
                                     name="conv2"))
        self.attach(ReLULayer(name="relu2"))
        self.attach(PoolingLayer(ksize=[5, 5],
                                 strides=[2, 2],
                                 padding="SAME",
                                 name="pool2"))

        self.attach(InnerProductLayer(
            in_channel_num=2304,
            out_channel_num=512,
            name="ip1"))
        self.attach(ReLULayer(name="relu3"))

        self.attach(InnerProductLayer(
            in_channel_num=512,
            out_channel_num=1,
            initial_bias_value=None,
            name="ip2"))
        self.attach(ReshapeLayer(name="ip"))
        self.attach(HingeLossLayer(
            inputs=[
                {"name": "ip", "idxs": [0]},
                {"name": "system_in", "idxs": [1]}],
        ))
        self.attach(BinaryAccuracy(hinge_loss_label=True, inputs=[{"name": "ip"}, {"name": "system_in", "idxs": [1]}]))


# A sensor specifically for computing spectrum
s_spectrum = MNISTBinarySource(example_per_class=200, name="mnist_spectrum")
s_spectrum.setup()
spectrum_sensor = ParallelSensor(
    source_in=s_spectrum,
    batch_size=200,
    # Do not shuffle training set for reproducible test
    sampler="sequence",
    name='mnist_spectrum')


def lanczos_nn(kid):
    global spectrum_sensor
    def dataset_hessian_vector_product_cr(f, v, c=None, d=None):
        """
        This function takes a coroutine `f` that needs to delegate the
        computation of Hessian vector product upward, since the Hessian vector
        product is accumulated batch by batch. `v` is the vector to
        multiply. `c` and `d` are normalization parameters to normalize the
        Hessian if needed.
        """
        while True:
            Hv = 0
            for data in spectrum_sensor:
                kid.run_step(update=False, data=data)
                grad = A.nn.grad(kid.brain.loss, kid.brain.get_filters(), flatten=True)
                Hv_ = A.nn.hessian_vector_product(grad, kid.brain.get_filters(), v, allow_unused=True)
                if c is not None and d is not None:
                    # We need to normalize the Hessian.
                    Hv_ = Hv_ / d - c / d * v
                Hv += 1/spectrum_sensor.num_batches_per_epoch * Hv_
            v = f.send(Hv)
            if type(v) is tuple:
                f.close()
                break

        return v

    # Normalize the Hessian to be in [-1, 1].
    N = kid.brain.num_parameters()
    f, v = ops.matrix_ops.center_unit_eig_normalization_cr(N, iter_num=128, kappa=0.05)
    v = dataset_hessian_vector_product_cr(f, v)
    c, d = v # The parameters to renormalize Hessian.

    # Run Lanczos spectrum approximation
    f, v = ops.matrix_ops.lanczos_spectrum_approx_cr(N, iter_num=128, K=1024,
                                                     n_vec=1)
    v = dataset_hessian_vector_product_cr(f, v, c=c, d=d)

    psi = v[0]
    return psi, c, d


dynamics = []
fig = plt.figure()

def lanczos_hook(kid):
    # Plot eigenspectrum at the end of each epoch.
    global spectrum_sensor
    global dynamics
    if not spectrum_sensor.is_setup or spectrum_sensor.mode != "train":
        spectrum_sensor.set_mode("train")
        spectrum_sensor.setup()

    psi, c, d = lanczos_nn(kid)
    psi, c, d = A.eval(psi), A.eval(c), A.eval(d)

    # sns.lineplot(data=pd.DataFrame(psi, index=np.linspace(-d + c, d, 1024)) )
    # fig = plt.figure()
    plt.plot(np.linspace(-d + c, d + c, 1024), psi, figure=fig)
    plt.yscale("log")
    plt.xscale("symlog")
    # fig.savefig("hessian_eigspectrum_vgg_{}.jpg".format(kid.epoch))
    dynamics.append([
        (c, d),
        psi,
    ])


def update_lr(kid):
    if kid.epoch <= 5:
        kid.kongfu.set_lr(0.01)
    else:
        kid.kongfu.set_lr(0.001)

# Clean data so it only has two class.
# s = MNISTBinarySource(name='mnist')
s = MNISTBinarySource(name='mnist')
s.setup()
sensor = ParallelSensor(
    source_in=s,
    batch_size=200,
    # Do not shuffle training set for reproducible test
    # sampler="sequence",
    val_batch_size=10,
    sampler="shuffle",
    name='mnist')


# brain = VGG11()
# brain = OneLayerBrain()
brain = LeNet()
kid = Kid(sensor,
          brain,
          MomentumKongFu(),
          log_by_step=False,
          log_by_epoch=True,
          train_log_step=10,
          # debug=True,
          max_epoch=10)
kid.do_summary = False
kid.hooks.on_train_begin.append(lanczos_hook)
kid.hooks.on_epoch_end.append(lanczos_hook)
kid.hooks.on_epoch_end.append(update_lr)
kid.setup()
# lanczos_hook(kid)
kid.practice()
kid.teardown()
spectrum_sensor.teardown()
A.reset()

fig.savefig("hessian_eigspectrum_lenet.jpg")

print(dynamics)

with open("eigenspectrum.pk", 'wb') as f:
    pk.dump(dynamics, f)
