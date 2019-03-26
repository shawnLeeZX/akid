"""
Observers to keep an eye on kids.

This model contains classes that are used to visualize network.
"""
from __future__ import absolute_import, division, print_function

import os
import sys
import inspect

import matplotlib as mpl
from six.moves import range
from six.moves import zip
mpl.use('Agg')
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d
import numpy as np
import math
import seaborn as sns
import tensorflow as tf
from pandas import Series, DataFrame

from .common import (
    ACTIVATION_COLLECTION,
    AUXILLIARY_STAT_COLLECTION,
    SPARSITY_SUMMARY_SUFFIX
)
from ..layers.synapse_layers import (
    ConvolutionLayer,
    InnerProductLayer,
    SynapseLayer,
)
from ..layers.activation_layers import ReLULayer
from .sensors import FeedSensor
from ..utils import glog as log, tf_to_caffe
from .. import backend as A

# Big figure size for visualization large amount of filters.
_FIG_SIZE = (100, 100)


class Observer(object):
    """
    An observer to visualize any trained brains.

    It takes the folder that saves the checkpoints(saved by tf.train.Saver) of
    a trained brain, as input to do visualization. It supports

    * activation visualization
    * filter visualization
    * 3D visualization of a single filter
    * statistical plotting on filters norm

    It also holds some general plotting routines which does not need a
    `Kid` to use. In that case, `kid` in the constructor could be None.

    Call APIs to draw any of those you want. Visualization will be saved at the
    model dir of the brain your has passed in.

    """
    def __init__(self, kid=None):
        """
        Args:
            kid: akid.core.kids.kid
                The kid whose brain is going to be observed.
        """
        self.kid = kid

    def visualize_classifying(self, name, idxs=None):
        """
        Visualize the prediction of a batch of samples from validation set.

        Args:
            name: str
                The name of the layer that whose `data` property holds the
                desired prediction, which could be logit or prediction
                probability.
            idxs: list
                Indices of images to be displayed. If not given, plot all
                images in the batch.

        """
        try:
            self._maybe_setup_kid()

            with self.kid.sess as sess:
                # Find the layer for logit or probability by name
                self.kid.restore_from_ckpt()
                feed_dict = self._feed_data(sess, get_val=True)
                data = self.kid.engine.get_layer_data(name, get_val=True)

                # Get the prediction for a batch.
                if type(self.kid.sensor) is FeedSensor:
                    _pred = data.eval(feed_dict=feed_dict)
                    data = feed_dict[self.kid.sensor.data(get_val=True)]
                    _ = feed_dict[self.kid.sensor.labels(get_val=True)]
                    if type(_) is list:
                        labels = _[0]
                    else:
                        labels = _
                else:
                    _pred, data, labels = sess.run([
                        data,
                        self.kid.sensor.data(get_val=True),
                        self.kid.sensor.labels(get_val=True)[0]])
                assert (len(_pred.shape) is 2,
                        """
                        Prediction result should be of shape (N, P), while
                        being {}, where N is the batch size, P the logit or
                        probability.
                        """.format(_pred.shape))
                pred = np.argmax(_pred, 1)

                # Plot images alongside labels.
                batch_size = pred.shape[0]
                is_color = data.shape[-1] == 3
                if is_color:
                    cmap = "cubehelix"
                else:
                    cmap = "gray"

                if idxs is None:
                    log.info("No indices of images are given. Will use the"
                             " all batch")
                    idxs = list(range(0, batch_size))

                # Initialize canvas.
                # Arrange image in a grid form as square as possible.
                img_num = len(idxs)
                cols = np.ceil(np.sqrt(img_num))
                rows = np.ceil(img_num / cols)
                log.info("Rows: {}; Cols: {};".format(rows, cols))
                fig = plt.figure()

                # Squeeze if grey image so it could be plotted.
                if data.shape[-1] is 1:
                    data = np.squeeze(data)
                # Plot images.
                log.info("Data will be display in {} row X {} col.".format(
                    rows, cols))
                for idx in idxs:
                    (img, label, pred_label) \
                        = (data[idx], labels[idx], pred[idx])
                    # Subplot No starts at 1. We add one to the idx.
                    axe = fig.add_subplot(rows, cols, idx+1)
                    axe.axis("off")
                    axe.set_title("Label: {}; PL: {}.".format(
                        label, pred_label))
                    axe.imshow(img, cmap=cmap)

                # Configure and show images.
                # Use plt.show() so that even in command line it could display
                # images.
                plt.show()

        except tf.OpError as e:
            log.info("Tensorflow error when running: {}".format(e.message))
            sys.exit(0)

    def visualize_activation(self):
        """
        Load trained net, do one forward propagation, and save activations.

        See more at doc string of `Observer` class.
        """
        log.info("Begin to draw activation of {}".format(self.kid.brain.name))
        try:
            self._maybe_setup_kid()

            with self.kid.sess.as_default():
                self.kid.restore_from_ckpt()
                feed_dict = self._feed_data(self.kid.sess)

                for block in self.kid.brain.blocks:
                    if block.data is None:
                        continue
                    log.info("Drawing activation of layer {}.".format(
                        block.name))
                    batch_activation = block.data.eval(feed_dict=feed_dict)
                    # For now we only visualize the first idx.
                    activation = batch_activation[0]
                    self.kid.sess.run(self.kid.brain.eval,
                             feed_dict=feed_dict)
                    activations_img = self._tile_to_one_square(activation)
                    title = "{} Layer".format(block.name)
                    # Visualization will be saved to an sub-folder of
                    # self.kid.model_dir
                    visualization_dir = self.kid.model_dir + "/visualization"
                    if not os.path.exists(visualization_dir):
                        os.mkdir(visualization_dir)
                    filename = visualization_dir + '/' + \
                        block.name + "_fmap.jpg"
                    self._heatmap_to_file(activations_img, title, filename)
        except tf.OpError as e:
            log.info("Tensorflow error when running: {}".format(e.message))
            sys.exit(0)

    def visualize_filters(self,
                          layers=None,
                          layout={"type": "normal", "num": 1},
                          big_resolution=True):
        """
        Load trained model and draw its filters to files.

        See more at doc string of `Observer` class.

        Args:
            layers: list
                If not None, only visualize filters of layers in the list.
            layout: dict
                How many columns of filters we should lay out.  It supports
                layouts: "square", "normal", "inverse", and "dynamic". "normal"
                and "inverse" support a parameter "num".

                "square" layout is for saving space, such as putting visualized
                filters on page, and is useful for plotting first layer
                filters, and inner product layers. If True, filters won't be
                organized by its 3D structure, but grouped into square to save
                space. Bias will not be drawn.

                The remaining layouts will layout filters preserving their 3D
                structure, which is to say to put 2D filters in a channel
                together, and also draw bias. Filters are organized by columns.

                "dynamic" will determined column number dynamically to make the
                final map as square as possible.

                "normal" layout will layout filters by specified column number
                with a further parameter `num`, which means the number of
                columns will the final filter map has. A larger column number
                corresponds to a fatter feature map. For example, if the value
                is 1, all filters will be in one column, which would result in
                a very thin image. If the column number cannot divide filter
                number, one more column will be used.

                "inverse" layout will layout filters by specified filters
                number in a column, with a further parameter, `num`, which
                means the number of filters in a column.
        """
        log.info("Begin to draw filters of {}".format(self.kid.brain.name))
        self._maybe_setup_kid()

        if A.backend() == A.TF:
            A.init(continue_from_chk_point=True, model_dir=self.kid.model_dir)

        for block in self.kid.brain.blocks:
            if layers:
                if block.name not in layers:
                    continue
            if not issubclass(type(block), SynapseLayer):
                continue
            log.info("Begin to tile the filter of layer {}".format(
                block.name))
            filters_img = self._tile_filters(block,
                                             layout,
                                             padding=2)
            title = "{} Layer".format(block.name)
            # Visualization will be saved to an sub-folder of
            # self.kid.model_dir
            visualization_dir = self.kid.model_dir + "/visualization"
            if not os.path.exists(visualization_dir):
                os.mkdir(visualization_dir)
            filename = visualization_dir + '/' \
                + block.name + "_para.png"
            self._heatmap_to_file(
                filters_img,
                title,
                filename,
                fig_size=_FIG_SIZE if big_resolution else None)

    def do_stat_on_filters(self):
        """
        Load trained model and do statistics on filters.

        See more at doc string of `Observer` class.
        """
        log.info("Begin to do stat on filters of {}".format(
            self.kid.brain.name))
        try:
            self._maybe_setup_kid()

            with self.kid.sess as sess:
                self.kid.restore_from_ckpt()

                stat_ops = tf.get_collection(AUXILLIARY_STAT_COLLECTION)
                stats = sess.run(stat_ops)

                # Draw histogram to file.
                for i, s in enumerate(stats):
                    # Plot.
                    fig = plt.figure()
                    axe = fig.add_subplot(111)
                    sns.set_context("poster")
                    sns.distplot(s, kde=False, rug=True)

                    # Save to file.
                    name = stat_ops[i].op.name.replace('/', '_')
                    axe.set_title(name)
                    # Visualization will be saved to an sub-folder of
                    # self.kid.model_dir
                    visualization_dir = self.kid.model_dir + "/visualization"
                    if not os.path.exists(visualization_dir):
                        os.mkdir(visualization_dir)
                    filename = visualization_dir + '/' + \
                        name + ".jpg"
                    fig.savefig(filename)
                    # We reuse title as a description of the file.
                    log.info("{} saved to {}.".format(name, filename))

        except tf.OpError as e:
            log.info("Tensorflow error when running: {}".format(e.message))
            sys.exit(0)

    def stem3_data(self):
        log.info("Stem3D output data of sensor `{}`".format(
            self.kid.sensor.name))

        try:
            if not self.kid.sensor.is_setup:
                self.kid.sensor.forward()

            with tf.Session(graph=self.kid.graph) as sess:

                if type(self.kid.sensor) is FeedSensor:
                    # Placeholder of `FeedSensor` should be filled.
                    feed_dict = self.kid.sensor.fill_feed_dict()
                    data_batch = feed_dict[self.kid.sensor.data()]
                else:
                    data_batch = self.kid.sensor.data().eval(session=sess)

                # For now we only use the first idx.
                data = data_batch[0]
                shape = data.shape
                # We only deal with gray level image now.
                data = data.reshape(shape[:-1])
                # TODO(Shuai): Deal with color image.
                self._stem3(data)
        except tf.OpError as e:
            log.info("Tensorflow error when running: {}".format(e.message))
            sys.exit(0)

    def _maybe_setup_kid(self):
        if not self.kid.brain.is_setup:
            # Do not do summary during visualization so we do not need to
            # create useless event files.
            self.kid.do_summary = False

            if A.backend() == A.TF:
                A.restore(self.kid.model_dir)

            self.kid.setup()

    def _feed_data(self, sess, get_val=False):
        if type(self.kid.sensor) is FeedSensor:
            # Placeholder of `FeedSensor` should be filled.
            feed_dict = self.kid.sensor.fill_feed_dict(get_val=get_val)
        else:
            feed_dict = None
            tf.train.start_queue_runners(sess=sess)

        return feed_dict

    def _heatmap_to_file(self, img,  title, filename, fig_size=None, **kwargs):
        """
        Given a image, draw it as heat map and save it to file.

        Parameters
        ----------
        img : numpy.array
            A heat map in form of 2D numpy array.
        title: str
            The title that will be appended at the top of the heat map.
        filename: str
            File name of the saved file.
        Remaining keywords will be passed to seaborn's `heatmap` function
        directly.
        """
        fig = plt.figure(figsize=fig_size)
        axe = fig.add_subplot(111)
        sns.set_context("poster")
        if img.min() >= 0:
            # If we know the date are non-negative, we tell seaborn so it could
            # use better visualization color map.
            sns.heatmap(data=img,
                        vmin=0, vmax=img.max(),
                        xticklabels=False, yticklabels=False,
                        square=True, robust=True,
                        **kwargs)
        else:
            sns.heatmap(data=img, center=0,
                        xticklabels=False, yticklabels=False,
                        square=True,
                        **kwargs)
        axe.set_title(title)
        fig.savefig(filename)
        # We reuse title as a description of the file.
        log.info("{} saved to {}.".format(title, filename))

    def _dataframe_to_file(self, df, title, filename):
        """
        Given an data frame whose columns are data that are supposed to be
        plotted as line graphs in one figure, save it to file.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        df.plot(title=title, ax=ax)
        sns.set_context("poster")
        ax.set_title(title)
        fig.savefig(filename)
        # We reuse title as a description of the file.
        log.info("{} saved to {}.".format(title, filename))

    def _get_a_canvas(self, height, width):
        """
        Get a background canvas for drawing images.

        We separate this out basically because we could centralize the thing
        related to background image, so change would be easier, such as
        changing background from white to black to make the boundary more
        obvious.

        Parameters
        ----------
        height : int
            Height of the canvas.
        width : int
            Width of the canvas

        Returns
        -------
        canvas : numpy.array
            A background image matrix.
        """
        return np.ones((height, width)) * 0

    def _get_shape_to_draw(self, filter_weights, name):
        """
        Get the actual shape we want to use to draw filter weights.

        It mainly deal with the case of inner product layer. When the previous
        inner product layer is a spatial feature map, it is better to keep the
        spatial information. In other cases, the returned shape is just the
        weights' original shape.

        WARNING: it is the legacy code to visualize Caffe models. It is not
        used anymore.

        Args:
            filter_weights: numpy.array
                Filter weights we want to visualize.
            name: str
                The name of the layer. It is used to locate the layer in the
                net's list, so previous layer could be fetched to determine the
                shape to draw for inner product layer.
        Returns:
            shape_to_draw: list
                shape used to draw the weights of this layer.
        """
        shape_to_draw = []
        if len(filter_weights.shape) is 2:
            # Means we are dealing with a inner product layer. If the previous
            # layer of this inner product layer is a convolution layer, we
            # display weights of inner product layer in the layout of last
            # layer's feature map so spatial information is kept.

            # Get the shape of last layer's feature map.
            layer_idx = self.kid.brain.block.index(name)
            previous_layer = self.kid.brain.blocks[layer_idx - 1]
            shape = previous_layer.data.get_shape().as_list()
            if len(shape) is 4:
                shape_to_draw.append(shape[1])
                shape_to_draw.append(shape[2])
                shape_to_draw.append(shape[3])
                shape_to_draw.append(filter_weights.shape[1])
                shape_to_draw = tuple(shape_to_draw)
            elif len(shape) is 2:
                shape_to_draw = filter_weights.shape
        else:
            # means we are dealing with a conv layer.
            shape_to_draw = filter_weights.shape

        return shape_to_draw

    def _draw_4D_filter(self,
                        filter,
                        layout={"type": "normal", "num": 2},
                        padding=1):
        """
        Draw 4D filters in multiple columns without biases.
        """
        w = filter[0]
        b = filter[1]
        w_height, w_width, channel_num, filter_num = w.shape

        pad_h = padding
        pad_w = padding
        w_width_pad = w_width + pad_w
        _w_height = w_height + pad_h
        _w_width = w_width_pad * channel_num
        _w_width += pad_w + 1  # For bias

        _col_num = None
        if layout["type"] == "dynamic":
            # First get the column number to handle most of the filters
            _col_num = int(math.floor(
                (filter_num * _w_height / _w_width) ** (0.5)))
            # Handle the special case we only need one column. It cannot be
            # absorbed in the special case because we need compute filters per
            # column using the un-incremented value first, in which case
            # division by 0 error would happen.
            if _col_num is 0:
                _col_num = 1
                filters_per_col = filter_num // _col_num
            else:
                filters_per_col = filter_num // _col_num
                # Deal with remaining filters if there is any
                if filter_num % _col_num is not 0:
                    _col_num += 1
            log.info("Column number is dynamically determined as {}".format(
                _col_num))
        elif layout["type"] == "inverse":
            filters_per_col = layout["num"]
            _col_num = int(np.ceil(filter_num / filters_per_col))
        elif layout["type"] == "normal":
            filters_per_col = filter_num // layout["num"]
            if filter_num % layout["num"] is not 0:
                # Append one more column to deal with the remaining filters.
                _col_num = layout["num"] + 1
            else:
                _col_num = layout["num"]
        else:
            log.info("Layout type `{}` is not supported. You perhaps have a"
                     " type".format(layout["type"]))
            sys.exit(1)

        canvas_h = _w_height * filters_per_col
        canvas_w = _w_width * _col_num
        out_img = self._get_a_canvas(canvas_h, canvas_w)
        log.info(
            "Final image's dimension is going to be {}.".format(out_img.shape))

        for f_idx in range(0, filter_num):
            # Flatten channels into one image
            filter_img = self._get_a_canvas(_w_height, _w_width)
            for c_idx in range(0, channel_num):
                col_idx = c_idx*w_width_pad
                filter_img[0:w_height, col_idx:col_idx+w_width] \
                    = w[..., c_idx, f_idx]
            # Add bias in.
            filter_img[0, -1] = b[f_idx]

            row_idx = (f_idx % filters_per_col) * _w_height
            col_idx = (f_idx // filters_per_col) * _w_width
            out_img[row_idx:row_idx+_w_height, col_idx:col_idx+_w_width] \
                = filter_img

        return out_img

    def _draw_4d_filter_one_col(self, w, b, padding=1):
        """
        Deprecated, and not used anymore. It is legacy code from days I still
        used Caffe.

        Draw 4D filters with bias on the right of a filter.
        """
        w_height, w_width, channel_num, filter_num \
            = w.shape
        # Transpose to Caffe style filter since the code below is original
        # written for Caffe.
        w = tf_to_caffe.tf_to_caffe_blob(w)

        # Traditional CNN only has one bias term per set of filters, we need to
        # how which case we are dealing with.
        if len(b.shape) == 1:
            log.info("Traditional CNN only has one bias term per filter set."
                     " Bias term will be drawn at the end of each row of"
                     " filter set")
            is_old_cnn = True
        else:
            log.info("Each filter of CNN has a bias term. The bias term will"
                     " be drawn side by side with the filter weight image.")
            is_old_cnn = False

        # Init output image size.
        # #################################################################
        # We do some pad between each image so we could distinguish them.
        pad_h = padding
        pad_w = padding
        # One filter consists of filter weight matrix and bias term.
        bias_w = 1  # We explicitly define bias width here for clarity.
        _w_height = w_height + pad_h
        if is_old_cnn:
            _w_width = w_width + pad_w
            out_img = self._get_a_canvas(_w_height * filter_num,
                                         # We add one for the bias term of one
                                         # set of filters
                                         _w_width * channel_num + 1)
        else:
            _w_width = w_width + bias_w + pad_w
            out_img = self._get_a_canvas(_w_height * filter_num,
                                         _w_width * channel_num)
        log.info(
            "Final image's dimension is going to be {}.".format(out_img.shape))

        # Tile the image
        # #################################################################
        for filter_idx in range(0, filter_num*channel_num):
            # Gather stats.
            filter_No = np.floor(filter_idx / channel_num)
            filter_channel_No = filter_idx % channel_num

            # Merge weight image and bias first.
            filter_img = self._get_a_canvas(_w_height, _w_width)
            filter_img[0:w_height, 0:w_width] \
                = w[filter_No, filter_channel_No, ...]
            # If it is not traditional CNN, we append bias term after each
            # filter.
            if not is_old_cnn:
                filter_img[_w_height, _w_width] \
                    = b[filter_No, filter_channel_No]

            # Tile weight image and bias in.
            row_idx = filter_No * _w_height
            col_idx = filter_channel_No * _w_width
            out_img[row_idx:(row_idx+_w_height), col_idx:(col_idx+_w_width)] \
                = filter_img

        # If it is traditional CNN, we append the bias term at the end of each
        # row of output image.,
        for filter_No in range(0, filter_num):
            out_img[filter_No*_w_height, channel_num*_w_width] \
                = b[filter_No]

        return out_img

    def _draw_2D_filter(self, filter):
        raise NotImplementedError("2D filter visualization is not implemented"
                                  " yet!")
        sys.exit()

    def _stem3(self, X, ax=None):
        """
        Draw an image X using 3D stem plot on the given axe object.

        Note that this routine will automatically do a horizontal mirror
        operation given the difference between the image coordinate system and
        the normal Euclidean system. So it does not work for normal stem3
        plotting.

        Parameters
        ----------
        X : numpy.array
            2D matrix to draw.
        ax : matplotlib.axes._subplots.Axes3DSubplot
            The axe object.
        """
        assert len(X.shape) is 2, \
            "Only gray level images are supported. The input's shape is"\
            " {}".format(X.shape)
        # Setup coordinates.
        h, w = X.shape
        x = np.linspace(0, w-1, w)
        y = np.linspace(0, h-1, h)
        xv, yv = np.meshgrid(x, y)

        # If no axis object is given, we need to create it here.
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1, projection='3d')
        # Draw.
        for row in range(h):
            for x_, y_, z_ in zip(xv[row, :], yv[row, :], X[h-1-row, :]):
                line = art3d.Line3D(*list(zip((x_, y_, 0), (x_, y_, z_))),
                                    marker='o')
                ax.add_line(line)
        ax.set_xlim3d(0, w)
        ax.set_ylim3d(0, h)
        ax.set_zlim3d(min(0, X.min()), X.max() * 1.3)

        plt.show()

    def _tile_to_one_square(self, images):
        """
        Tile a set of images into one image.

        Given that there is no sub-level group in images, we tile it to an
        image as square as possible to save space.

        To be able to zoom in and move around to look into the filtered
        results, in the case that we do not need the label information as
        inspecting input data, all filtered results are better to be displayed
        in one image.

        Parameters
        ----------
        images : numpy.array
            The set of images with dimension (H, W, C), where the C is the
            channel number the set of images.

        Returns
        -------
        out_img : numpy.array
            The 2D image tiled with input images.
        """
        log.info("Tiling 3D images to 2D with square layout.")
        if len(images.shape) is 1:
            # means we are dealing with inner product layer.
            channel_num = images.shape[0]
            height = width = 1
            # No padding for ip layer, since each channel only has one pixel.
            pad_h = 0
            pad_w = 0
        else:
            # means we are dealing with conv layer.
            # Convert from tf style to caffe style. (We do this conversion
            # since below is the code that works with Caffe, which I wrote in
            # the days of using Caffe.)
            images = tf_to_caffe.tf_to_caffe_blob(images)
            (channel_num, height, width) = images.shape
            # We do some pad between each image so we could distinguish them.
            pad_h = 1
            pad_w = 1
        # Calculate the size of the final image.
        # Arrange image in a grid form as square as possible.
        cols = int(np.ceil(np.sqrt(channel_num)))
        rows = int(np.ceil(channel_num / cols))
        pad_h = height + pad_h
        pad_w = width + pad_w
        out_img_h = pad_h * rows
        out_img_w = pad_w * cols
        out_img = self._get_a_canvas(out_img_h, out_img_w)
        # Fill the feature map in the output image.
        for map_idx in range(0, channel_num):
            row_idx = int(np.floor(map_idx / cols))
            col_idx = map_idx % cols
            x = row_idx * pad_h
            y = col_idx * pad_w
            out_img[x:(x+height), y:(y+width)] \
                = images[map_idx, ...]
        log.info("Tiling ends.")
        return out_img

    def _tile_filters(self,
                      block,
                      layout,
                      padding=1):
        """
        Tile filters of one layer into one image.

        To be able to compare filters applied on the same feature map, One set
        of filters is going to be grouped in one line. Bias will be side by
        side with the filter weight image.

        Parameters
        ----------
        block: akid.layers.ConnectionLayer
            Which block's filters to visualize.
        layout: an dict
            See method `visualize_filters`.

        Returns
        -------
        out_img : numpy.array
            The tiled image.
        """
        block_type = type(block)

        assert issubclass(block_type, ConvolutionLayer) or\
            issubclass(block_type, InnerProductLayer), \
            "Block type {} is not supported!".format(block_type)

        ret = A.eval(block.var_list)
        w, b = ret[0], ret[1]
        if A.backend() == A.TORCH:
            if len(w.shape) == 4:
                w = np.einsum('{}->{}'.format('oihw', 'hwio'), w)
        if block.bag:
            if "filters_to_visual" in block.bag:
                filters_to_visual = block.bag["filters_to_visual"]
                w = w[..., filters_to_visual]
                b = b[filters_to_visual]

        try:
            layout_type = layout["type"]
        except KeyError as e:
            raise KeyError("Error: {}. Layout `type` is not found in layout: {}. A"
                           " layout type is needed to determine how to draw"
                           " filters. You perhaps have a typo".format(e.message,
                                                                      layout))

        if layout_type is "square":
            shape = w.shape
            if len(shape) is 2:
                # it's a inner product layer.
                w = w.reshape([-1])
                filter_img = self._tile_to_one_square(w)
            else:
                w = w.reshape([shape[0], shape[1], -1])
                filter_img = self._tile_to_one_square(w)
        else:
            if issubclass(block_type, InnerProductLayer):

                # We do not want to modify block.in_shape's value. So make a
                # copy of it.
                if len(block.in_shape) == 3:
                    if A.backend() == A.TORCH:
                        c_in, h_in, w_in = block.in_shape[0], block.in_shape[1], block.in_shape[2]
                        shape_to_draw = [h_in, w_in, c_in]
                else:
                    shape_to_draw = [i for i in block.in_shape]
                shape_to_draw.append(w.shape[-1])

                # If previous layer is IP layer, we just square tile it.
                if len(shape_to_draw) is 2:
                    w = w.reshape([-1])
                    filter_img = self._tile_to_one_square(w)
                else:
                    # else we reshape it to the shape of last feature map for
                    # better visualization.
                    log.info("Inner product layer {} is reshaped to the shape"
                             " of its previous layer to visualize: {} ->"
                             " {}.".format(
                                 block.name, block.shape, shape_to_draw))
                    w = w.reshape(shape_to_draw)

                    filter_img = self._draw_4D_filter([w, b],
                                                      layout=layout,
                                                      padding=1)
            else:
                filter_img = self._draw_4D_filter([w, b],
                                                  layout=layout,
                                                  padding=1)

        return filter_img

    def get_filters(self, model_dir):
        """
        Load trained model from `model_dir`.

        This is a legacy code from the days when I was visualizing Caffe
        trained model. In that case, I do not have easy access to network
        topology, so have to hack to get all filters. `self.visualize_filters`
        now does not use this method anymore.

        Returns:
            filter_dict: dict
                Filters are returned in form of dictionary:
                {
                    name: value
                    ...
                }
                name is layer name with type str while value is a numpy array.
        """
        filter_dict = {}
        with tf.Session(graph=self.kid.graph) as sess:
            if self.kid.brain.infer_graph is None:
                self.kid.setup(sess, continue_from_chk_point=True)
            filters = self.kid.val_brain.get_filters()
            # Eval all parameters.
            # We use an explicit loop since we need both weights and biases
            # at the same time.
            for i in range(0, len(filters), 2):
                weights_tensor = filters[i]
                biases_tensor = filters[i+1]
                name = weights_tensor.name.split('/')[0]
                weight_array = weights_tensor.eval()
                biases_array = biases_tensor.eval()
                filter_dict[name] = [weight_array, biases_array]
        return filter_dict

    def get_activation(self, model_dir, idxs=[0]):
        """
        Notice:

        This is a obsolete method that is used at days when I just ported the
        code from visualization code for Caffe. It is not used now.

        Load trained model from `model_dir`.
        Activation are returned in form of dictionary:
        {
            name: value
            ...
        }
        name is a str while value is a numpy array.

        Args:
            model_dir: str
                where the trained net stored.
            idxs: list of integer
                Normally, the net returns activations in batch. idxs is the
                indices of activations we want. By default, only the first
                activation is returned.
        """
        activation_dict = {}
        with tf.Session(graph=self.kid.graph) as sess:
            if self.kid.brain.infer_graph is None:
                self.kid.brain._setup_infer_graph()
            # Recover inference part of the net.
            activations = tf.get_collection(ACTIVATION_COLLECTION)
            # Remove data activation from the list, given we have to feed the
            # value to it.
            for a in activations:
                if a.name.find("data") > -1:
                    # Save data tensor so it could be used to get data values
                    # later.
                    data = a
                    activations.remove(a)
            # Init.
            saver = tf.train.Saver(tf.global_variables())
            self.restore_from_ckpt(sess, saver, model_dir)
            # Run the graph once to get all activation.
            feed_dict = self.fill_feed_dict(self.data_sets.train)
            activation_values = sess.run(activations, feed_dict=feed_dict)
            # Add data to activation_dict so it could also be visualized.
            if len(idxs) is 1:
                activation_dict["data"] = feed_dict[data][idxs[0], ...]
            else:
                activation_dict["data"] = feed_dict[data][idxs, ...]
            # Add actual activation.
            for i in range(0, len(activations)):
                # Shorten the name.
                name = activations[i].name.split('/')[0]
                if len(idxs) is 1:
                    activation_dict[name] = activation_values[i][idxs[0], ...]
                else:
                    activation_dict[name] = activation_values[i][idxs, ...]
        return activation_dict

    def plot_relu_sparsity(self, EVENT_FILE_PATH):
        log.info("Begin to plot relu sparsity curve of {}".format(
            self.kid.brain.name))
        try:
            if not self.kid.brain.is_setup:
                # Do not do summary during visualization so we do not need to
                # create useless event files.
                self.kid.do_summary = False
                self.kid.setup()

            # Gather tag names for sparsity of relu layers.
            relu_sparsity_tag_list = []
            # The list holds name for columns of final data frame, which would
            # be used as legend name is the produced plot.
            dt_col_name_list = []
            for block in self.kid.brain.blocks:
                if type(block) is ReLULayer:
                    relu_sparsity_tag_list.append(
                        block.data.op.name + '/' + SPARSITY_SUMMARY_SUFFIX)
                    dt_col_name_list.append(block.data.op.name.split('/')[-2])

            # Set up a dictionary to hold sparsity data
            relu_sparsity_dict = {}
            for name in dt_col_name_list:
                relu_sparsity_dict[name] = Series()

            log.info("Gathering sparsity info of {}.".format(
                self.kid.brain.name))
            for e in tf.train.summary_iterator(EVENT_FILE_PATH):
                for v in e.summary.value:
                    for i, tag in enumerate(relu_sparsity_tag_list):
                        if v.tag == tag:
                            relu_sparsity_dict[dt_col_name_list[i]].set_value(
                                e.step, v.simple_value)

            sparsity_df = DataFrame(relu_sparsity_dict)
            title = "Activation sparsity of different layers after ReLU."
            # Visualization will be saved to an sub-folder of
            # self.kid.model_dir
            visualization_dir = self.kid.model_dir + "/visualization"
            if not os.path.exists(visualization_dir):
                os.mkdir(visualization_dir)
            filename = visualization_dir + "/relu_sparsity.jpg"
            self._dataframe_to_file(sparsity_df, title, filename)
        except tf.OpError as e:
            log.info("Error when reading event file: {}".format(e.message))
            sys.exit(0)

    def line_plot(self,
                  x_axis,
                  y_axis,
                  x_label,
                  y_label,
                  title,
                  log_x=False):
        """
        Given a list of data, and corresponding meta info, plot a line graph
        using it.

        Args:
            x_axis: list
                The list contains the values of y axis.
            y_axis: list
                The list contains the values of x axis.
            log_x: Boolean
                If True, x_axis will be taken log.
        """
        series = Series(data=y_axis, index=x_axis)

        # Plot
        sns.set_context("talk")
        fig = plt.figure()
        axe = fig.add_subplot(111)
        axe.set_xlabel(x_label)
        axe.set_ylabel(y_label)
        axe.set_title(title)
        series.plot(ax=axe, kind="line", logx=log_x)
        filename = "{}.jpg".format(title.replace(' ', '_'))
        fig.savefig(filename)

__all__ = [name for name, x in locals().items() if
           not inspect.ismodule(x) and not inspect.isabstract(x)]
