import numpy as np

from akid.utils.test import AKidTestCase, main, TestFactory, debug_on
from akid.sugar import cnn_block
from akid import sugar
from akid import GraphBrain
from akid.layers import SoftmaxWithLossLayer
from akid.layers import ConvolutionLayer, SLUConvLayer
from akid import backend as A


class TestSynapseLayers(AKidTestCase):
    def setUp(self):
        super(TestSynapseLayers, self).setUp()
        sugar.init()

        self.use_cuda_save = A.use_cuda()
        A.use_cuda(False)

        A.reset()

    def tearDown(self):
        A.use_cuda(self.use_cuda_save)

    def test_ip_forward(self):
        X_in = A.Tensor([[1, 1], [1, 0]], requires_grad=True)
        weight = A.Tensor([
            [1, 1],
            [0, 1]
        ])
        X_out_ref = np.array([
            [1, 2],
            [1, 1]
        ])
        X_out_ref += 1

        from akid.layers import InnerProductLayer
        l = InnerProductLayer(in_channel_num=2,
                              out_channel_num=2,
                              initial_bias_value=1.,
                              init_para={"name": "tensor",
                                         "value": weight},
                              name='ip')
        X_out = l.forward(X_in)

        A.init()
        X_out_eval = A.eval(X_out)
        self.assertNdarrayEquals(X_out_eval, X_out_ref)

    def test_conv_forward(self):
        filter = np.array([[
            [[1, 1, 1],
             [1, 1, 1],
             [1, 1, 1]],
            [[2, 2, 2],
             [2, 2, 2],
             [2, 2, 2]]
        ]],
                          dtype=np.float32)
        X_out_ref = np.array([[[[46]]]],
                        dtype=np.float32)
        X_in = np.array(
            [[
                [[1, 1, 1],
                 [1, 1, 1],
                 [1, 1, 1]],
                [[2, 2, 2],
                 [2, 2, 2],
                 [2, 2, 2]]
            ]],
            dtype=np.float32
        )
        filter = A.standardize_data_format(filter, 'oihw')
        X_out_ref = A.standardize_data_format(X_out_ref, 'nchw')
        X_in = A.standardize_data_format(X_in, 'nchw')

        l = ConvolutionLayer(ksize=[3, 3],
                             strides=[1, 1, 1, 1],
                             in_channel_num=2,
                             out_channel_num=1,
                             padding="VALID",
                             initial_bias_value=1.,
                             init_para={"name": "tensor",
                                        "value": filter},
                             do_summary=False,
                             name="test_conv_forward")
        X_out = l.forward(A.Tensor(X_in, requires_grad=True))

        A.init()
        X_out_eval = A.eval(X_out)
        assert (X_out_eval == X_out_ref).all(), "\nX_out_eval = {}\nX_out_ref = {}".format(X_out_eval, X_out_ref)

    def test_summary(self):
        l = ConvolutionLayer(ksize=[3, 3],
                             strides=[1, 1, 1, 1],
                             in_channel_num=10,
                             out_channel_num=100,
                             padding="VALID",
                             initial_bias_value=1.,
                             init_para={"name": "default"},
                             do_summary=True,
                             name="test_summary")
        l.setup()

        # Do forward once to build ops.
        X_in = np.random.uniform(-1, 1, [100, 10, 9, 9])
        X_in = A.standardize_data_format(X_in, 'nchw')
        out = l.forward(A.Tensor(X_in, requires_grad=True))
        summary_ops = A.summary.get_collection()
        summary_op = A.summary.merge(summary_ops)

        A.init()
        A.summary.init()

        for i in xrange(100):
            if A.backend() == A.TORCH:
                l.forward(A.Tensor(X_in, requires_grad=True))
            else:
                A.run(out)
            if i % 10 == 0:
                A.summary.run_summary_op(summary_op)
            A.step()

    # def test_conv_backward(self):
    #     filter = np.array([[
    #         [[1, 1, 1],
    #          [1, 1, 1],
    #          [1, 1, 1]],
    #         [[2, 2, 2],
    #          [2, 2, 2],
    #          [2, 2, 2]]
    #     ]],
    #                       dtype=np.float32)
    #     X_in = np.array([[[[1, 0],
    #                      [0, 2]]]],
    #                     dtype=np.float32)
    #     X_in += 1  # Add bias to substract
    #     X_in = np.einsum('nchw->nhwc', X_in)
    #     out_channel_1_ref = np.array([[1, 1, 1, 0],
    #                                   [1, 1, 1, 0],
    #                                   [1, 1, 1, 0],
    #                                   [0, 0, 0, 0],],
    #                                  dtype=np.float32) +\
    #         np.array([[0, 0, 0, 0],
    #                   [0, 2, 2, 2],
    #                   [0, 2, 2, 2],
    #                   [0, 2, 2, 2],],
    #                  dtype=np.float32)
    #     out_channel_2_ref = out_channel_1_ref * 2
    #     X_out_ref = np.array([out_channel_1_ref, out_channel_2_ref])
    #     X_out_ref = np.array([X_out_ref])
    #     X_out_ref = np.einsum('nchw->nhwc', X_out_ref)
    #     # Convert to H X W X IN_CHANNEL X OUT_CHANNEL
    #     filter = np.einsum('oihw->hwio', filter)

    #     l = ConvolutionLayer(ksize=[3, 3],
    #                          strides=[1, 1, 1, 1],
    #                          in_channel_num=2,
    #                          out_channel_num=1,
    #                          padding="VALID",
    #                          initial_bias_value=1.,
    #                          init_para={"name": "tensor",
    #                                     "value": filter},
    #                          name="test_backward_conv")
    #     # TODO: Ideally no forward should be needed, but now (2b1416) setup is
    #     # coupled with forward, so we do forward here. What is forwarded does
    #     # not matter, since we do not need to use them.
    #     l.forward(A.Tensor(X_out_ref))
    #     X_out = l.backward(X_in)

    #     A.init()
    #     X_out_eval = A.eval(X_out)
    #     assert (X_out_eval == X_out_ref).all()

    # def test_l1_regularization(self):
    #     brain = GraphBrain(name="test_brain")
    #     brain.attach(cnn_block(ksize=[5, 5],
    #                            initial_bias_value=0.,
    #                            init_para={"name": "truncated_normal",
    #                                       "stddev": 0.1},
    #                            wd={"type": "l1", "scale": 0.0005},
    #                            in_channel_num=1,
    #                            out_channel_num=32,
    #                            pool_size=[5, 5],
    #                            pool_stride=[5, 5],
    #                            activation={"type": "gsmax",
    #                                        "group_size": 2}))

    #     brain.attach(cnn_block(ksize=None,
    #                            initial_bias_value=0.1,
    #                            init_para={"name": "truncated_normal",
    #                                       "stddev": 0.1},
    #                            wd={"type": "l1", "scale": 0.0005},
    #                            in_channel_num=1152,
    #                            out_channel_num=512,
    #                            activation={"type": "gsmax",
    #                                        "group_size": 8}))

    #     brain.attach(cnn_block(ksize=None,
    #                            initial_bias_value=0.1,
    #                            init_para={"name": "truncated_normal",
    #                                       "stddev": 0.1},
    #                            wd={"type": "l1", "scale": 0.0005},
    #                            in_channel_num=512,
    #                            out_channel_num=10,
    #                            activation=None))
    #     brain.attach(SoftmaxWithLossLayer(
    #         class_num=10,
    #         inputs=[{"name": "ip3", "idxs": [0]},
    #                 {"name": "system_in", "idxs": [1]}],
    #         name="loss"))

    #     source = TestFactory.get_test_feed_source()
    #     kid = TestFactory.get_test_kid(source, brain)
    #     kid.setup()

    #     loss = kid.practice()
    #     assert loss < 13

    # def test_slu_conv(self):
    #     filter = np.array([
    #         [[[1, 1, 1],
    #           [1, 1, 1],
    #           [1, 1, 1]],
    #          [[-1, -1, -1],
    #           [-1, -1, -1],
    #           [-1, -1, -1]]],
    #         [[[-1, -1, -1],
    #           [-1, -1, -1],
    #           [-1, -1, -1]],
    #          [[1, 1, 1],
    #           [1, 1, 1],
    #           [1, 1, 1]]]
    #     ],
    #                       dtype=np.float32)
    #     # Convert to H X W X IN_CHANNEL X OUT_CHANNEL
    #     filter = np.einsum('oihw->hwio', filter)
    #     X_in = np.array([[
    #         [[1, 1, 1],
    #          [1, 1, 1],
    #          [1, 1, 1]],
    #         [[-1, -1, -1],
    #          [-1, -1, -1],
    #          [-1, -1, -1]]]],
    #                     dtype=np.float32)
    #     X_in = np.einsum('nchw->nhwc', X_in)
    #     X_out_ref = np.array([[[[10.]], [[10.]]]])
    #     X_out_ref = np.einsum('nchw->nhwc', X_out_ref)

    #     l = SLUConvLayer(ksize=[3, 3],
    #                      strides=[1, 1, 1, 1],
    #                      in_channel_num=2,
    #                      out_channel_num=2,
    #                      padding="VALID",
    #                      initial_bias_value=1.,
    #                      init_para={"name": "tensor",
    #                                 "value": filter},
    #                      name="test_slu_conv")
    #     # TODO: Ideally no forward should be needed, but now (2b1416) setup is
    #     # coupled with forward, so we do forward here. What is forwarded does
    #     # not matter, since we do not need to use them.
    #     X_out = l.forward(A.Tensor(X_in))

    #     A.init()
    #     X_out_eval = A.eval(X_out)
    #     assert (X_out_eval == X_out_ref).all(), "X_out_eval: {} while X_out_ref: {}".format(X_out_eval, X_out_ref)

    # def test_colorful_conv(self):
    #     filter = np.array([
    #         [[[1, 1, 1],
    #           [1, 1, 1],
    #           [1, 1, 1]]]
    #     ],
    #                       dtype=np.float32)
    #     filter = np.einsum('oihw->hwio', filter)
    #     X_in = np.array([[
    #         [[1, 1, 1],
    #          [1, 1, 1],
    #          [1, 1, 1]]
    #     ]],
    #                     dtype=np.float32)
    #     X_in = np.einsum('nchw->nhwc', X_in)
    #     C_in = np.array([[
    #         [[1, 1, 1],
    #          [1, 1, 1],
    #          [1, 1, 1]],

    #         [[1, 1, 1],
    #          [1, 1, 1],
    #          [1, 1, 1]],

    #         [[1, 1, 1],
    #          [1, 1, 1],
    #          [1, 1, 1]],
    #     ]],
    #                     dtype=np.float32)
    #     C_in = np.einsum('nchw->nhwc', C_in)
    #     C_filter = np.array([
    #         [
    #             [[1, 1, 1],
    #              [1, 1, 1],
    #              [1, 1, 1]],

    #             [[1, 1, 1],
    #              [1, 1, 1],
    #              [1, 1, 1]],

    #             [[1, 1, 1],
    #              [1, 1, 1],
    #              [1, 1, 1]],
    #      ]
    #     ],
    #                     dtype=np.float32)
    #     C_filter = np.einsum('oihw->hwio', C_filter)
    #     X_out_ref = np.array([[
    #         [[18]],
    #     ]],
    #                     dtype=np.float32)
    #     X_out_ref = np.einsum('nchw->nhwc', X_out_ref)

    #     from akid.layers import ColorfulConvLayer
    #     l = ColorfulConvLayer(in_channel_num=1,
    #                           out_channel_num=1,
    #                           use_bn=False,
    #                           init_para={
    #                               "name": "tensor",
    #                               "value": filter
    #                           },
    #                           ksize=3,
    #                           padding="VALID",
    #                           c_W_initializer={
    #                               "name": "tensor",
    #                               "value": C_filter
    #                           },
    #                           name="c_conv"
    #     )
    #     X_out = l.forward([A.Tensor(X_in), A.Tensor(C_in)])
    #     A.init()
    #     X_out_eval = A.eval(X_out)
    #     assert X_out_ref == X_out_eval, \
    #         "X_out_eval: {}; X_out_ref: {}".format(X_out_eval, X_out_ref)

    # def test_colorful_conv_equivariant(self):
    #     filter = np.array([
    #         [[[1, 1, 1],
    #           [1, 1, 1],
    #           [1, 1, 1]]],
    #         [[[2, 2, 2],
    #           [2, 2, 2],
    #           [2, 2, 2]]]
    #     ],
    #                       dtype=np.float32)
    #     filter = np.einsum('oihw->hwio', filter)
    #     X_in = np.array([[
    #         [[1, 1, 1],
    #          [1, 1, 1],
    #          [1, 1, 1]]
    #     ]],
    #                     dtype=np.float32)
    #     X_in = np.einsum('nchw->nhwc', X_in)
    #     C_in = np.array([[
    #         [[1, 1, 1],
    #          [1, 1, 1],
    #          [1, 1, 1]],

    #         [[2, 2, 2],
    #          [2, 2, 2],
    #          [2, 2, 2]],

    #         [[3, 3, 3],
    #          [3, 3, 3],
    #          [3, 3, 3]],
    #     ]],
    #                     dtype=np.float32)
    #     C_in = np.einsum('nchw->nhwc', C_in)
    #     C_filter = np.array([
    #         [
    #             [[1]],

    #             [[1]],

    #             [[1]],
    #         ],
    #         [
    #             [[1]],

    #             [[1]],

    #             [[1]],
    #         ]
    #     ],
    #                     dtype=np.float32)
    #     C_filter = np.einsum('oihw->hwio', C_filter)
    #     X_out_ref = np.array([[
    #         [[4, 6, 4],
    #          [6, 9, 6],
    #          [4, 6, 4]],
    #         [[8, 12, 8],
    #          [12, 18, 12],
    #          [8, 12, 8]],
    #     ]],
    #                     dtype=np.float32)
    #     X_out_c_list = []
    #     for i in xrange(3):
    #         X_out_c = X_out_ref + i+1
    #         X_out_c_list.append(X_out_c)
    #     X_out_ref = np.concatenate(X_out_c_list, axis=1)
    #     X_out_ref = np.einsum('nchw->nhwc', X_out_ref)

    #     from akid.layers import ColorfulConvLayer
    #     l = ColorfulConvLayer(in_channel_num=1,
    #                           out_channel_num=2,
    #                           init_para={
    #                               "name": "tensor",
    #                               "value": filter
    #                           },
    #                           use_bn=False,
    #                           equivariant=True,
    #                           ksize=3,
    #                           padding="SAME",
    #                           c_W_initializer={
    #                               "name": "tensor",
    #                               "value": C_filter
    #                           },
    #                           name="c_conv"
    #     )
    #     X_out = l.forward([A.Tensor(X_in), A.Tensor(C_in)])
    #     A.init()
    #     X_out_eval = A.eval(X_out)
    #     assert (X_out_ref == X_out_eval).all(), \
    #         "X_out_eval: {}; X_out_ref: {}".format(
    #             np.einsum('nhwc->nchw', X_out_eval),
    #             np.einsum('nhwc->nchw', X_out_ref)
    #             )

    # def test_equivariant_projection(self):
    #     X_in = np.array([[
    #         [[1, 1, 1],
    #          [1, 1, 1],
    #          [1, 1, 1]],
    #         [[2, 2, 2],
    #          [2, 2, 2],
    #          [2, 2, 2]],
    #         [[3, 3, 3],
    #          [3, 3, 3],
    #          [3, 3, 3]],
    #     ]],
    #                     dtype=np.float32)
    #     X_in = np.einsum('nchw->nhwc', X_in)
    #     f_list = []
    #     for i in xrange(3):
    #         filter = np.array([
    #             [
    #                 [[1]],
    #             ],
    #             [
    #                 [[1]],
    #             ]
    #         ],
    #                         dtype=np.float32)
    #         filter *= i+1
    #         f_list.append(filter)
    #     filter = np.stack(f_list, axis=0)
    #     filter = np.einsum('goihw->ghwio', filter)
    #     X_out_ref = np.array([[
    #         [[1, 1, 1],
    #          [1, 1, 1],
    #          [1, 1, 1]],
    #         [[1, 1, 1],
    #          [1, 1, 1],
    #          [1, 1, 1]],
    #         [[4, 4, 4],
    #          [4, 4, 4],
    #          [4, 4, 4]],
    #         [[4, 4, 4],
    #          [4, 4, 4],
    #          [4, 4, 4]],
    #         [[9, 9, 9],
    #          [9, 9, 9],
    #          [9, 9, 9]],
    #         [[9, 9, 9],
    #          [9, 9, 9],
    #          [9, 9, 9]],
    #     ]],
    #                     dtype=np.float32)
    #     X_out_ref = np.einsum('nchw->nhwc', X_out_ref)

    #     from akid.layers import EquivariantProjectionLayer
    #     l = EquivariantProjectionLayer(g_size=3,
    #                                    in_channel_num=1,
    #                                    out_channel_num=2,
    #                                    init_para={
    #                                        "name": "tensor",
    #                                        "value": filter
    #                                    },
    #                                    name="c_conv"
    #     )
    #     X_out = l.forward(A.Tensor(X_in))
    #     A.init()
    #     X_out_eval = A.eval(X_out)
    #     assert (X_out_ref == X_out_eval).all(), \
    #         "X_out_eval: {}; X_out_ref: {}".format(
    #             np.einsum('nhwc->nchw', X_out_eval),
    #             np.einsum('nhwc->nchw', X_out_ref)
    #             )

if __name__ == "__main__":
    main()
