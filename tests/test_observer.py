import tensorflow as tf

from akid import Observer
from akid.utils.test import AKidTestCase, TestFactory, main
from akid.models.brains import OneLayerBrain
from akid import FeedSensor, MomentumKongFu, Kid
from akid import backend as A


class TestObserver(AKidTestCase):
    def setUp(self):
        super(TestObserver, self).setUp()
        brain = OneLayerBrain(do_stat_on_norm=True, name="test_brain")
        source = TestFactory.get_test_feed_source()

        kid = Kid(
            FeedSensor(source_in=source, name='data'),
            brain,
            MomentumKongFu(),
            # Fix log folder so we do not need to retrain each time.
            log_dir="log_test_observer",
            max_steps=900)
        kid.setup()
        self.kid = kid

        # Do not train the model again if it is already there.
        checkpoint = tf.train.get_checkpoint_state(kid.model_dir)
        if not checkpoint:
            loss = kid.practice()
            assert loss < 0.2

        # Since it is the methods that we want to test and the `__init__` of
        # `Observer` is trivial, we set it up here.
        self.observer = Observer(kid)

    def tearDown(self):
        A.reset()

    def test_visualize_classifying(self):
        self.observer.visualize_classifying(name="ip1")

    def test_visualize_filters(self):
        self.observer.visualize_filters()

    def test_visualize_selected_filters(self):
        self.kid.brain.blocks[0].bag = {"filters_to_visual": [0, 1]}
        self.observer.visualize_filters()

    def test_visualize_filters_multi_col(self):
        self.observer.visualize_filters(layout={"type": "normal", "num": 2})

    def test_visualize_filters_inverse_col(self):
        self.observer.visualize_filters(layout={"type": "inverse", "num": 6})

    def test_visualize_filters_dynamic_col(self):
        self.observer.visualize_filters(layout={"type": "dynamic"})

    def test_visualize_filters_square_laytout(self):
        self.observer.visualize_filters(layout={"type": "square"})

    def test_visualize_activation(self):
        self.observer.visualize_activation()

    def test_stem3_data(self):
        self.observer.stem3_data()

    def test_do_stat_on_filters(self):
        self.observer.do_stat_on_filters()

    def test_plot_relu_sparsity(self):
        # TODO(Shuai): Think how to get around hard coded event file name.
        self.observer.plot_relu_sparsity(
            "./log_observer_test/events.out.tfevents.1461574264.shawn-world")

if __name__ == "__main__":
    main()
