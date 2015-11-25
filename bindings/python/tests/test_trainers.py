import unittest

import vinnpy
from contexts import contexts
from nose_parameterized import parameterized


class test_trainers(unittest.TestCase):
    def build_network(self, context):
        net = vinnpy.network()
        activation = vinnpy.sigmoid_activation()
        hidden = vinnpy.layer(context, activation, 4, 4)
        net.add(hidden)
        return net

    @parameterized.expand(contexts().enumerate, contexts().name)
    def test_minibatch_train(self, context):
        net = self.build_network(context)

        cost_function = vinnpy.squared_error_cost()
        example_count = 10
        features = vinnpy.matrix(context, example_count, 4, 2.0)
        targets = vinnpy.matrix(context, example_count, 4, 1.0)

        trainer = vinnpy.minibatch_gradient_descent(5, 0.01, 2, 5)

        class update_callback(vinnpy.training_callback):
            def __init__(self):
                self.call_count = 0
                super(update_callback, self).__init__()

            def __call__(self, network, current_epoch, current_cost):
                self.call_count += 1
                return False

        cb = update_callback()
        trainer.set_stop_early_callback(cb)
        final_cost = trainer.train(net, features, targets, cost_function)
        self.assertEqual(5, cb.call_count)
