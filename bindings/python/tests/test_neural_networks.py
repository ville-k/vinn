import unittest

import vinnpy
from contexts import contexts
from nose_parameterized import parameterized


class test_neural_networks(unittest.TestCase):
    @parameterized.expand(contexts().enumerate, contexts().name)
    def test_create_network(self, context):
        net = vinnpy.network()
        hidden = vinnpy.layer(context, vinnpy.sigmoid_activation(), 4, 4)
        net.add(hidden)

    @parameterized.expand(contexts().enumerate, contexts().name)
    def test_adding_incompatible_layers_raises(self, context):
        net = vinnpy.network()
        hidden_outputs = 4
        hidden = vinnpy.layer(context, vinnpy.sigmoid_activation(), hidden_outputs, 4)
        net.add(hidden)
        incompatible = vinnpy.layer(context, vinnpy.sigmoid_activation(), 4, hidden_outputs + 1)
        self.assertRaises(vinnpy.invalid_configuration, lambda: net.add(incompatible))

    @parameterized.expand(contexts().enumerate, contexts().name)
    def test_layer_ownership(self, context):
        '''
        Creation of network from layers and activation functions
        that have gone out of scope should succeed
        '''
        def create_activation():
            activation = vinnpy.sigmoid_activation()
            return activation

        def create_layer(context):
            activation = create_activation()
            hidden = vinnpy.layer(context, activation, 4, 4)
            return hidden

        net = vinnpy.network()
        hidden = create_layer(context)
        net.add(hidden)

    @parameterized.expand(contexts().enumerate, contexts().name)
    def test_network_forward(self, context):
        net = vinnpy.network()
        net.add(vinnpy.layer(context, vinnpy.sigmoid_activation(), 4, 4))
        inputs = vinnpy.matrix(context, 1, 4, 2.0)
        net.forward(inputs)

    @parameterized.expand(contexts().enumerate, contexts().name)
    def test_network_backward(self, context):
        net = vinnpy.network()
        net.add(vinnpy.layer(context, vinnpy.sigmoid_activation(), 4, 4))

        cost_function = vinnpy.squared_error_cost()
        example_count = 1000
        features = vinnpy.matrix(context, example_count, 4, 2.0)
        targets = vinnpy.matrix(context, example_count, 4, 1.0)
        cost_and_gradients = net.backward(features, targets, cost_function)
