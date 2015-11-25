import unittest
import tempfile
import shutil

from contexts import contexts
import vinnpy
from nose_parameterized import parameterized


class test_model(unittest.TestCase):
    def build_network(self, context):
        net = vinnpy.network()
        activation = vinnpy.sigmoid_activation()
        hidden = vinnpy.layer(context, activation, 4, 4)
        net.add(hidden)
        return net

    @parameterized.expand(contexts().enumerate, contexts().name)
    def test_model_store_and_load(self, context):
        '''
        Test models can be stored and loaded from Python
        '''
        model_dir = tempfile.mkdtemp()
        try:
            out_net = self.build_network(context)
            out_model = vinnpy.model(model_dir)
            out_model.store(out_net)

            in_model = vinnpy.model(model_dir)
            in_net = vinnpy.network()
            in_model.load(in_net, context)

            self.assertEqual(out_net.size(), in_net.size())
        finally:
            shutil.rmtree(model_dir)
