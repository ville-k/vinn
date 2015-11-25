import vinnpy


class contexts(object):
    '''
    Encapsulate one time initialization of compute contexts and
    provide easy access to their names from parametrized tests
    '''
    contexts = None
    context_names = None

    def __init__(self):
        contexts.initialize_contexts()

    def enumerate(self):
        return [(context,) for context in contexts.contexts]

    def name(self, testcase_func, param_index, param):
        return "%s_%s" % (
            testcase_func.__name__,
            contexts.context_names[param_index])

    @classmethod
    def initialize_contexts(cls):
        if cls.contexts == None:
            cls.contexts = [vinnpy.cpu_context()]
            cls.context_names = ["CPU"]

            devices = vinnpy.opencl_context.supported_devices()
            cls.contexts.append(vinnpy.opencl_context(devices))
            cls.context_names.append("OpenCL")
