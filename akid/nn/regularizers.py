from .. import backend as A


regularizers = {}


def compute(reg_name, **kwargs):
    reg = regularizers[reg_name]

    # Make sure required fields exist.
    for p in reg.required_fields:
        if p not in kwargs:
            raise KeyError("Required field {} not found in the provided"
                           " initialization parameters, `init_para`. Perhaps"
                           " you have some typos.".format(p))

    return reg.func(**kwargs)


class RegularizerRegistry(object):
    def __init__(self,
                 name,
                 func,
                 required_fields=(),
                 msg=None):
        self.name = name
        self.func = func
        self.required_fields = required_fields
        self.msg = msg
        regularizers[name] = self


def l2_regularizer(var, scale, name=None):
    return A.mul(A.nn.l2_loss(var), scale, name=name)


def l1_regularizer(var, scale, name=None):
    return A.mul(A.nn.l1_loss(var), scale, name=name)


RegularizerRegistry("l2",
                    l2_regularizer,
                    required_fields=("var", "scale"),
                    msg="l2 regularization")
RegularizerRegistry("l1",
                    l1_regularizer,
                    required_fields=("var", "scale"),
                    msg="l1 regularization")
