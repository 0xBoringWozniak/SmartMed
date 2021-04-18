DEGUG_MODE = True


import logging


def debug(fn):

    def wrapper(*args, **kwargs):

        if DEGUG_MODE:
            print("Entering {:s}.{:s}...".format(fn.__module__,
                                                 fn.__name__))
            result = fn(*args, **kwargs)
            print("Finished {:s}.{:s}.".format(fn.__module__,
                                               fn.__name__))
        else:
            result = fn(*args, **kwargs)

        return result

    return wrapper
