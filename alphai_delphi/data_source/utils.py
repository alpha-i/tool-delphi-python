import logging
import time

from functools import wraps


def logtime(f):
    @wraps(f)
    def with_logs(*args, **kwargs):
        start_time = time.time()
        result = f(*args, *kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logging.info("%r execution time: %2.4f sec", f.__name__, execution_time)
        return result
    return with_logs
