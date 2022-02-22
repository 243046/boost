from time import time


def timeit(f):
    def timed(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        print(f'Time elapsed: {time()-ts:.3f}s')
        return result
    return timed
