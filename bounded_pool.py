from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, wait
import multiprocessing
import threading

# to fix pickling error: Can't pickle local object 'Experiments.__init__.<locals>.<lambda>' caused by future.result()
def use_dill_pickler():
    import dill
    dill.Pickler.dumps, dill.Pickler.loads = dill.dumps, dill.loads
    multiprocessing.reduction.ForkingPickler = dill.Pickler
    multiprocessing.reduction.dump = dill.dump
    multiprocessing.queues._ForkingPickler = dill.Pickler

def BoundedPool(max_workers=None, type=1):
    Base = ThreadPoolExecutor if type == 0 else ProcessPoolExecutor

    class _BoundedPool(Base):
        def __init__(self, max_workers, type):
            super().__init__(None if max_workers == 0 else max_workers)
            modu = threading if type == 0 else multiprocessing
            self.semaphore = modu.BoundedSemaphore(self._max_workers)

        def submit(self, fn, *args, **kwargs):
            self.semaphore.acquire()
            future = super().submit(fn, *args, **kwargs)
            future.add_done_callback(lambda x: self.semaphore.release())
            return future

        def wait(self, futs):
            wait(futs)

    return _BoundedPool(max_workers, type)
