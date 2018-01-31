from multiprocessing import Process, Queue, Event

class SingleBatchGenerator(Process):
    def __init__(self, single_task_q: Queue, stop_event: Event):
        super().__init__()
        self.done_q = single_task_q
        self.stop_event = stop_event
        self.myseed = 0
    # bucketing, padding sequences; and transforming, normalizing labelling matrix
    def next_batch(self, seed: int):
        pass
    def run(self):
        while not self.stop_event.is_set():
            if not self.done_q.full():
                self.done_q.put(self.next_batch(self.myseed))
class BatchAggregator(Process):
    def __init__(self, single_task_q: Queue, multi_task_q: Queue, stop_event: Event):
        super().__init__()
        self.pending_q = single_task_q
        self.done_q = multi_task_q
        self.stop_event = stop_event
        self.mt_batch = {}
    # merge single task batches with same job seed to a multi-task batch
    def merge_st_batches(self, st_batches: Dict[str, Any]):
        pass
    # check whether the multi-task batch contains all tasks
    def is_complete(self, st_batches: Dict[str, Any]):
        pass
    def run(self):
        while not self.stop_event.is_set():
            if not self.done_q.full():
                st_batch = self.pending_q.get()
                job_task = st_batch['task_name']
                job_seed = st_batch['myseed']
                self.mt_batch[job_seed][job_task] = st_batch
                if is_complete(self.mt_batch[job_seed]):
                    self.done_q.put(merge_st_batches(self.mt_batch.pop(job_seed)))

class MultiTaskBatchManager:
    def __init__(self):
        self.stop_event = Event()
        self.single_task_q = Queue(MAX_CAPACITY)
        self.multi_task_train_q = Queue(MAX_CAPACITY)
        self.batch_aggregator = BatchAggregator(self.single_task_q, self.multi_task_train_q, self.stop_event)
        self.batch_generator = {task: SingleBatchGenerator(self.single_task_q, self.stop_event) for task in all_tasks}
        for w in self.batch_generator.values():
            w.start()
        self.batch_aggregator.start()

    def next_batch(self):
        return self.multi_task_train_q.get()

    def close(self, timeout: int = 5):
        self.stop_event.set()
        for w in self.batch_generator.values():
            w.join(timeout=timeout)
            w.terminate()
        self.batch_aggregator.join(timeout=timeout)
        self.batch_aggregator.terminate()