import multiprocessing
from multiprocessing.pool import ThreadPool

num_threads = 2
args=[0,1,2,3,4,5]

def func(data):
    print("input:",data)
    return -1*data

def mutliprocess_pool():
    print("multiprocess pool")
    pool = multiprocessing.Pool(processes=num_threads)
    it = pool.imap(func, args, chunksize= 2)
    #for arg in args:
        ##ret = it.next()
    print("return data:",it)
    for s in it:
        print("ret:", s )

def thread_pool():
    print("thread pool")
    import threading
    progress_thread = threading.Thread()
    progress_thread.daemon = True
    progress_thread.start()

    pool = ThreadPool(num_threads)
    it = pool.imap_unordered(func, args, chunksize=2)
    for s in it:
        print(s)

if __name__ == "__main__":
    # 多进程时一定要加判断
    mutliprocess_pool()

#多线程
thread_pool()


