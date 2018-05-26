import multiprocessing

def foo(index,q):
    q.put([11,'hello',True])
    print("index",index," size:",q.qsize())

if __name__ == '__main__':
    q = multiprocessing.Queue() #主进程创建一个q进程队列
    for i in range(10):
        p=multiprocessing.Process(target=foo,args=(i,q,)) #因为名称空间不同，子进程的主线程找不到q队列，所以会报错提示没有q
        p.join()
        p.start()
    print("main:",q.qsize())