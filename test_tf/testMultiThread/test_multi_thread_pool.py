from multiprocessing.pool import ThreadPool
def cube(x):  # 定义函数
    return x ** 2

if __name__ == "__main__":
    POOL_SIZE =10
    pool = ThreadPool(processes=POOL_SIZE)
    res = pool.map(func=cube,iterable=range(100))
    print(res)
