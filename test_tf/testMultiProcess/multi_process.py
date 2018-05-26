import multiprocessing as mp

def cube(x):  # 定义函数
    return x ** 3


if __name__ == "__main__":
    pool = mp.Pool(processes=4)  # 同时运行的进程数为4个
    results = [pool.apply(cube, args=(x,)) for x in range(1, 7)]  # 起6个进程
    print(results)

    pool = mp.Pool(processes=4)
    results = pool.map(cube, range(1, 7))
    print(results)

    pool = mp.Pool(processes=4)
    results = [pool.apply_async(cube, args=(x,)) for x in range(1,7)]
    output = [p.get() for p in results]
    print(output)