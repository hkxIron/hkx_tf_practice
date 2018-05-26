from time import ctime, sleep

def tsfunc(func):
    def wrappedFunc():
        print("[%s] %s() called"%(ctime(),func.__name__))
        return func()
    return wrappedFunc

@tsfunc
def foo():
    print("foo() is invoked!")

foo() # 相当于组合函数： tsfunc(foo)
sleep(2)

for i in range(2):
    print("i:%d"%i)
    sleep(1)
    foo()
