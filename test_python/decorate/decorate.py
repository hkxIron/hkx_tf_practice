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


# 用装饰器来实现对类成员的约束
# decorator with arguments
def hkx_deco(*args, **kwargs):
    def wrapper(cls):
        for k, v in kwargs.items():
            print('key: %s, value: %s' % (k, v))
        return cls
    return wrapper

# 先会去执行decorate
@hkx_deco(name=str, age=int, gender=str, salary=float)  # People = deco(People)
class People(object):
    def __init__(self, name, age, gender, salary):
        self.name = name
        self.age = age
        self.gender = gender
        self.salary = salary

p1 = People('J', 18, 'male', 9.9)
p2 = People(324, 18, 'male', 9.9)
