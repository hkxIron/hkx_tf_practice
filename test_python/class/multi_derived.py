"""
测试多重继承

Python中，类自身或者其父类继承了object那么这个类就是个新式类，若没有继承object，则是经典类。
因为Python中存在多重继承，在新式类中，要查找或调用一个方法或属性时，使用的是广度优先搜索算法；而在经典类中则使用深度优先搜索算法。
"""

def t0():
    print("------------test t0--------------")
    class A():
    #class A(object):
        """
        A 继承自object，
        """
        def __init__(self):
            self.name = 'Jeremy'


    class B(A):
        def __init__(self):
            # 首先找到B的父类（比如是类A），然后把类B的对象self转换为类A的对象，然后“被转换”的类A对象调用A自己的__init__函数
            # 所以本意是向基类转换
            super(B, self).__init__()
            self.name = 'Jian'


    class C(B):
        def __init__(self):
            super(C, self).__init__()
            self.name = 'Zhao'


    class D(A):
        def __init__(self):
            super(D, self).__init__()
            self.name = 'Zhao Jian'


    class E(D, C, A):
        def __init__(self):
            super(E, self).__init__() # 如果不调用父类，那么E将没有name属性
            self.lover = 'Aimee'
            print(self.name + " love " + self.lover)

    e = E()
t0()

# ----------------------
def t1():
    print("------------test t1--------------")
    class Base(object):
    #class Base(): # 我发现，即使不继承object，也没有问题
        def __init__(self):
            print('Base create')

    class childA(Base):
        def __init__(self):
            print('creat A ')
            Base.__init__(self)


    class childB(Base):
        def __init__(self):
            print('creat B ')
            super(childB, self).__init__()

    base = Base()
    a = childA()
    b = childB()
t1()
"""
#------------
# def super(class_name, self):
#     mro = self.__class__.mro()
#     return mro[mro.index(class_name) + 1]
# mro()用来获得类的继承顺序。

supder和父类没有关联，因此执行顺序是A —> B—>Base

执行过程相当于：初始化childC()时，先会去调用childA的构造方法中的 super(childA, self).__init__()， super(childA, self)返回当前类的继承顺序中childA后的一个类childB；然后再执行childB().__init()__,这样顺序执行下去。

在多重继承里，如果把childA()中的 super(childA, self).__init__() 换成Base.__init__(self)，在执行时，继承childA后就会直接跳到Base类里，而略过了childB：
"""

def t2():
    print("------------test t2--------------")
    class Base(object):
        def __init__(self):
            print('Base create')


    class childA(Base):
        def __init__(self):
            print('enter A')
            #Base.__init__(self)
            super(childA, self).__init__()
            print('leave A')


    class childB(Base):
        def __init__(self):
            print('enter B')
            # Base.__init__(self)
            super(childB, self).__init__()
            print('leave B')


    class childC(childA, childB):
        def __init__(self):
            print('enter c')
            super(childC, self).__init__()
            print('leave c')

    c = childC()
    print(c.__class__.__mro__)
t2()


def t3():
    print("------------test t3--------------")
    """
    如果childA基础Base, childB继承childA和Base，如果childB需要调用Base的__init__()方法时，就会导致__init__()被执行两次：
    """
    class Base(object):
        def __init__(self):
            print('Base create')

    class childA(Base):
        def __init__(self):
            print('enter A ')
            Base.__init__(self)
            print('leave A')

    class childB(childA, Base):
        def __init__(self):
            childA.__init__(self)
            Base.__init__(self)

    b = childB()
    print(b.__class__.mro())

t3()

def t4():
    print("------------test t4--------------")
    """
    使用super避免重复调用
    """
    class Base(object):
        def __init__(self):
            print('Base create')

    class childA(Base):
        def __init__(self):
            print('enter A ')
            super(childA, self).__init__()
            print('leave A')

    class childB(childA, Base):
        def __init__(self):
            super(childB, self).__init__()

    b = childB()
    print(b.__class__.mro())

t4()

