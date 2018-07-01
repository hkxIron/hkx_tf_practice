# blog: https://cloud.tencent.com/developer/article/1153752
from functools import partial

def power(base, exponent):
    return base ** exponent

square = partial(power, exponent=2)
cube = partial(power, exponent=3)

def test_partials():
    print("test partials")
    assert square(2) == 4
    assert cube(2) == 8

def test_partial_docs():
    print("test partial doc")
    assert square.keywords == {"exponent": 2}
    assert square.func == power
    assert cube.keywords == {"exponent": 3}
    assert cube.func == power

test_partials()
test_partial_docs()


def test_power_partials():
    # 准备一个存储新函数的列表
    #power_partials = []
    """
    for x in range(1, 11):
        # 创建新的函数
        f = partial(power, exponent=x)
        # 将新的函数加入列表中
        power_partials.append(f)
    """
    # 当然我们也可以使用列表解析式来完成上面的工作
    power_partials = [partial(power, exponent=x) for x in range(1, 11)]
    # 测试第一个新函数
    assert power_partials[0](2) == 2
    # 测试第五个新函数
    assert power_partials[4](2) == 32
    # 测试第十个新函数
    assert power_partials[9](2) == 1024

test_power_partials()


from six import add_metaclass

class PowerMeta(type):
    def __init__(cls, name, bases, dct):
        # 在这里，我生成50个新函数
        for x in range(1, 51):
            # 这里使用了python的反射
            setattr(
                # cls就是我们这个类了
                cls,
                # 给新函数取一个名字
                "p{}".format(x),
                # 新函数的具体定义
                partial(power, exponent=x)
            )
        super(PowerMeta, cls).__init__(name, bases, dct)

@add_metaclass(PowerMeta)
class PowerStructure(object):
    pass

def test_power_structure_object():
    p = PowerStructure()
    # p2的10次方
    assert p.p2(10) == 100
    # p5的2次方
    assert p.p5(2) == 32
    # p50的2次方
    assert p.p50(2) == 1125899906842624

test_power_structure_object()

def test_power_structure_class():
    # 这里就能感受到元类的强大了吧！
    assert PowerStructure.p2(10) == 100
    assert PowerStructure.p5(2) == 32
    assert PowerStructure.p50(2) == 1125899906842624

test_power_structure_class()
