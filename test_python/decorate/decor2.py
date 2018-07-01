# https://blog.csdn.net/Techml/article/details/72626519

class TypeCheck(object):
    def __init__(self, name, expect_type):
        self.name = name
        self.expect_type = expect_type

    def __get__(self, instance, owner):
        if instance in None:
            return self
        return instance.__dict__[self.name]

    def __set__(self, instance, value):
        if not isinstance(value, self.expect_type):
            raise TypeError('%s should be type of %s' % (self.name, self.expect_type))
        instance.__dict__[self.name] = self.expect_type

    def __delete__(self, instance):
        instance.__dict__.pop(self.name)

# decorator with arguments
def deco(*args, **kwargs):
    def wrapper(cls):
        for k, v in kwargs.items():
            setattr(cls, k, TypeCheck(k, v))
        return cls
    return wrapper


@deco(name=str, age=int, gender=str, salary=float) # 添加salary的属性检测
class People(object):
    def __init__(self, name, age, gender, salary):  # 添加salary属性
        self.name = name
        self.age = age
        self.gender = gender
        self.salary = salary

p1 = People("hkx", 18, 'male', 9.9)
p2 = People(7, 18, 'male', 9.9)