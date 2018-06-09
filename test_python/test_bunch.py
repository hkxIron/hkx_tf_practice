from collections import defaultdict

class Bunch(dict):
    def __init__(self, *args, **kwargs):
        super(Bunch, self).__init__(*args, **kwargs)
        self.__dict__ = self

dt ={"a":1, "b":2, "c":3}
bun = Bunch(dt)
print(bun)
print(bun.a) # 直接通过 "." 即可访问其属性

x = Bunch(age="54", address="Beijing")
print(x.age)

# 或者import bunch
# from bunch import Bunch

