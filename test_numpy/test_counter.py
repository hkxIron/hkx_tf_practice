from collections import Counter
import numpy as np

def t1():
    s = '''A Counter is a dict subclass for counting hashable objects. It is an unordered collection where elements are stored as dictionary keys and their counts are stored as dictionary values. Counts are allowed to be any integer value including zero or negative counts. The Counter class is similar to bags or multisets in other languages.'''.lower()

    c = Counter(s)
    # 获取出现频率最高的5个字符
    print(c.most_common(5))

def t2():
    bag_id =[123,456,789,678,234,2343]
    num=20

    yy=np.array(bag_id).reshape((len(bag_id),1))
    print("yy:",yy)

    cc = np.pad(yy,pad_width=((0,5),(0,2)),mode='constant',constant_values=(0,0))
    print("cc:",cc)
    np.concatenate()


def t3():
    np.random.seed(0)
    xx=np.random.random((2,3))
    print("xx:",xx)
    ss ="\t".join(["%.4f"%x for x in xx[0]])
    #ss ="\t".join(["%.4f"%x for x in xx[0].tolist()])
    print("xx value:",ss)
    print("out:",np.random.random())
    xx.astype(np.float32)

t3()
