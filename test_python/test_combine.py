def combine(word_list, window=2):
    """构造在window下的单词组合，用来构造单词之间的边。

    Keyword arguments:
    word_list  --  list of str, 由单词组成的列表。
    windows    --  int, 窗口大小。

    这种方式产生pair感觉很妙啊
    ('life', 'is')
    """
    if window < 2: window = 2
    for x in range(1, window):
        if x >= len(word_list):
            break
        word_list2 = word_list[x:]
        res = zip(word_list, word_list2) #  不能对齐的会被扔掉
        for r in res:
            yield r

word_list=["life", "is", "short", "i", "use", "python"]

for x in combine(word_list, 3):
    print(x)

"""
('life', 'is')
('is', 'short')
('short', 'i')
('i', 'use')
('use', 'python')
('life', 'short')
('is', 'i')
('short', 'use')
('i', 'python')
"""