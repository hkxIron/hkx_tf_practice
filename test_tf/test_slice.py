import tensorflow as tf
sess = tf.Session()

def test_slice():
    t = tf.constant([[[1, 1, 1], [2, 2, 2]],
                     [[3, 3, 3], [4, 4, 4]],
                     [[5, 5, 5], [6, 6, 6]]])
    print("test slice:")
    print(sess.run(tf.slice(t, begin=[1, 0, 0], size=[1, 1, 3]))) # [[[3, 3, 3]]]

    print(sess.run(tf.slice(t, begin=[1, 0, 0], size=[1, 2, 3])))  # [[[3, 3, 3],

    print(sess.run(tf.slice(t, begin=[1, 0, 0], size=[2, 1, 3])))  # [[[3, 3, 3]],
                                                        #  [[5, 5, 5]]]

def t1():
    data = [
            [[1, 1, 1], [2, 2, 2]],
            [[3, 3, 3], [4, 4, 4]],
            [[5, 5, 5], [6, 6, 6]]
           ]
    #data_tensor =tf.convert_to_tensor(data,dtype=tf.int32)
    data_tensor =tf.constant(data,dtype=tf.int32)
    print("data:",data, "\ndata_tensor:",data_tensor)
    x = tf.strided_slice(data,begin=[0,0,0],end=[1,1,1]) # 不包含end
    print("stride slice:", sess.run(x))

    """
    [[[1]]]
    """
    print("----------------")
    # strided_slice是不包含end索引
    x = tf.strided_slice(data,[0,0,0],[2,2,3]) # 0维是最外面那层括号，第1维是次外层括号，第2维是最里层括号
    print(sess.run(x))
    """
    [
     [[1 1 1]
      [2 2 2]]

     [[3 3 3]
      [4 4 4]]
    ]
    """

    # 当指定stride为[1,1,1]输出和没有指定无区别，可以判断默认的步伐就是每个维度为1
    print("----------------")
    x = tf.strided_slice(data,[0,0,0],[2,2,2],[1,2,1])
    print(sess.run(x))

    """
    [[[1 1]]

     [[3 3]]]
    """

    print("----------------")
    # 当begin为正值，stride任意位置为负值，输出都是空的
    x = tf.strided_slice(data, [1, -1, 0], [2, -3, 3], [1, -1, 1])
    print(sess.run(x))
    """
    [[[4 4 4]
      [3 3 3]]]
    """

def t2():
    data = [[[1, 1, 1], [2, 2, 2]],
            [[3, 3, 3], [4, 4, 4]],
            [[5, 5, 5], [6, 6, 6]]]

    x = tf.strided_slice(data, [0, 0, 0], [1, 1, 1])
    y = tf.strided_slice(data, [0, 0, 0], [2, 2, 2], [1, 1, 1])
    z = tf.strided_slice(input_=data, begin=[0, 0, 0], end=[2, 2, 2], strides=[1, 2, 1])
    # begin=[dim0,dim1,dim2], end=[dim0,dim1,dim2]
    t = tf.strided_slice(input_=data, begin=[0, 0, 0], end=[2, 2, 2], strides=[1, 1, 2])

    print("x:",sess.run(x))
    print("y:",sess.run(y))
    print("z:",sess.run(z))
    print("t:",sess.run(t))
"""
# stride=[dim0=1,dim1=1,dim2=1]
# count0=count1=count2=(end-start)/stride=(1-0)/1=1
x: [[[1]]]

# stride=[dim0=1,dim1=1,dim2=1]
# count0=count1=count2=(end-start)/stride=(2-0)/1=2
所谓count,即count0=2,即最外层"["下有2个"[]",即y:[
    y1:[]
    y2:[]
]
count1=2,则y1下有2个[][],即 y1:[ z1:[]
                                 z2:[]]
count2=2,则z1下有2个元素:即 z1:[a, b]
最后如下图:
y: [[[1 1]
     [2 2]]

    [[3 3]
     [4 4]]]
  
  
  
# stride=[dim0=1,dim1=2,dim2=1]
# count0=count2=(end-start)/stride=2, count1=(end-start)/stride = 2/2=1, y1下只有一个[],即y1:[z:[  ]]
z: [[[1 1]]

    [[3 3]]]


# stride=[dim0=1,dim1=1,dim2=2]
# count0=count1=(end-start)/stride=2, count2=(end-start)/stride = 2/2=1
# z1下只有一个元素,即z1:[a]
t: [[[1] 
     [2]]

    [[3] 
     [4]]]
    
"""
test_slice()
t1()
t2()

sess.close()
