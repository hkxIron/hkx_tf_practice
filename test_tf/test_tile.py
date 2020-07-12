import tensorflow as tf

def index_matrix_to_pairs(index_matrix):
  """
  [[3,1,2],
   [2,3,1]] ->

 [[[0 3]
  [0 1]
  [0 2]]

 [[1 2]
  [1 3]
  [1 1]]]
  """
  replicated_first_indices = tf.range(tf.shape(index_matrix)[0])
  rank = len(index_matrix.get_shape())
  if rank == 2:
    replicated_first_indices = tf.tile(
        tf.expand_dims(replicated_first_indices, dim=1), # [[0, 1]]
        [1, tf.shape(index_matrix)[1]])
  return tf.stack([replicated_first_indices, index_matrix], axis=rank)

if __name__=="__main__":
  index_matrix = tf.constant([[3,1,2],
                              [2,3,1]
                             ])

  with tf.Session() as sess:
    out_matrix = sess.run(index_matrix_to_pairs(index_matrix))
    print("out_matrix", out_matrix)

    print("range:",sess.run(tf.range(tf.shape(index_matrix)[0])), " shape:", tf.range(3).shape) # [0,1,2], shape:(3,)
    replicated_first_indices = tf.range(tf.shape(index_matrix)[0])

    print("indices:", sess.run(replicated_first_indices)) # [0 1], shape:(2,)
    # 当前的shape为2,在axis=1处插入就是[2,1]
    expand_indices = tf.expand_dims(replicated_first_indices, dim=1),  # shape:[2,1]
    #  (array([[0],
    #         [1]]),)
    out_expand_indices = sess.run(expand_indices)
    print("expand_indices:", out_expand_indices[0], " shape:", out_expand_indices[0].shape)
    """
expand_indices: 
(array([[0],
        [1]]),)  shape: (2, 1)
    """

    # 当前的shape为2,在axis=0处插入就是[1,2]
    print("expand indices2:", sess.run(tf.expand_dims(replicated_first_indices, dim=0)))  # [[0 1]], shape:[1,2]
    """
    expand indices2: [[0 1]]
    """

    replicated_indices = tf.tile(tf.constant(out_expand_indices[0]), [1, 3]) # 第0维上保持不变,第1维上复制3次
    print("replicated_indices:", sess.run(replicated_indices))
    """
replicated_indices: [[0 0 0]
                     [1 1 1]] 
    """
