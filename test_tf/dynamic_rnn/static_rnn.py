import tensorflow as tf
import numpy as np

# 创建输入数据
batch_size=2
hidden_size=3
timestep_size=10
input_dim=2

np.random.seed(0)
# X:[batch,timestep,input_dim]
X = np.random.randn(batch_size, timestep_size, input_dim)
# 第一个长度为10, 第二个example长度为6
X[1, 6:] = 0
X_lengths = [timestep_size, 6]
print("X:",X)
#sys.exit(-1)
cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, state_is_tuple=True)
_init_state= cell.zero_state(batch_size, dtype=tf.float32)
print("init_state:", _init_state)
"""
c:[batch, hidden_size]
init_state: LSTMStateTuple(c=<tf.Tensor 'LSTMCellZeroState/zeros:0' shape=(2, 3) dtype=float32>, h=<tf.Tensor 'LSTMCellZeroState/zeros_1:0' shape=(2, 3) dtype=float32>)
"""

X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)

input_tensor_list = tf.unstack(X_tensor, num=timestep_size, axis=1) # tf.split()
print("input:", input_tensor_list) # 10个[batch=2, dim=2]的tensor
output_all_hidden_states, last_cell_and_hidden_state = tf.nn.static_rnn(  # 准确的讲是: all_hidden_state, last_cell_and_hidden_state
    cell=cell,
    inputs=input_tensor_list,
    dtype=tf.float32,
    initial_state=_init_state)

# 也可以这样写
state_h = last_cell_and_hidden_state.h
state_c = last_cell_and_hidden_state.c

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    [output_all_hidden_states_out, last_cell_and_hidden_state_out] = sess.run([output_all_hidden_states, last_cell_and_hidden_state])
    print("output_hidden:", output_all_hidden_states_out)
    print("last_cell_and_hidden:", last_cell_and_hidden_state_out)
    #print("state_h:", sess.run(state_h)) # (batch, hidden)
    #print("state_c:", sess.run(state_c)) # (batch, hidden)

"""
output_hidden: time_step* [batch, hidden_size]
[
 array([[-0.08043327,  0.03942952, -0.08093768],
       [-0.02433698,  0.0708902 , -0.00701448]], dtype=float32), 
 array([[-0.13673759,  0.03610585, -0.1539865 ],
       [-0.03445462,  0.12702672, -0.00038486]], dtype=float32), 
 array([[-0.19220088,  0.10089475, -0.12508702],
       [-0.07478397,  0.13480707, -0.03433445]], dtype=float32), 
 array([[-0.16312504,  0.13762623, -0.0376162 ],
       [ 0.07475603,  0.021191  ,  0.0817403 ]], dtype=float32), 
 array([[-0.09182419,  0.17341809, -0.00676246],
       [ 0.02189413,  0.02956883,  0.07144436]], dtype=float32), 
 array([[0.00337426, 0.24886863, 0.05470206],
       [0.03500113, 0.15059653, 0.06972466]], dtype=float32), 
 array([[0.07244096, 0.0595331 , 0.13098729],
       [0.0253542 , 0.12337567, 0.09532985]], dtype=float32), 
 array([[0.05225845, 0.07244087, 0.09814122],
       [0.02119715, 0.11334925, 0.0899382 ]], dtype=float32), 
 array([[-0.05749295,  0.10248318,  0.01442759],
       [ 0.01830486,  0.10407028,  0.08498989]], dtype=float32), 
 array([[-0.09584065,  0.03794328, -0.05328367],
       [ 0.01613625,  0.09559111,  0.07984833]], dtype=float32)
]
       
last_cell_and_hidden: c:[batch, hidden_size] 
LSTMStateTuple(c=array([[-0.22451545,  0.08021037, -0.0838517 ],
                        [ 0.03166688,  0.19583946,  0.16012913]], dtype=float32), 
               h=array([[-0.09584065,  0.03794328, -0.05328367],
                        [ 0.01613625,  0.09559111,  0.07984833]], dtype=float32))
                        
"""
