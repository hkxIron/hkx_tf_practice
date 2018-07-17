
import tensorflow as tf
 
a=tf.get_variable(name='a',
                  shape=[3,4],
                  dtype=tf.float32,
                  initializer=tf.random_uniform_initializer(minval=-1,maxval=1))
b=tf.argmax(input=a,axis=0) #argmax会降维
c=tf.argmax(input=a,dimension=1)   #此处用dimesion或用axis是一样的
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
print(sess.run(a))
#[[ 0.04261756 -0.34297419 -0.87816691 -0.15430689]
# [ 0.18663144  0.86972666 -0.06103253  0.38307118]
# [ 0.84588599 -0.45432305 -0.39736366  0.38526249]]
print(sess.run(b))
#[2 1 1 2]
print(sess.run(c))
#[0 1 0]
