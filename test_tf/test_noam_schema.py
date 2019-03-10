import numpy as np
import matplotlib.pyplot as plt


def noam_scheme(step, init_lr=1e-3, warmup_steps=4000.):
    '''Noam scheme learning rate decay
    init_lr: initial learning rate. scalar.
    global_step: scalar.
    warmup_steps: scalar. During warmup_steps, learning rate increases
        until it reaches init_lr.
    '''
    """
    init_lr * warmup^0.5 * min(step*warmup^(-1.5), step^(-0.5))
    """
    init = init_lr * warmup_steps ** 0.5
    lr1 =step* warmup_steps**-1.5
    lr2 = step**-0.5
    min_lr = np.min(np.hstack((np.expand_dims(lr1,-1),np.expand_dims(lr2,-1))), axis=1, keepdims=False)
    return init*min_lr

steps= np.arange(1, 100000)
lr = noam_scheme(steps)
max_lr = np.max(lr)
min_lr =np.min(lr)
print("max_lr:", max_lr) # =init_lr = 1e-4
print("min_lr:",min_lr)
plt.plot(steps,lr)
plt.show()

