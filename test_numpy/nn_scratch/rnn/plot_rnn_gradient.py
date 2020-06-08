import numpy as np
import matplotlib.pyplot as plt

#rnn: h(t) = tanh(Wh*h(t-1))
# dht/dh(t-T) = (1-[tanh(Wh*h(t-1))]^2)*Wh
#               * dh(t-1)/dh(t-T)
# 也可简化为:
# dh(t) = (1-[tanh(Wh*h(t-1))]^2) * Wh * dh(t-1)
# 先求dh(t),再求dh(t-1),dh(t-2)
#
def forward_backward_prop(Wh, T):
    ht = [0.5] # 初始ht=0.5
    # 先求ht,再求h(t-1),h(t-2),...
    for _ in range(T):
        ht.append(np.tanh(Wh * ht[-1]))

    # 先求dh(t),再求dh(t-1),dh(t-2)
    dh = 1 # 初始梯度=1
    for t in range(T):
        dh = (1 - ht[-1 - t] ** 2) * Wh * dh

    return ht[-1], dh

T = 10 # sequence length
wlim = 4 # limit of interval over weights Wh

results = []
ws = np.linspace(-wlim, wlim, 1000)
for w in ws:
    ht, dh = forward_backward_prop(w, T)
    results.append((ht, dh))

plt.plot(ws, [r[0] for r in results], label='RNN state')
plt.plot(ws, [r[1] for r in results], label='Gradients')
plt.legend(loc='upper right')
plt.show()
