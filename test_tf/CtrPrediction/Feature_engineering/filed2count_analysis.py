import matplotlib.pyplot as plt
import pickle
from collections import Counter
import pandas as pd
# load field2count dictionaries
dataPath = '../data/'
with open(dataPath+'field2count/C14.pkl','rb') as f:
    C14 = pickle.load(f)

with open(dataPath+'field2count/C17.pkl','rb') as f:
    C17 = pickle.load(f)

with open(dataPath+'field2count/C19.pkl','rb') as f:
    C19 = pickle.load(f)

with open(dataPath+'field2count/C21.pkl','rb') as f:
    C21 = pickle.load(f)

with open(dataPath+'field2count/site_id.pkl','rb') as f:
    site_id = pickle.load(f)

with open(dataPath+'field2count/site_domain.pkl','rb') as f:
    site_domain = pickle.load(f)

with open(dataPath+'field2count/app_id.pkl','rb') as f:
    app_id = pickle.load(f)

with open(dataPath+'field2count/app_domain.pkl','rb') as f:
    app_domain = pickle.load(f)

with open(dataPath+'field2count/device_model.pkl','rb') as f:
    device_model = pickle.load(f)

with open(dataPath+'field2count/device_id.pkl','rb') as f:
    device_id = pickle.load(f)

with open(dataPath+'field2count/device_ip.pkl','rb') as f:
    device_ip = pickle.load(f)
#C21:{value: value_freq_count}
field = C21
result = Counter(field.values())
b = sorted(result.items(), key=lambda x:x[1], reverse=True)

frequency = []
[frequency.append(x[0]) for x in b]
count = []
[count.append(x[1]) for x in b]

fig1 = plt.figure(1)
ax1 = fig1.add_subplot(1,1,1)
ax1.plot(frequency, count, 'o-', color = 'blue')
ax1.set_xscale('log') # 坐标轴按log进行缩放,但值并没有取log
ax1.grid(True, which='major', axis='both')
ax1.set_xlabel('frequency',fontsize = 15)
ax1.set_ylabel('counts', fontsize = 15)
ax1.set_title('field of C21', fontsize=15)
plt.show()
