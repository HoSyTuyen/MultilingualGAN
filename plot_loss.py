import pickle
import matplotlib.pyplot as plt
import numpy as np

path_exp = "/mmlabworkspace/WorkSpaces/danhnt/tuyensh/HWGAN/HWGAN/Cinamon_Gray_BaselineOctConv_ngf64_lpips_1_10_results/"

name_exp = path_exp.split('/')[-2]
print(name_exp)
loss_hist = path_exp + "weights/train_hist.pkl" 

with open(loss_hist, 'rb') as f:
    data = pickle.load(f)

num_step = len(data['Disc_loss'])
step = int(num_step/300)
print(step)

Disc_loss = [data['Disc_loss'][i] for i in range(0,num_step,step)]
Gen_loss = [data['Gen_loss'][i] for i in range(0,num_step,step)]
Con_loss = [data['Con_loss'][i] for i in range(0,num_step,step)]

epoch = np.arange(0, 300, 1)
plt.plot(epoch, Disc_loss, label='Disc_loss')
plt.plot(epoch, Gen_loss, label='Gen_loss')
plt.plot(epoch, Con_loss, label='Con_loss')
plt.legend()
plt.savefig(path_exp + '/' + name_exp + '_train_hist.png')