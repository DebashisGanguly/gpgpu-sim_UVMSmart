import matplotlib.pyplot as plt
import math

k = 1024.0
x = [4*k, 8*k, 16*k, 32*k, 64*k, 128*k, 256*k, 512*k, k*k, 2*k*k]
y = [2.337,3.558,4.915,6.532,8.15,9.489,10.24,10.8,11.017,11.13];

text = ["4KB", "8KB", "16KB", "32KB", "64KB", "128KB", "256KB", "512KB", "1MB", "2MB"]


for i in range(len(x)):
	x[i] = x[i]+1024*1024

font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 12}

plt.rc('font', **font)

plt.figure(figsize=(5,4))

ax = plt.gca()

ax.yaxis.grid(b=True, which='major', color='grey', linestyle='-')
ax.yaxis.grid(b=True, which='minor', color='grey', linestyle='--') 

plt.minorticks_on()

plt.plot(x, y, '.b--', marker=(5, 2), color='black', markersize=13)

plt.ylabel('PCI-e Transfer Rate (GB/s)')
plt.xlabel('Transfer Size')

# Add the text manually
plt.text(x[0]*1.05, y[0], text[0])
plt.text(x[1]*1.05, y[1], text[1])
plt.text(x[2]*1.05, y[2], text[2])
plt.text(x[3]*1.05, y[3], text[3])
plt.text(x[4]*1.05, y[4], text[4])
plt.text(x[5]*1.05, y[5], text[5])
plt.text(x[6]/1.25, y[6]+0.3, text[6])
plt.text(x[7], y[7]+0.3, text[7])
plt.text(x[8], y[8]+0.3, text[8])
plt.text(x[9], y[9]+0.3, text[9])



plt.axis('auto')

# x with no number shown
plt.xticks([])

plt.savefig('./pcie_version/curve_fitting.png',  dpi=300)
