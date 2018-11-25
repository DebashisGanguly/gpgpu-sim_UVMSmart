# libraries
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import gridspec

#######################
# data extraction section
#######################

# declaration of variables
parent_folder = '../output_logs'

experiment_folder = 'HWPrefetchers'
sub_folders = ['1_0_0_100', '2_0_0_100', '3_0_0_100', '0_0_0_100']

benchmarks = ['backprop', 'bfs', 'hotspot', 'needle', 'pathfinder', 'srad', 'stencil']

rt_0_0_0_100 = []
rt_1_0_0_100 = []
rt_2_0_0_100 = []
rt_3_0_0_100 = []

rbw_0_0_0_100 = []
rbw_1_0_0_100 = []
rbw_2_0_0_100 = []
rbw_3_0_0_100 = []

for b in benchmarks:
	for sf in sub_folders:
		file_name = './' + parent_folder + '/' + experiment_folder + '/' + sf + '/' + b + '.log'
		
		with open(file_name, 'r') as b_file:
			file_content = b_file.read()

			line = re.findall(r"^Tot_kernel_exec_time_and_fault_time.*", file_content, flags=re.MULTILINE)[0]
			rt = float(line[line.find(', ')+2:line.rfind('(us)')])

			line = re.findall(r"Pcie_read_utilization.*", file_content, flags=re.MULTILINE)[0]
			rbw = float(line[line.find(': ')+2:len(line)]) * 12

			if sf == '1_0_0_100':
				rt_1_0_0_100.append(rt)
				rbw_1_0_0_100.append(rbw)
			elif sf == '2_0_0_100':
				rt_2_0_0_100.append(rt)
				rbw_2_0_0_100.append(rbw)
			elif sf == '3_0_0_100':
				rt_3_0_0_100.append(rt)
				rbw_3_0_0_100.append(rbw)
			elif sf == '0_0_0_100':
				rt_0_0_0_100.append(rt)
				rbw_0_0_0_100.append(rbw)			


#######################
# plotting section
#######################

# set width of bar
barWidth = 0.2
 
# Set position of bar on X axis
r1 = np.arange(len(rt_1_0_0_100), dtype=float)

r1a = [0, 1.2, 2.4, 3.6, 4.8, 6.0, 7.2]

for i in range(len(rt_1_0_0_100)):
	r1[i] = r1[i] + 0.2

for i in range(len(rt_1_0_0_100)):
	r1a[i] = r1a[i] + 0.2

r2 = [x + barWidth for x in r1a]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]
r5 = [x + barWidth for x in r4]


plt.figure(1)
font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 12}

plt.rc('font', **font)

plt.figure(figsize=(8,4))

plt.bar(r2, rt_0_0_0_100, hatch="-", color='#ffffff', width=barWidth, edgecolor='black', label='On-demand paging')
plt.bar(r3, rt_3_0_0_100, hatch="+", color='#ffffff', width=barWidth, edgecolor='black', label='Random 4KB along with on-demand paging')         
plt.bar(r4, rt_2_0_0_100, hatch="x", color='#ffffff', width=barWidth, edgecolor='black', label='Sequential local 64KB')
plt.bar(r5, rt_1_0_0_100, hatch="\\", color='#ffffff', width=barWidth, edgecolor='black', label='Spatio-temporal')


plt.xticks([r + 0.3 + barWidth for r in r1a], benchmarks)

ax = plt.gca()
ax.yaxis.grid(b=True, which='major', color='grey', linestyle='-')

ax.set_yscale('log', nonposy='clip')

plt.ylabel('Kernel Execution Time (log scale)')

ax.xaxis.set_ticks_position('none')

# Create legend & Show graphic
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.29), prop={'size': 12})

plt.savefig('./HWPrefetchers/hwprefetchers.png',  dpi=300, bbox_inches="tight")

# Child plot

plt.cla()
plt.clf()

# set width of bar
barWidth = 0.2
 
# Set position of bar on X axis
r1 = np.arange(len(rbw_1_0_0_100), dtype=float)

r1a = [0, 1.2, 2.4, 3.6, 4.8, 6.0, 7.2]

for i in range(len(rbw_1_0_0_100)):
	r1[i] = r1[i] + 0.2

for i in range(len(rbw_1_0_0_100)):
	r1a[i] = r1a[i] + 0.2

r2 = [x + barWidth for x in r1a]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]
r5 = [x + barWidth for x in r4]


plt.figure(2)

font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 12}

plt.rc('font', **font)

plt.figure(figsize=(8,4))

plt.bar(r2, rbw_0_0_0_100, hatch="-", color='#ffffff', width=barWidth, edgecolor='black', label='On-demand paging')
plt.bar(r3, rbw_3_0_0_100, hatch="+", color='#ffffff', width=barWidth, edgecolor='black', label='Random 4KB along with on-demand paging')         
plt.bar(r4, rbw_2_0_0_100, hatch="x", color='#ffffff', width=barWidth, edgecolor='black', label='Sequential local 64KB')
plt.bar(r5, rbw_1_0_0_100, hatch="\\", color='#ffffff', width=barWidth, edgecolor='black', label='Spatio-temporal')


plt.xticks([r + 0.3 + barWidth for r in r1a], benchmarks)

ax = plt.gca()
ax.yaxis.grid(b=True, which='major', color='grey', linestyle='-')

plt.ylabel('Average PCI-e Read Bandwidth (GB/s)')

ax.xaxis.set_ticks_position('none')

# Create legend & Show graphic
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.29), prop={'size': 12})

plt.savefig('./HWPrefetchers/hwprefetchers_read_bw.png',  dpi=300, bbox_inches="tight")

