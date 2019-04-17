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

experiment_folder = 'ReserveLRU'
sub_folders = ['SL_110', 'TBN_110', 'TBN_110_Rsv10', 'TBN_110_Rsv20']

benchmarks = ['backprop', 'bfs', 'fdtd', 'hotspot', 'nw', 'pathfinder', 'srad']

rt_SL_110 = []
rt_TBN_110 = []
rt_TBN_110_Rsv10 = []
rt_TBN_110_Rsv20 = []

avg_th_SL_110 = []
avg_th_TBN_110 = []
avg_th_TBN_110_Rsv10 = []
avg_th_TBN_110_Rsv20 = []

std_th_SL_110 = []
std_th_TBN_110 = []
std_th_TBN_110_Rsv10 = []
std_th_TBN_110_Rsv20 = []

for b in benchmarks:
	for sf in sub_folders:
		file_name = './' + parent_folder + '/' + experiment_folder + '/' + sf + '/' + b + '.log'
		
		with open(file_name, 'r') as b_file:
			file_content = b_file.read()

			line = re.findall(r"^Tot_kernel_exec_time_and_fault_time.*", file_content, flags=re.MULTILINE)[0]
			rt = float(line[line.find(', ')+2:line.rfind('(us)')])

			line = re.findall(r"^Page_tot_thrash.*", file_content, flags=re.MULTILINE)[0]
			tot_th = float(line.split()[1])

			avg_th = 0
			std_th = 0

			if tot_th != 0:
				page_thrash = []

				for l in re.findall(r"^Page.*Thrashed.*", file_content, flags=re.MULTILINE):
					page_thrash.append(float(l.split()[3]))

				avg_th = np.mean(page_thrash)
				std_th = np.std(page_thrash)

			if sf == 'SL_110':
				rt_SL_110.append(rt)
				avg_th_SL_110.append(avg_th)
				std_th_SL_110.append(std_th)
			elif sf == 'TBN_110':
				rt_TBN_110.append(rt)
				avg_th_TBN_110.append(avg_th)
				std_th_TBN_110.append(std_th)
			elif sf == 'TBN_110_Rsv10':
				rt_TBN_110_Rsv10.append(rt)
				avg_th_TBN_110_Rsv10.append(avg_th)
				std_th_TBN_110_Rsv10.append(std_th)
			elif sf == 'TBN_110_Rsv20':
				rt_TBN_110_Rsv20.append(rt)
				avg_th_TBN_110_Rsv20.append(avg_th)
				std_th_TBN_110_Rsv20.append(std_th)				


#######################
# plotting section
#######################

# set width of bar
barWidth = 0.2
 
# Set position of bar on X axis
r1 = np.arange(len(rt_TBN_110), dtype=float)

r1a = [0, 1.2, 2.4, 3.6, 4.8, 6.0, 7.2]

for i in range(len(rt_TBN_110)):
	r1[i] = r1[i] + 0.2

for i in range(len(rt_TBN_110)):
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

plt.rcParams['hatch.linewidth'] = 1.5

plt.figure(figsize=(8,4))

plt.bar(r2, rt_SL_110, hatch="xx", color='r', width=barWidth, edgecolor='black', label='SL 64KB eviction + SL prefetcher after eviction')
plt.bar(r3, rt_TBN_110, hatch="\\\\", color='c', width=barWidth, edgecolor='black', label='TBN eviction + TBN prefetcher after eviction')
plt.bar(r4, rt_TBN_110_Rsv10, hatch="OO", color='yellow', width=barWidth, edgecolor='black', label='TBN eviction + TBN prefetcher after eviction\n + Reserve 10% of LRU page list from eviction')
plt.bar(r5, rt_TBN_110_Rsv20, hatch="oo", color='g', width=barWidth, edgecolor='black', label='TBN eviction + TBN prefetcher after eviction\n + Reserve 20% of LRU page list from eviction')

plt.xticks([r + 0.3 + barWidth for r in r1a], benchmarks)

ax = plt.gca()
ax.yaxis.grid(b=True, which='major', color='grey', linestyle='-')

plt.ylabel('Kernel Execution Time (us)')

ax.xaxis.set_ticks_position('none')

# Create legend & Show graphic
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.48), prop={'size': 12})

plt.savefig('../plots/ReserveLRU/reservelru.png',  dpi=300, bbox_inches="tight")


# Child plot

plt.cla()
plt.clf()

# set width of bar
barWidth = 0.2
 
# Set position of bar on X axis
r1 = np.arange(len(avg_th_TBN_110), dtype=float)

r1a = [0, 1.2, 2.4, 3.6, 4.8, 6.0, 7.2]

for i in range(len(avg_th_TBN_110)):
	r1[i] = r1[i] + 0.2

for i in range(len(avg_th_TBN_110)):
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

plt.rcParams['hatch.linewidth'] = 1.5

plt.figure(figsize=(8,4))

plt.bar(r2, avg_th_SL_110, yerr=std_th_SL_110, hatch="xx", ecolor='black', color='r', width=barWidth, edgecolor='black', label='SL 64KB eviction + TBN prefetcher after eviction')
plt.bar(r3, avg_th_TBN_110, yerr=std_th_TBN_110, hatch="\\\\", ecolor='black', color='c', width=barWidth, edgecolor='black', label='TBN eviction + TBN prefetcher after eviction')
plt.bar(r4, avg_th_TBN_110_Rsv10, yerr=std_th_TBN_110_Rsv10, hatch="OO", ecolor='black', color='yellow', width=barWidth, edgecolor='black', label='TBN eviction + TBN prefetcher after eviction\n + Reserve 10% of LRU page list from eviction')
plt.bar(r5, avg_th_TBN_110_Rsv20, yerr=std_th_TBN_110_Rsv20, hatch="oo", ecolor='black', color='g', width=barWidth, edgecolor='black', label='TBN eviction + TBN prefetcher after eviction\n + Reserve 20% of LRU page list from eviction')

plt.xticks([r + 0.3 + barWidth for r in r1a], benchmarks)

ax = plt.gca()
ax.yaxis.grid(b=True, which='major', color='grey', linestyle='-')

plt.ylabel('Average Thrashing per Page')

ax.xaxis.set_ticks_position('none')

# Create legend & Show graphic
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.48), prop={'size': 12})

plt.savefig('../plots/ReserveLRU/reservelru_thrashing.png',  dpi=300, bbox_inches="tight")
