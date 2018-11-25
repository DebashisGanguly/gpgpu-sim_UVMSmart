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

experiment_folder = 'EvictionPolicy_HWPrefetcher_Combo'
sub_folders = ['1_0_0_110', '1_3_3_110', '1_2_2_110', '1_1_1_110']

benchmarks = ['backprop', 'bfs', 'hotspot', 'needle', 'pathfinder', 'srad', 'stencil']

rt_1_0_0_110 = []
rt_1_3_3_110 = []
rt_1_2_2_110 = []
rt_1_1_1_110 = []

for b in benchmarks:
	for sf in sub_folders:
		file_name = './' + parent_folder + '/' + experiment_folder + '/' + sf + '/' + b + '.log'
		
		with open(file_name, 'r') as b_file:
			file_content = b_file.read()

			line = re.findall(r"^Tot_kernel_exec_time_and_fault_time.*", file_content, flags=re.MULTILINE)[0]
			rt = float(line[line.find(', ')+2:line.rfind('(us)')])

			if sf == '1_0_0_110':
				rt_1_0_0_110.append(rt)
			elif sf == '1_3_3_110':
				rt_1_3_3_110.append(rt)
			elif sf == '1_2_2_110':
				rt_1_2_2_110.append(rt)
			elif sf == '1_1_1_110':
				rt_1_1_1_110.append(rt)			

#rt_1_0_0_100 = np.array(np.divide(rt_1_0_0_100, rt_1_0_0_100))
#rt_0_0_0_100 = np.array(np.divide(rt_0_0_0_100, rt_1_0_0_100))
#rt_2_0_0_100 = np.array(np.divide(rt_2_0_0_100, rt_1_0_0_100))
#rt_3_0_0_100 = np.array(np.divide(rt_3_0_0_100, rt_1_0_0_100))


#######################
# plotting section
#######################

# set width of bar
barWidth = 0.2
 
# Set position of bar on X axis
r1 = np.arange(len(rt_1_0_0_110), dtype=float)

r1a = [0, 1.2, 2.4, 3.6, 4.8, 6.0, 7.2]

for i in range(len(rt_1_0_0_110)):
	r1[i] = r1[i] + 0.2

for i in range(len(rt_1_0_0_110)):
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

plt.bar(r2, rt_1_0_0_110, hatch="-", color='#ffffff', width=barWidth, edgecolor='black', label='STP before eviction + LRU 4KB eviction + OD after eviction')
plt.bar(r3, rt_1_3_3_110, hatch="+", color='#ffffff', width=barWidth, edgecolor='black', label='STP before eviction + Random 4KB eviction + RP after eviction')         
plt.bar(r4, rt_1_2_2_110, hatch="x", color='#ffffff', width=barWidth, edgecolor='black', label='STP before eviction + SL 64KB eviction + SLP after eviction')
plt.bar(r5, rt_1_1_1_110, hatch="\\", color='#ffffff', width=barWidth, edgecolor='black', label='STP before eviction + ST eviction + STP after eviction')


plt.xticks([r + 0.3 + barWidth for r in r1a], benchmarks)

ax = plt.gca()
ax.yaxis.grid(b=True, which='major', color='grey', linestyle='-')

ax.set_yscale('log', nonposy='clip')

plt.ylabel('Kernel Execution Time (log scale)')

ax.xaxis.set_ticks_position('none')

# Create legend & Show graphic
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.29), prop={'size': 12})

plt.savefig('./EvictionPolicy_HWPrefetcher_Combo/combo.png',  dpi=300, bbox_inches="tight")
