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

experiment_folder = 'EvictionPolicies'
sub_folders = ['LRU', 'TBN', 'SL', 'Random']

benchmarks = ['backprop', 'bfs', 'fdtd', 'hotspot', 'nw', 'pathfinder', 'srad']

rt_LRU = []
rt_TBN = []
rt_SL = []
rt_Random = []

dp_LRU = []
dp_TBN = []
dp_SL = []
dp_Random = []

ndp_LRU = []
ndp_TBN = []
ndp_SL = []
ndp_Random = []

for b in benchmarks:
	for sf in sub_folders:
		file_name = './' + parent_folder + '/' + experiment_folder + '/' + sf + '/' + b + '.log'
		
		with open(file_name, 'r') as b_file:
			file_content = b_file.read()

			line = re.findall(r"^Tot_kernel_exec_time_and_fault_time.*", file_content, flags=re.MULTILINE)[0]
			rt = float(line[line.find(', ')+2:line.rfind('(us)')])

			line = re.findall(r"^Page_validate.*Page_evict_dirty.*Page_evict_not_dirty.*", file_content, flags=re.MULTILINE)[0]
			dp = float(line.split()[3])
			ndp = float(line.split()[5])

			if sf == 'LRU':
				rt_LRU.append(rt)
				dp_LRU.append(dp)
				ndp_LRU.append(ndp)
			elif sf == 'TBN':
				rt_TBN.append(rt)
				dp_TBN.append(dp)
				ndp_TBN.append(ndp)
			elif sf == 'SL':
				rt_SL.append(rt)
				dp_SL.append(dp)
				ndp_SL.append(ndp)
			elif sf == 'Random':
				rt_Random.append(rt)
				dp_Random.append(dp)
				ndp_Random.append(ndp)			


#######################
# plotting section
#######################

# set width of bar
barWidth = 0.3
 
# Set position of bar on X axis
r1 = np.arange(len(rt_LRU), dtype=float)

r1a = [0, 1.6, 3.2, 4.8, 6.4, 8.0, 9.6]

for i in range(len(rt_LRU)):
	r1[i] = r1[i] + 0.3

for i in range(len(rt_LRU)):
	r1a[i] = r1a[i] + 0.3

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

plt.figure(figsize=(10,4))

plt.bar(r2, rt_LRU, hatch="--", color='r', width=barWidth, edgecolor='black', label='LRU 4KB')
plt.bar(r3, rt_Random, hatch="++", color='c', width=barWidth, edgecolor='black', label='Random 4KB')         
plt.bar(r4, rt_SL, hatch="xx", color='yellow', width=barWidth, edgecolor='black', label='Sequential-local 64KB')
plt.bar(r5, rt_TBN, hatch="\\\\", color='g', width=barWidth, edgecolor='black', label='Tree-based neighborhood')


plt.xticks([r + 0.3 + barWidth for r in r1a], benchmarks)

ax = plt.gca()
ax.yaxis.grid(b=True, which='major', color='grey', linestyle='-')

plt.ylabel('Kernel Execution Time (us)')

ax.xaxis.set_ticks_position('none')

# Create legend & Show graphic
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.36), prop={'size': 12})

plt.savefig('../plots/EvictionPolicies/evictions.png',  dpi=300, bbox_inches="tight")

# Child plot

plt.cla()
plt.clf()

plt.figure(2)

plt.rc('font', **font)

plt.rcParams['hatch.linewidth'] = 1.5

plt.figure(figsize=(10,4))

plt.bar(r2, dp_LRU, hatch="--", color='r', width=barWidth, edgecolor='black', label='Dirty Pages Evicted by LRU 4KB')
plt.bar(r2, ndp_LRU, hatch="oo", color='orangered', width=barWidth, edgecolor='black', label='Not Dirty Pages Evicted by LRU 4KB', bottom = dp_LRU)
plt.bar(r3, dp_Random, hatch="++", color='c', width=barWidth, edgecolor='black', label='Dirty Pages Evicted by Random 4KB')
plt.bar(r3, ndp_Random, hatch="OO", color='cyan', width=barWidth, edgecolor='black', label='Not Dirty Pages Evicted by Random 4KB', bottom = dp_Random)         
plt.bar(r4, dp_SL, hatch="xx", color='yellow', width=barWidth, edgecolor='black', label='Dirty Pages Evicted by Sequential-local 64KB')       
plt.bar(r4, ndp_SL, hatch="**", color='#ffffff', width=barWidth, edgecolor='black', label='Not Dirty Pages Evicted by Sequential-local 64KB', bottom = dp_SL)
plt.bar(r5, dp_TBN, hatch="\\\\", color='g', width=barWidth, edgecolor='black', label='Dirty Pages Evicted by Tree-based neighborhood')
plt.bar(r5, ndp_TBN, hatch="..", color='lime', width=barWidth, edgecolor='black', label='Not Dirty Pages Evicted by Tree-based neighborhood', bottom = dp_TBN)


plt.xticks([r + 0.3 + barWidth for r in r1a], benchmarks)

ax = plt.gca()
ax.yaxis.grid(b=True, which='major', color='grey', linestyle='-')

plt.ylabel('Number of Pages Evicted')

ax.xaxis.set_ticks_position('none')

# Create legend & Show graphic
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.68), prop={'size': 12})

plt.savefig('../plots/EvictionPolicies/evictions_dirty_not.png',  dpi=300, bbox_inches="tight")
