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
sub_folders = ['1_0_0_110', '1_1_0_110', '1_2_0_110', '1_3_0_110']

benchmarks = ['backprop', 'bfs', 'hotspot', 'needle', 'pathfinder', 'srad', 'stencil']

rt_1_0_0_110 = []
rt_1_1_0_110 = []
rt_1_2_0_110 = []
rt_1_3_0_110 = []

dp_1_0_0_110 = []
dp_1_1_0_110 = []
dp_1_2_0_110 = []
dp_1_3_0_110 = []

ndp_1_0_0_110 = []
ndp_1_1_0_110 = []
ndp_1_2_0_110 = []
ndp_1_3_0_110 = []

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

			if sf == '1_0_0_110':
				rt_1_0_0_110.append(rt)
				dp_1_0_0_110.append(dp)
				ndp_1_0_0_110.append(ndp)
			elif sf == '1_1_0_110':
				rt_1_1_0_110.append(rt)
				dp_1_1_0_110.append(dp)
				ndp_1_1_0_110.append(ndp)
			elif sf == '1_2_0_110':
				rt_1_2_0_110.append(rt)
				dp_1_2_0_110.append(dp)
				ndp_1_2_0_110.append(ndp)
			elif sf == '1_3_0_110':
				rt_1_3_0_110.append(rt)
				dp_1_3_0_110.append(dp)
				ndp_1_3_0_110.append(ndp)			


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

plt.bar(r2, rt_1_0_0_110, hatch="-", color='#ffffff', width=barWidth, edgecolor='black', label='LRU 4KB')
plt.bar(r3, rt_1_3_0_110, hatch="+", color='#ffffff', width=barWidth, edgecolor='black', label='Random 4KB')         
plt.bar(r4, rt_1_2_0_110, hatch="x", color='#ffffff', width=barWidth, edgecolor='black', label='Sequential local 64KB')
plt.bar(r5, rt_1_1_0_110, hatch="\\", color='#ffffff', width=barWidth, edgecolor='black', label='Spatio-temporal')


plt.xticks([r + 0.3 + barWidth for r in r1a], benchmarks)

ax = plt.gca()
ax.yaxis.grid(b=True, which='major', color='grey', linestyle='-')

ax.set_yscale('log', nonposy='clip')

plt.ylabel('Kernel Execution Time (log scale)')

ax.xaxis.set_ticks_position('none')

# Create legend & Show graphic
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.29), prop={'size': 12})

plt.savefig('./EvictionPolicies/evictions.png',  dpi=300, bbox_inches="tight")

# Child plot

plt.cla()
plt.clf()


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


plt.figure(2)
font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 12}

plt.rc('font', **font)

plt.figure(figsize=(8,4))

plt.bar(r2, dp_1_0_0_110, hatch="-", color='#ffffff', width=barWidth, edgecolor='black', label='Dirty Pages Evicted by LRU 4KB')
plt.bar(r2, ndp_1_0_0_110, hatch="o", color='#ffffff', width=barWidth, edgecolor='black', label='Not Dirty Pages Evicted by LRU 4KB', bottom = dp_1_0_0_110)
plt.bar(r3, dp_1_3_0_110, hatch="+", color='#ffffff', width=barWidth, edgecolor='black', label='Dirty Pages Evicted by Random 4KB')
plt.bar(r3, ndp_1_3_0_110, hatch="O", color='#ffffff', width=barWidth, edgecolor='black', label='Not Dirty Pages Evicted by Random 4KB', bottom = dp_1_3_0_110)         
plt.bar(r4, dp_1_2_0_110, hatch="x", color='#ffffff', width=barWidth, edgecolor='black', label='Dirty Pages Evicted by Sequential local 64KB')       
plt.bar(r4, ndp_1_2_0_110, hatch="*", color='#ffffff', width=barWidth, edgecolor='black', label='Not Dirty Pages Evicted by Sequential local 64KB', bottom = dp_1_2_0_110)
plt.bar(r5, dp_1_1_0_110, hatch="\\", color='#ffffff', width=barWidth, edgecolor='black', label='Dirty Pages Evicted by Spatio-temporal')
plt.bar(r5, ndp_1_1_0_110, hatch=".", color='#ffffff', width=barWidth, edgecolor='black', label='Not Dirty Pages Evicted by Spatio-temporal', bottom = dp_1_1_0_110)


plt.xticks([r + 0.3 + barWidth for r in r1a], benchmarks)

ax = plt.gca()
ax.yaxis.grid(b=True, which='major', color='grey', linestyle='-')

plt.ylabel('Number of Pages Evicted')

ax.xaxis.set_ticks_position('none')

# Create legend & Show graphic
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.5), prop={'size': 12})

plt.savefig('./EvictionPolicies/evictions_dirty_not.png',  dpi=300, bbox_inches="tight")
