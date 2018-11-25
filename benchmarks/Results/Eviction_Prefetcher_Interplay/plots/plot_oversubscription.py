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

experiment_folder = 'Oversubscription'
sub_folders = ['1_0_0_100', '1_0_0_110', '1_0_0_120', '1_0_0_110_5', '1_0_0_110_10']

benchmarks = ['backprop', 'bfs', 'hotspot', 'needle', 'pathfinder', 'srad', 'stencil']

rt_1_0_0_100 = []
rt_1_0_0_110 = []
rt_1_0_0_120 = []
rt_1_0_0_110_5 = []
rt_1_0_0_110_10 = []

pt4k_1_0_0_100 = []
pt4k_1_0_0_110 = []
pt4k_1_0_0_120 = []
pt4k_1_0_0_110_5 = []
pt4k_1_0_0_110_10 = []

for b in benchmarks:
	for sf in sub_folders:
		file_name = './' + parent_folder + '/' + experiment_folder + '/' + sf + '/' + b + '.log'
		
		with open(file_name, 'r') as b_file:
			file_content = b_file.read()

			line = re.findall(r"^Tot_kernel_exec_time_and_fault_time.*", file_content, flags=re.MULTILINE)[0]
			rt = float(line[line.find(', ')+2:line.rfind('(us)')])

			if sf == '1_0_0_100':
				rt_1_0_0_100.append(rt)
				pt4k_1_0_0_100.append(file_content.count('Sz: 4096 	 Sm: 0 	 T: memcpy_h2d'))
			elif sf == '1_0_0_110':
				rt_1_0_0_110.append(rt)
				pt4k_1_0_0_110.append(file_content.count('Sz: 4096 	 Sm: 0 	 T: memcpy_h2d'))
			elif sf == '1_0_0_120':
				rt_1_0_0_120.append(rt)
				pt4k_1_0_0_120.append(file_content.count('Sz: 4096 	 Sm: 0 	 T: memcpy_h2d'))
			elif sf == '1_0_0_110_5':
				rt_1_0_0_110_5.append(rt)
				pt4k_1_0_0_110_5.append(file_content.count('Sz: 4096 	 Sm: 0 	 T: memcpy_h2d'))
			elif sf == '1_0_0_110_10':
				rt_1_0_0_110_10.append(rt)
				pt4k_1_0_0_110_10.append(file_content.count('Sz: 4096 	 Sm: 0 	 T: memcpy_h2d'))
			

rt_1_0_0_110 = np.array(np.divide(rt_1_0_0_110, rt_1_0_0_100))
rt_1_0_0_120 = np.array(np.divide(rt_1_0_0_120, rt_1_0_0_100))
rt_1_0_0_110_5 = np.array(np.divide(rt_1_0_0_110_5, rt_1_0_0_100))
rt_1_0_0_110_10 = np.array(np.divide(rt_1_0_0_110_10, rt_1_0_0_100))

pt4k_1_0_0_110 = np.array(np.divide(pt4k_1_0_0_110, pt4k_1_0_0_100))
pt4k_1_0_0_120 = np.array(np.divide(pt4k_1_0_0_120, pt4k_1_0_0_100))
pt4k_1_0_0_110_5 = np.array(np.divide(pt4k_1_0_0_110_5, pt4k_1_0_0_100))
pt4k_1_0_0_110_10 = np.array(np.divide(pt4k_1_0_0_110_10, pt4k_1_0_0_100))

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

plt.bar(r2, rt_1_0_0_110, hatch="-", color='#ffffff', width=barWidth, edgecolor='black', label='Working set == gddr size * 110%')
plt.bar(r3, rt_1_0_0_120, hatch="+", color='#ffffff', width=barWidth, edgecolor='black', label='Working set == gddr size * 120%')         
plt.bar(r4, rt_1_0_0_110_5, hatch="x", color='#ffffff', width=barWidth, edgecolor='black', label='Working set == gddr size * 110% and pre-eviction at 95%')
plt.bar(r5, rt_1_0_0_110_10, hatch="\\", color='#ffffff', width=barWidth, edgecolor='black', label='Working set == gddr size * 110% and pre-eviction at 90%')

plt.text(r1[6]+1.15, 30, int(rt_1_0_0_110[6]), bbox=dict(facecolor='red', alpha=1))
plt.text(r1[6]+1.35, 40, int(rt_1_0_0_120[6]), bbox=dict(facecolor='red', alpha=1))
plt.text(r1[6]+1.55, 50, int(rt_1_0_0_110_5[6]), bbox=dict(facecolor='red', alpha=1))
plt.text(r1[6]+1.75, 60, int(rt_1_0_0_110_10[6]), bbox=dict(facecolor='red', alpha=1))

plt.xticks([r + 0.3 + barWidth for r in r1a], benchmarks)

ax = plt.gca()
ax.yaxis.grid(b=True, which='major', color='grey', linestyle='-')
ax.axes.set_ylim([0,65])

plt.ylabel('Kernel Execution Time\n(Normalized to no oversubscription)')

ax.xaxis.set_ticks_position('none')

# Create legend & Show graphic
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.37), prop={'size': 12})

plt.savefig('./Oversubscription/oversub.png',  dpi=300, bbox_inches="tight")

# Child plot
plt.cla()
plt.clf()


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


font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 12}

plt.rc('font', **font)

f, (ax, ax2) = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(10,6))
gs  = gridspec.GridSpec(2, 1, height_ratios=[3, 2])
ax2 = plt.subplot(gs[1])
ax = plt.subplot(gs[0])
f.subplots_adjust(top=0.85)

ax.bar(r2, pt4k_1_0_0_110, hatch="-", color='#ffffff', width=barWidth, edgecolor='black', label='Working set == gddr size * 110%')
ax.bar(r3, pt4k_1_0_0_120, hatch="+", color='#ffffff', width=barWidth, edgecolor='black', label='Working set == gddr size * 120%')         
ax.bar(r4, pt4k_1_0_0_110_5, hatch="x", color='#ffffff', width=barWidth, edgecolor='black', label='Working set == gddr size * 110% and pre-eviction at 95%')
ax.bar(r5, pt4k_1_0_0_110_10, hatch="\\", color='#ffffff', width=barWidth, edgecolor='black', label='Working set == gddr size * 110% and pre-eviction at 90%')

ax2.bar(r2, pt4k_1_0_0_110, hatch="-", color='#ffffff', width=barWidth, edgecolor='black', label='Working set == gddr size * 110%')
ax2.bar(r3, pt4k_1_0_0_120, hatch="+", color='#ffffff', width=barWidth, edgecolor='black', label='Working set == gddr size * 120%')         
ax2.bar(r4, pt4k_1_0_0_110_5, hatch="x", color='#ffffff', width=barWidth, edgecolor='black', label='Working set == gddr size * 110% and pre-eviction at 95%')
ax2.bar(r5, pt4k_1_0_0_110_10, hatch="\\", color='#ffffff', width=barWidth, edgecolor='black', label='Working set == gddr size * 110% and pre-eviction at 90%')

ax.set_xticklabels([])

ax2.set_xticks([r + 0.3 + barWidth for r in r1a])
ax2.set_xticklabels(benchmarks)

d = .015  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((-d, +d), (-0.01-d, 0.01+d), **kwargs)        # top-left diagonal
ax.plot((1 - d, 1 + d), (-0.01-d, 0.01+d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

ax.yaxis.grid(b=True, which='major', color='grey', linestyle='-')
ax2.yaxis.grid(b=True, which='major', color='grey', linestyle='-')

ax2.set_xlim([0,8.6])
ax2.set_ylim([0,250])

ax.set_xlim([0,8.6])
ax.set_ylim([1000,1500])

ax2.set_ylabel('Number of 4KB Page Transfers\n(Normalized to no oversubscription)')
ax2.yaxis.set_label_coords(-0.07,1.5)

ax.xaxis.set_ticks_position('none')
ax2.xaxis.set_ticks_position('none')

# Create legend & Show graphic
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), prop={'size': 12})

plt.savefig('./Oversubscription/oversub_4k.png',  dpi=300, bbox_inches="tight")

