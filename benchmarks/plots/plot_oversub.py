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

experiment_folder = 'OversubscriptionPercentage'
sub_folders = ['0', '110', '125']

experiment_folder2 = 'Pre-evictionThresold'
sub_folders2 = ['0', '5', '10']

benchmarks = ['bfs', 'backprop', 'hotspot', 'needle', 'pathfinder', 'stencil', 'srad']

k_s_g0 = []
k_s_g1 = []
k_s_g2 = []


k_s_f1 = []
k_s_f2 = []

fk_c_g0 = []
fk_c_g1 = []
fk_c_g2 = []

fk_c_f1 = []
fk_c_f2 = []

for b in benchmarks:
	for sf in sub_folders:
		file_name = './' + parent_folder + '/' + experiment_folder + '/' + sf + '/' + b + '.log'
		
		with open(file_name, 'r') as b_file:
			file_content = b_file.read()

			line = re.findall(r"^Tot_kernel_exec_time_and_fault_time.*", file_content, flags=re.MULTILINE)[0]
			k = float(line[line.find(', ')+2:line.rfind('(us)')])

			if sf == '0':
				k_s_g0.append(k)
				fk_c_g0.append(file_content.count('Sz: 4096 	 Sm: 0 	 T: memcpy_h2d'))
			elif sf == '110':
				k_s_g1.append(k)
				fk_c_g1.append(file_content.count('Sz: 4096 	 Sm: 0 	 T: memcpy_h2d'))
			else:
				k_s_g2.append(k)
				fk_c_g2.append(file_content.count('Sz: 4096 	 Sm: 0 	 T: memcpy_h2d'))




for b in benchmarks:
	for sf in sub_folders2:
		file_name = './' + parent_folder + '/' + experiment_folder2 + '/' + sf + '/' + b + '.log'
		
		#print file_name
		with open(file_name, 'r') as b_file:
			file_content = b_file.read()

			line = re.findall(r"^Tot_kernel_exec_time_and_fault_time.*", file_content, flags=re.MULTILINE)[0]
			k = float(line[line.find(', ')+2:line.rfind('(us)')])

			if sf == '5':
				k_s_f1.append(k)
				fk_c_f1.append(file_content.count('Sz: 4096 	 Sm: 0 	 T: memcpy_h2d'))
			elif sf == '10':
				k_s_f2.append(k)
				fk_c_f2.append(file_content.count('Sz: 4096 	 Sm: 0 	 T: memcpy_h2d'))

			
			

#######################
# plotting section
#######################

# set width of bar
barWidth = 0.2
 
# Set position of bar on X axis
r1 = np.arange(len(k_s_g0), dtype=float)

r1a = [0, 1.2, 2.4, 3.6, 4.8, 6.0, 7.2]

for i in range(len(k_s_g0)):
	r1[i] = r1[i] + 0.2

for i in range(len(k_s_g0)):
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

plt.bar(r1a, k_s_g0, hatch=".", color='#ffffff', width=barWidth, edgecolor='black', label='Working set == gddr size * 100%')            
plt.bar(r2, k_s_g1, hatch="+", color='#ffffff', width=barWidth, edgecolor='black', label='Working set == gddr size * 110%')
plt.bar(r3, k_s_g2, hatch="|", color='#ffffff', width=barWidth, edgecolor='black', label='Working set == gddr size * 125%')         
plt.bar(r4, k_s_f1, hatch="\\", color='#ffffff', width=barWidth, edgecolor='black', label='Working set == gddr size * 110% and pre-eviction at 95%')
plt.bar(r5, k_s_f2, hatch="O", color='#ffffff', width=barWidth, edgecolor='black', label='Working set == gddr size * 110% and pre-eviction at 90%')


plt.xticks([r + 0.3 + barWidth for r in r1a], benchmarks)

ax = plt.gca()
ax.yaxis.grid(b=True, which='major', color='grey', linestyle='-')

ax.set_yscale('log')

plt.ylabel('Kernel Execution Time (Log Scale)')

ax.axes.set_xlim([0,8.6])

ax.xaxis.set_ticks_position('none')

# Create legend & Show graphic
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.29), prop={'size': 12})

plt.savefig('./oversubscription/oversub.png',  dpi=300, bbox_inches="tight")


plt.cla()
plt.clf()


font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 12}

plt.rc('font', **font)

f, (ax, ax2) = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(8,4))
gs  = gridspec.GridSpec(2, 1, height_ratios=[1, 7])
ax2 = plt.subplot(gs[1])
ax = plt.subplot(gs[0])
f.subplots_adjust(top=0.85)

r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]
r5 = [x + barWidth for x in r4]

ax.bar(r1, fk_c_g1, hatch="+", color='#ffffff', width=barWidth, edgecolor='black', label='Working set == gddr size * 110%')   
ax.bar(r2, fk_c_g2, hatch="|", color='#ffffff', width=barWidth, edgecolor='black', label='Working set == gddr size * 125%')            
ax.bar(r3, fk_c_f1, hatch="\\", color='#ffffff', width=barWidth, edgecolor='black', label='Working set == gddr size * 110% and pre-eviction at 95%')
ax.bar(r4, fk_c_f2, hatch="O", color='#ffffff', width=barWidth, edgecolor='black', label='Working set == gddr size * 110% and pre-eviction at 90%')


ax2.bar(r1, fk_c_g1, hatch="+", color='#ffffff', width=barWidth, edgecolor='black', label='Working set == gddr size * 110%')   
ax2.bar(r2, fk_c_g2, hatch="|", color='#ffffff', width=barWidth, edgecolor='black', label='Working set == gddr size * 125%')            
ax2.bar(r3, fk_c_f1, hatch="\\", color='#ffffff', width=barWidth, edgecolor='black', label='Working set == gddr size * 110% and pre-eviction at 95%')
ax2.bar(r4, fk_c_f2, hatch="O", color='#ffffff', width=barWidth, edgecolor='black', label='Working set == gddr size * 110% and pre-eviction at 90%')


ax.set_xticklabels([])

ax2.set_xticks([r + 0.4 + barWidth for r in range(len(fk_c_g0))])
ax2.set_xticklabels(benchmarks)


d = .015  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((-d, +d), (-0.1-d, 0.1+d), **kwargs)        # top-left diagonal
ax.plot((1 - d, 1 + d), (-0.1-d, 0.1+d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal


ax.yaxis.grid(b=True, which='major', color='grey', linestyle='-')
ax2.yaxis.grid(b=True, which='major', color='grey', linestyle='-')


ax2.set_xlim([0,7.2])
ax2.set_ylim([0,14000])


ax.set_xlim([0,7.2])
ax.set_ylim([38000,40000])
ax.yaxis.set_ticks([38000,40000])

# Set visible
ax.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)

ax2.set_ylabel('Number of 4KB Page Transfers')
ax2.yaxis.set_label_coords(-0.1,0.62)

ax.xaxis.set_ticks_position('none')
ax2.xaxis.set_ticks_position('none')

# Create legend & Show graphic
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 3.8), prop={'size': 12})

plt.savefig('./oversubscription/oversub_4k.png', dpi=300, bbox_inches="tight")


