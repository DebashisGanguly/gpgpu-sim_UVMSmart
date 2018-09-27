# libraries
import re
import numpy as np
import matplotlib.pyplot as plt

#######################
# data extraction section
#######################

# declaration of variables
parent_folder = '../output_logs'
experiment_folder = 'RDMA'
sub_folders = ['NO_RDMA', 'RDMA', 'GDDR110_NO_RDMA', 'GDDR110_RDMA']
benchmarks = ['bfs', 'backprop', 'hotspot', 'needle', 'pathfinder', 'stencil', 'srad']

kernel_without_rdma = []
kernel_rdma = []
kernel_without_rdma_evict = []
kernel_rdma_evict = []

for b in benchmarks:
	for sf in sub_folders:
		file_name = './' + parent_folder + '/' + experiment_folder + '/' + sf + '/' + b + '.log'

		with open(file_name, 'r') as b_file:
			file_content = b_file.read()

			
			line = re.findall(r"Tot_kernel_exec_time_and_fault_time.*", file_content, flags=re.MULTILINE)[0]
			k = float(line[line.find(', ')+2:line.rfind('(us)')])

			if sf == 'GDDR110_NO_RDMA':
				kernel_without_rdma_evict.append(k)
						
			elif sf == 'GDDR110_RDMA':
				kernel_rdma_evict.append(k)

			elif sf == 'RDMA':
				kernel_rdma.append(k)

			elif sf == 'NO_RDMA':
				kernel_without_rdma.append(k)

			line = re.findall(r"Tot_kernel_exec_time_and_fault_time.*", file_content, flags=re.MULTILINE)[0]
			k = float(line[line.find(', ')+2:line.rfind('(us)')])
				


#######################
# plotting section
#######################

# set width of bar
barWidth = 0.2
 
plt.figure(1)

#plt.figure(figsize=(7,4))

# set font
font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 12}

plt.rc('font', **font)

plt.figure(figsize=(7,3))

r1 = np.arange(len(kernel_without_rdma), dtype=float)

for i in range(len(kernel_without_rdma)):
	r1[i] = r1[i] + 0.4

r2 = [x + barWidth for x in r1]

# Plot the bar chart
plt.bar(r1, kernel_without_rdma, hatch="-", color='#ffffff', width=barWidth, edgecolor='black', label='Working set == gddr size * 100% without RDMA')
plt.bar(r2, kernel_rdma, hatch="/", color='#ffffff', width=barWidth, edgecolor='black', label='Working set == gddr size * 100% with RDMA')                     

# set ticks
plt.xticks([r + 0.4 + barWidth for r in range(len(kernel_without_rdma))], benchmarks)


ax = plt.gca()
ax.yaxis.grid(b=True, which='major', color='grey', linestyle='-')
ax.yaxis.grid(b=True, which='minor', color='grey', linestyle='--') 
#ax.axes.set_ylim([0.4,1.1])
plt.minorticks_on()

plt.ylabel('Kernel Execution Time (us)')

ax.xaxis.set_ticks_position('none') 

# Create legend & Show graphic
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), prop={'size': 12})

plt.tight_layout()

plt.savefig('./rdma/rdma.png', dpi=300, bbox_inches="tight")


plt.cla()
plt.clf()

plt.figure(2)

plt.figure(figsize=(7,5))

# set font
font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 12}

plt.rc('font', **font)

plt.figure(figsize=(7,3))

r1 = np.arange(len(kernel_without_rdma), dtype=float)

for i in range(len(kernel_without_rdma)):
	r1[i] = r1[i] + 0.4

r2 = [x + barWidth for x in r1]

for i in range(len(kernel_rdma)):

	kernel_rdma_evict[i] /= kernel_without_rdma_evict[i]
	kernel_without_rdma_evict[i] /= kernel_without_rdma_evict[i]

# Plot the bar chart                   
plt.bar(r1, kernel_without_rdma_evict, hatch="\\", color='#ffffff', width=barWidth, edgecolor='black', label='Working set == gddr size * 110% without RDMA') 
plt.bar(r2, kernel_rdma_evict, hatch=".", color='#ffffff', width=barWidth, edgecolor='black', label='Working set == gddr size * 110% with RDMA') 

# set ticks
plt.xticks([r + 0.4 + barWidth for r in range(len(kernel_without_rdma))], benchmarks)


ax = plt.gca()
ax.yaxis.grid(b=True, which='major', color='grey', linestyle='-')
plt.minorticks_on()

plt.ylabel('Kernel Execution Time\nNormalized to RDMA Disabled')

ax.xaxis.set_ticks_position('none') 

# Create legend & Show graphic
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), prop={'size': 12})

plt.tight_layout()

plt.savefig('./rdma/rdma_eviction.png', dpi=300, bbox_inches="tight")

