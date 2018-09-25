# libraries
import re
import numpy as np
import matplotlib.pyplot as plt

#######################
# data extraction section
#######################

# declaration of variables
parent_folder = '../output_logs'
experiment_folder = 'PageSize'
sub_folders = ['4K', '2M']
benchmarks = ['bfs', 'backprop', 'hotspot', 'needle', 'pathfinder', 'stencil', 'srad']

kernel_4k = []
kernel_2m = []

for b in benchmarks:
	for sf in sub_folders:
		file_name = './' + parent_folder + '/' + experiment_folder + '/' + sf + '/' + b + '.log'

		with open(file_name, 'r') as b_file:
			file_content = b_file.read()

			
			line = re.findall(r"Tot_kernel_exec_time_and_fault_time.*", file_content, flags=re.MULTILINE)[0]
			k = float(line[line.find(', ')+2:line.rfind('(us)')])

			if sf == '4K':
				kernel_4k.append(k)
						
			elif sf == '2M':
				kernel_2m.append(k)
				

for i in range(len(kernel_2m)):
	kernel_2m[i] /= kernel_4k[i]
	kernel_4k[i] /= kernel_4k[i]

#######################
# plotting section
#######################

# set width of bar
barWidth = 0.2
 
plt.figure(1)

plt.figure(figsize=(7,3))

# set font
font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 12}

plt.rc('font', **font)

r1 = np.arange(len(kernel_4k), dtype=float)

for i in range(len(kernel_4k)):
	r1[i] = r1[i] + 0.4

r2 = [x + barWidth for x in r1]

# Plot the bar chart
plt.bar(r1, kernel_4k, hatch="-", color='#ffffff', width=barWidth, edgecolor='black', label='Page size 4KB and hardware prefetcher enabled')
plt.bar(r2, kernel_2m, hatch="/", color='#ffffff', width=barWidth, edgecolor='black', label='Page size 2MB and hardware prefetcher disabled')                     


# set ticks
plt.xticks([r + 0.4 + barWidth for r in range(len(kernel_4k))], benchmarks)


ax = plt.gca()
ax.yaxis.grid(b=True, which='major', color='grey', linestyle='-')
ax.axes.set_ylim([0.4,1.1])
plt.minorticks_on()

plt.ylabel('Kernel Execution Time\nNormalized to Page Size 4KB')

ax.xaxis.set_ticks_position('none') 

# Create legend & Show graphic
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), prop={'size': 12})

plt.tight_layout()

plt.savefig('./page_size/page_size.png', dpi=300, bbox_inches="tight")

