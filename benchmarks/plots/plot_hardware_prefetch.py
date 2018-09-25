# libraries
import re
import numpy as np
import matplotlib.pyplot as plt

#######################
# data extraction section
#######################

# declaration of variables
parent_folder = '../output_logs'
experiment_folder = 'HardwarePrefetch'
sub_folders = ['With', 'Without']
benchmarks = ['bfs', 'backprop', 'hotspot', 'needle', 'pathfinder', 'stencil', 'srad']

kernel_with = []
kernel_without = []
memcpy_with = []
memcpy_without = []

bandwith_with = []
bandwith_without = []
page_fault_with = []
page_fault_without = []

for b in benchmarks:
	for sf in sub_folders:
		file_name = './' + parent_folder + '/' + experiment_folder + '/' + sf + '/' + b + '.log'

		with open(file_name, 'r') as b_file:
			file_content = b_file.read()

			line = re.findall(r"^Tot_memcpy_h2d_time.*", file_content, flags=re.MULTILINE)[0]
			h2d = float(line[line.find(', ')+2:line.rfind('(us)')])
			
			line = re.findall(r"Tot_kernel_exec_time_and_fault_time.*", file_content, flags=re.MULTILINE)[0]
			k = float(line[line.find(', ')+2:line.rfind('(us)')])

			line = re.findall(r"Pcie_read_utilization.*", file_content, flags=re.MULTILINE)[0]
			bd = float(line[line.find(': ')+2:len(line)]) * 12.0

			line = re.findall(r"Page_tot_fault.*", file_content, flags=re.MULTILINE)[0]
			p = int(line[line.find(': ')+2:line.rfind('Page_tot_pending:')])

			

			if sf == 'With':
				kernel_with.append(k)
				memcpy_with.append(h2d)
				bandwith_with.append(bd)
				page_fault_with.append(p)

			elif sf == 'Without':
				kernel_without.append(k)
				memcpy_without.append(h2d)
				bandwith_without.append(bd)
				page_fault_without.append(p)

#######################
# plotting section
#######################

# set width of bar
barWidth = 0.2
 
plt.figure(1)

plt.figure(figsize=(7,4))


# set font
font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 12}

plt.rc('font', **font)

r1 = np.arange(len(memcpy_with),dtype=float)

for i in range(len(memcpy_with)):
	r1[i] = r1[i] + 0.2

r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]

plt.bar(r1, memcpy_with, hatch="-", color='#ffffff', width=barWidth, edgecolor='black', label='Host to device memory copy time with hardware prefetch')

plt.bar(r2, kernel_with, hatch="+", color='#ffffff', width=barWidth, edgecolor='black', label='Kernel execution time with hardware prefetch')            

plt.bar(r3, memcpy_without, hatch="/", color='#ffffff', width=barWidth, edgecolor='black', label='Host to device memory copy time without hardware prefetch')

plt.bar(r4, kernel_without, hatch="\\", color='#ffffff', width=barWidth, edgecolor='black', label='Kernel execution time without hardware prefetch')


# set ticks
plt.xticks([r + 0.4 + barWidth for r in range(len(kernel_with))], benchmarks)


ax = plt.gca()
ax.yaxis.grid(b=True, which='major', color='grey', linestyle='-')

plt.ylabel('Runtime (Log Scale)')

ax.axes.set_xlim([0,7.2])


ax.set_yscale('log')


ax.xaxis.set_ticks_position('none') 

# Create legend & Show graphic
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.28),prop={'size': 12})

plt.tight_layout()

plt.savefig('./hw_prefetch/hardwareprefetch.png', dpi=300, bbox_inches="tight")



plt.cla()
plt.clf()
 
plt.figure(2)

plt.figure(figsize=(7,2.5))

font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 12}

plt.rc('font', **font)

r1 = np.arange(len(bandwith_with),dtype=float)
for i in range(len(bandwith_with)):
	r1[i] = r1[i] + 0.2
r2 = [x + barWidth for x in r1]

# Plot the bar chart
plt.bar(r1, bandwith_with, hatch=".", color='#ffffff', width=barWidth, edgecolor='black', label='With hardware prefetch')
plt.bar(r2, bandwith_without, hatch="/", color='#ffffff', width=barWidth, edgecolor='black', label='Without hardware prefetch') 

# Set ticks
plt.xticks([r + 0.2 + barWidth for r in range(len(bandwith_with))], benchmarks)

ax = plt.gca()
ax.yaxis.grid(b=True, which='major', color='grey', linestyle='-')
plt.minorticks_on()

plt.ylabel('Average PCI-e Bandwidth\n Host to Device (GB/s)')

ax.xaxis.set_ticks_position('none') 

# Create legend & Show graphic

ax.axes.set_ylim([0,9])
ax.yaxis.set_ticks([0, 3, 6, 9]) 
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.27),prop={'size': 12})

plt.savefig('./hw_prefetch/hardwareprefetch_bandwith.png',  dpi=300, bbox_inches="tight")
