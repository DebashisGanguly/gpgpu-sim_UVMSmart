# libraries
import re
import numpy as np
import matplotlib.pyplot as plt

#######################
# data extraction section
#######################

# declaration of variables
parent_folder = '../output_logs'
experiment_folder = 'PCIE_Version'
sub_folders = ['3', '4', '5']
benchmarks = ['bfs', 'backprop', 'hotspot', 'needle', 'pathfinder', 'stencil', 'srad']

k_s_m3 = []
k_s_m4 = []
k_s_m5 = []

bd_3 = []
bd_4 = []
bd_5 = []

for b in benchmarks:
	for sf in sub_folders:
		file_name = './' + parent_folder + '/' + experiment_folder + '/' + sf + '/' + b + '.log'
		
		with open(file_name, 'r') as b_file:
			file_content = b_file.read()

			line = re.findall(r"^Tot_kernel_exec_time_and_fault_time.*", file_content, flags=re.MULTILINE)[0]
			k = float(line[line.find(', ')+2:line.rfind('(us)')])

			line = re.findall(r"Pcie_read_utilization.*", file_content, flags=re.MULTILINE)[0]
			bd = float(line[line.find(': ')+2:len(line)]) * 12.0

			if sf == '3':
				k_s_m3.append(k)
				bd_3.append(bd*12.0)
			elif sf == '4':
				k_s_m4.append(k)
				bd_4.append(bd*24.0)
			else:
				k_s_m5.append(k)
				bd_5.append(bd*36.0)

#######################
# plotting section
#######################

# set width of bar

plt.figure(1)

barWidth = 0.2
 
# Set position of bar on X axis
r1 = np.arange(len(k_s_m3), dtype=float)

for i in range(len(k_s_m3)):
	r1[i] = r1[i] + 0.2


r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 12}


plt.rc('font', **font)

plt.figure(figsize=(7,4))

# Normalize the values
for i in range(len(k_s_m3)):
	k_s_m5[i] /= k_s_m3[i]
	k_s_m4[i] /= k_s_m3[i]
	k_s_m3[i] /= k_s_m3[i]


plt.bar(r1, k_s_m3, hatch=".", color='#ffffff', width=barWidth, edgecolor='black', label='PCI-e 3.0')            
plt.bar(r2, k_s_m4, hatch="\\", color='#ffffff', width=barWidth, edgecolor='black', label='PCI-e 4.0')
plt.bar(r3, k_s_m5, hatch="|", color='#ffffff', width=barWidth, edgecolor='black', label='PCI-e 5.0')


# Set ticks
plt.xticks([r + 0.3 + barWidth for r in range(len(k_s_m3))], benchmarks)

ax = plt.gca()
ax.yaxis.grid(b=True, which='major', color='grey', linestyle='-') 
plt.minorticks_on()

plt.ylabel('Kernel Execution Time \nNormalized to PCI-e 3.0')

ax.axes.set_ylim([0.6,1.1])
ax.xaxis.set_ticks_position('none') 


# Create legend & Show graphic
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), prop={'size': 12})

plt.savefig('./pcie_version/pcie_version.png', dpi=300, bbox_inches="tight")

