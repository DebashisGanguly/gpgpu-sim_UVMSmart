# libraries
import re
import numpy as np
import matplotlib.pyplot as plt

#######################
# data extraction section
#######################

# declaration of variables
parent_folder = '../output_logs'
experiment_folder = 'UVM_Basic'
sub_folders = ['Unmanaged', 'Managed', 'UserPrefetch']
benchmarks = ['bfs', 'backprop', 'hotspot', 'needle', 'pathfinder', 'stencil', 'srad']

k_s_u = []
k_s_m = []
k_s_p = []
m_s_u = []
m_s_m = []
m_s_p = []

for b in benchmarks:
	for sf in sub_folders:
		file_name = './' + parent_folder + '/' + experiment_folder + '/' + sf + '/' + b + '.log'

		with open(file_name, 'r') as b_file:
			file_content = b_file.read()

			line = re.findall(r"^Tot_memcpy_h2d_time.*", file_content, flags=re.MULTILINE)[0]
			h2d = float(line[line.find(', ')+2:line.rfind('(us)')])
			
			line = re.findall(r"^Tot_kernel_exec_time_and_fault_time.*", file_content, flags=re.MULTILINE)[0]
			k = float(line[line.find(', ')+2:line.rfind('(us)')])

			if sf == 'Unmanaged':
				k_s_u.append(k)
				m_s_u.append(h2d)
			elif sf == 'Managed':
				k_s_m.append(k)
				m_s_m.append(h2d)
			else:
				k_s_p.append(k)
				m_s_p.append(h2d)

u_l = [50,50,50,50,50,50,50]
m_l = [150,150,150,150,150,150,150]
p_o = [150,150,150,150,150,150,150]

#######################
# plotting section
#######################
 
# set width of bar
barWidth = 0.2
 
# Set position of bar on X axis
r1 = np.arange(len(k_s_u),dtype=float)

for i in range(len(k_s_u)):
	r1[i] = r1[i] + 0.2


r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 12}

plt.figure(1)
plt.rc('font', **font)

plt.figure(figsize=(7,4))

# Plot the bar chart
plt.bar(r1, k_s_u, hatch=".", color='#ffffff', width=barWidth,   edgecolor='black', label='cudaMalloc+cudaMemcpy')    
plt.bar(r2, k_s_m, hatch="\\", color='#ffffff', width=barWidth, edgecolor='black', label='cudaMallocManaged')
plt.bar(r3, k_s_p, hatch="|", color='#ffffff', width=barWidth, edgecolor='black', label='cudaMallocManaged+cudaMemPrefetchAsync')

# Set ticks
plt.xticks([r + 0.3 + barWidth for r in range(len(k_s_u))], benchmarks)

ax = plt.gca()
ax.yaxis.grid(b=True, which='major', color='grey', linestyle='-')
ax.yaxis.grid(b=True, which='minor', color='grey', linestyle='--') 
plt.minorticks_on()

ax.xaxis.set_ticks_position('none') 

plt.ylabel('Kernel Execution Time (us)')

# Create legend & Show graphic	
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),prop={'size': 12})

plt.tight_layout()

plt.savefig('./uvm_basic/uvm_basic_kernel.png', dpi=300, bbox_inches="tight")
