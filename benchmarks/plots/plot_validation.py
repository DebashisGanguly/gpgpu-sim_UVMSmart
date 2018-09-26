# libraries
import re
import numpy as np
import matplotlib.pyplot as plt
 
#######################
# data extraction section
#######################

# declaration of variables
parent_folder = '../output_logs'
experiment_folder = 'Validation'
sub_folders = ['CRC_Managed', 'Sim_Managed']
benchmarks = ['bfs', 'backprop', 'hotspot', 'needle', 'pathfinder', 'stencil', 'srad']

k_s_m = []
m_s_m = []

k_c_m = []
m_c_m = []

for b in benchmarks:
	for sf in sub_folders:
		file_name = './' + parent_folder + '/' + experiment_folder + '/' + sf + '/' + b + '.log'
		
		with open(file_name, 'r') as b_file:
			file_content = b_file.read()

			if sf == 'Sim_Managed':
				line = re.findall(r"^Tot_memcpy_h2d_time.*", file_content, flags=re.MULTILINE)[0]
				h2d = float(line[line.find(', ')+2:line.rfind('(us)')])

				m_s_m.append(h2d)

				line = re.findall(r"^Tot_kernel_exec_time_and_fault_time.*", file_content, flags=re.MULTILINE)[0]
				k = float(line[line.find(', ')+2:line.rfind('(us)')])

				k_s_m.append(k)
			else:
				line = re.findall(r"^.*Host To Device", file_content, flags=re.MULTILINE)[0]
				h2d_text = line.split()[5]
				if h2d_text[len(h2d_text) - 2:] == 'us':
					h2d = float(h2d_text[:len(h2d_text) - 2])
				elif h2d_text[len(h2d_text) - 2:] == 'ms':
					h2d = float(h2d_text[:len(h2d_text) - 2]) * 1000
				
				m_c_m.append(h2d)

				k = 0

				lines = re.findall(r"^.*\(.*\*.*.\)", file_content, flags=re.MULTILINE)

				for line in lines:
					if line.startswith(' GPU activities:'):
						line = line.replace(' GPU activities:', '', 1)
					k_text = line.split()[1]

					if k_text[len(k_text) - 2:] == 'us':
						k += float(k_text[:len(k_text) - 2])
					elif k_text[len(k_text) - 2:] == 'ms':
						k += float(k_text[:len(k_text) - 2]) * 1000
				
				k_c_m.append(k)

#######################
# plotting section
#######################

# set width of bar
barWidth = 0.2
 
# Set position of bar on X axis
r1 = np.arange(len(k_c_m),dtype=float)
for i in range(len(k_c_m)):
	r1[i] = r1[i] + 0.2
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]

# set font
font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 12}

plt.rc('font', **font)

plt.figure(figsize=(7,4))


# plot the bars
plt.bar(r1, k_c_m, hatch="+", color='#ffffff', width=barWidth, edgecolor='black', label='Kernel execution time on real machine')            
plt.bar(r2, k_s_m, hatch="\\", color='#ffffff', width=barWidth, edgecolor='black', label='Kernel execution time on simulator')
plt.bar(r3, m_c_m, hatch="-", color='#ffffff', width=barWidth, edgecolor='black', label='Host to device memory copy time on real machine')
plt.bar(r4, m_s_m, hatch="/", color='#ffffff', width=barWidth, edgecolor='black', label='Host to device memory copy time on simulator')




# set ticks
plt.xticks([r + 0.4 + barWidth for r in range(len(k_c_m))], benchmarks)

ax = plt.gca()
ax.yaxis.grid(b=True, which='major', color='grey', linestyle='-')
ax.yaxis.grid(b=True, which='minor', color='grey', linestyle='--') 
plt.minorticks_on()

ax.axes.set_xlim([0,7.25])


plt.ylabel('Runtime (us)')

ax.xaxis.set_ticks_position('none') 

# plot error
for i in range(len(k_c_m)):
	if k_c_m[i] >  k_s_m[i]:
		height = k_c_m[i]
	else:
		height = k_s_m[i]

	err =  abs(float(k_c_m[i]-k_s_m[i]))/float(k_c_m[i])

	ax.text(r1[i] + barWidth, height + 20,
                '%d%%' % int(err*100),
                ha='center', va='bottom')


for i in range(len(m_c_m)):
	if m_c_m[i] >  m_s_m[i]:
		height = m_c_m[i]
	else:
		height = m_s_m[i]
	
	err =  abs(float(m_c_m[i]-m_s_m[i]))/float(m_c_m[i])

	ax.text(r3[i] + barWidth+0.1, height + 20,
                '%d%%' % int(err*100),
                ha='center', va='bottom')


# Create legend & Show graphic
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.33),prop={'size': 12})

plt.savefig('./validation/validation.png',  dpi=300, bbox_inches="tight")
