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

thresh_with = []
thresh_without = []

#rdma_read = []
#migrate_read = []
#migrate_write = []

nvlink_memcpy = []
rdma_read = []
rdma_memcpy = []

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

			

			
			line = re.findall(r"Tot_memcpy_h2d_time.*", file_content, flags=re.MULTILINE)[0]
			k = float(line[line.find(', ')+2:line.rfind('(us)')])

			if sf == 'RDMA':
				rdma_memcpy.append(k)

			elif sf == 'NO_RDMA':
				nvlink_memcpy.append(k)

			line = re.findall(r"Tot_rdma_time.*", file_content, flags=re.MULTILINE)[0]
			k = float(line[line.find(', ')+2:line.rfind('(us)')])

			if sf == 'RDMA':
				rdma_read.append(k)

			

			line = re.findall(r"Page_tot_thresh.*", file_content, flags=re.MULTILINE)[0]
			k = float(line.split()[1])
			if sf == 'GDDR110_NO_RDMA':
				thresh_without.append(k)
						
			elif sf == 'GDDR110_RDMA':
				thresh_with.append(k)



#print thresh_without
#print thresh_with
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




plt.cla()
plt.clf()

plt.figure(3)

plt.figure(figsize=(7,5))

# set font
font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 12}

plt.rc('font', **font)

plt.figure(figsize=(7,3))

r1 = np.arange(len(thresh_without), dtype=float)

for i in range(len(thresh_without)):
	r1[i] = r1[i] + 0.4

r2 = [x + barWidth for x in r1]

print thresh_with
print thresh_without
for i in range(len(thresh_with)):

	if thresh_with[i] == 0:
		thresh_without[i] = 1
		thresh_with[i] = 1
	else:
		thresh_without[i] /= thresh_with[i]
		thresh_with[i] /= thresh_with[i]

# Plot the bar chart                   
plt.bar(r1, thresh_without, hatch="\\", color='#ffffff', width=barWidth, edgecolor='black', label='Working set == gddr size * 110% without RDMA') 
plt.bar(r2, thresh_with, hatch=".", color='#ffffff', width=barWidth, edgecolor='black', label='Working set == gddr size * 110% with RDMA') 

# set ticks
plt.xticks([r + 0.4 + barWidth for r in range(len(thresh_without))], benchmarks)


ax = plt.gca()
ax.yaxis.grid(b=True, which='major', color='grey', linestyle='-')
plt.minorticks_on()

plt.ylabel('Total # of Pages Thrashed\nNormalized to RDMA Enabled')

ax.xaxis.set_ticks_position('none') 

# Create legend & Show graphic
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), prop={'size': 12})

plt.tight_layout()

plt.savefig('./rdma/rdma_thrash.png', dpi=300, bbox_inches="tight")







plt.cla()
plt.clf()

plt.figure(3)

plt.figure(figsize=(7,5))

# set font
font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 12}

plt.rc('font', **font)

plt.figure(figsize=(7,3))

r1 = np.arange(len(nvlink_memcpy), dtype=float)

for i in range(len(nvlink_memcpy)):
	r1[i] = r1[i] + 0.4

r2 = [x + barWidth for x in r1]


# Plot the bar chart                   
bar_nv = plt.bar(r1, nvlink_memcpy, hatch="\\", color='#ffffff', width=barWidth, edgecolor='black', label='Host to device memory copy time without RDMA') 
bar_cpy = plt.bar(r2, rdma_memcpy, hatch=".", color='#ffffff', width=barWidth, edgecolor='black', label='Host to device memory copy time with RDMA') 
bar_rd = plt.bar(r2, rdma_read, hatch="+", color='#ffffff',bottom = rdma_memcpy, width=barWidth, edgecolor='black', label='RDMA memory read time') 

# set ticks
plt.xticks([r + 0.4 + barWidth for r in range(len(nvlink_memcpy))], benchmarks)


print rdma_memcpy[3]+rdma_read[3]
print rdma_memcpy[4]+rdma_read[4]

plt.text(r1[3]-0.35, 400, '3047')
plt.text(r1[4]-0.25, 400, '576')

ax = plt.gca()
ax.yaxis.grid(b=True, which='major', color='grey', linestyle='-')
ax.yaxis.grid(b=True, which='minor', color='grey', linestyle='--') 

ax.axes.set_ylim([0,500])

plt.minorticks_on()

plt.ylabel('Runtime (us)')

ax.xaxis.set_ticks_position('none') 

# Create legend & Show graphic
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.4), prop={'size': 12})

plt.tight_layout()

plt.savefig('./rdma/rdma_rw.png', dpi=300, bbox_inches="tight")
