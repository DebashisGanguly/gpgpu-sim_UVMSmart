# libraries
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

#######################
# data extraction section
#######################

# declaration of variables
parent_folder = '../output_logs'
experiment_folder = 'EvictionPolicy'
sub_folders = ['LRU', 'Random']
access_folders = 'Access_pattern'
benchmarks = ['bfs', 'backprop', 'hotspot', 'needle', 'pathfinder', 'stencil', 'srad']

k_s_l = []
k_s_r = []

for b in benchmarks:
	for sf in sub_folders:
		file_name = './' + parent_folder + '/' + experiment_folder + '/' + sf + '/' + b + '.log'
		
		with open(file_name, 'r') as b_file:
			file_content = b_file.read()

			line = re.findall(r"^Tot_kernel_exec_time.*", file_content, flags=re.MULTILINE)[0]
			k = float(line[line.find(', ')+2:line.rfind('(us)')])

			if sf == 'LRU':
				k_s_l.append(k)
			elif sf == 'Random':
				k_s_r.append(k)

#######################
# plotting section
#######################

# set width of bar
barWidth = 0.2
 
# Set position of bar on X axis
r1 = np.arange(len(k_s_l),dtype=float)

for i in range(len(k_s_l)):
	r1[i] = r1[i] + 0.4

r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

plt.figure(1)
font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 12}

plt.rc('font', **font)


plt.figure(figsize=(7,4))

plt.bar(r1, k_s_l, hatch=".", color='#ffffff', width=barWidth, edgecolor='black', label='LRU')            
plt.bar(r2, k_s_r, hatch="\\", color='#ffffff', width=barWidth, edgecolor='black', label='Random')


plt.xticks([r + 0.4 + barWidth for r in range(len(k_s_l))], benchmarks)

ax = plt.gca()
ax.yaxis.grid(b=True, which='major', color='grey', linestyle='-')
ax.yaxis.grid(b=True, which='minor', color='grey', linestyle='--') 
plt.minorticks_on()

plt.ylabel('Kernel Execution Time (us)')

ax.xaxis.set_ticks_position('none')

# Create legend & Show graphic
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), prop={'size': 12})

plt.savefig('./eviction_policy/eviction_policy.png',  dpi=300, bbox_inches="tight")



acc_time = []
acc_num = []


plt.cla()
plt.clf()
 
plt.figure(2)

plt.figure(figsize=(6,3))

plt.rc('font', **font)
file_name = './' + parent_folder + '/' + experiment_folder + '/' + access_folders + '/stencil' 

f = open(file_name)

pg_index_pfr = []
acc_counter_prf = []

for line in f:
	t = line.split();
	if t[0] != 'K:':
		ate = float(t[0])
		apn = int(t[1])
		#if ate < 1808737:
		if ate >= 1767224 and ate<=2072423:
			acc_time.append(ate/1481.0)
			acc_num.append(apn)


f.close()


plt.plot(acc_time, acc_num, 'ro', markersize=0.1)

ax = plt.gca()
ax.xaxis.grid(b=True, which='major', color='grey', linestyle='-')
ax.xaxis.grid(b=True, which='minor', color='grey', linestyle='--') 
ax.axes.set_xlim([1170,1420])
plt.minorticks_on()

plt.xlabel('Time (us)')
plt.ylabel('Page Number')
	
	#ax.xaxis.set_ticks_position('none')
	#ax.yaxis.set_major_locator(MaxNLocator(integer=True))
#plt.axis('auto')
plt.tight_layout()
plt.savefig('./eviction_policy/stencil_pattern.png',  dpi=300)




'''
plt.cla()
plt.clf()
 
plt.figure(3)

plt.figure(figsize=(7,4))

plt.rc('font', **font)
file_name = './' + parent_folder + '/' + experiment_folder + '/' + access_folders + '/pathfinder' 

f = open(file_name)

pg_index_pfr = []
acc_counter_prf = []

for line in f:
	t = line.split();
	if t[0] != 'K:':
		ate = float(t[0])
		apn = int(t[1])
		#if ate < 1808737:
		if apn > 550000 and ate < 2659434.0:
			acc_time.append(ate/1481.0)
			acc_num.append(apn)


f.close()


plt.plot(acc_time, acc_num, 'ro', markersize=0.1)

ax = plt.gca()
ax.xaxis.grid(b=True, which='major', color='grey', linestyle='-')
ax.xaxis.grid(b=True, which='minor', color='grey', linestyle='--') 
#ax.axes.set_xlim([100,1300])
#ax.axes.set_ylim([100,1300])
plt.minorticks_on()

plt.xlabel('Time (us)')
plt.ylabel('Page Number')
	
	#ax.xaxis.set_ticks_position('none')
	#ax.yaxis.set_major_locator(MaxNLocator(integer=True))
#plt.axis('auto')
plt.tight_layout()
plt.savefig('./eviction_policy/bfs_pattern.png',  dpi=300)
'''

'''
i = 2
for b in benchmarks:
	plt.cla()
	plt.clf()
 
	plt.figure(i)

	plt.rc('font', **font)
	file_name = './' + parent_folder + '/' + experiment_folder + '/' + access_folders + '/' + b

	f = open(file_name)

	pg_index_pfr = []
	acc_counter_prf = []

	for line in f:
		t = line.split();
		if t[0] != 'K:':
			ate = int(t[0])
			apn = int(t[1])

			#if b == 'bfs' and apn > 720000:
				#acc_time.append(ate)
				#acc_num.append(apn)
			if apn > 720000:
				acc_time.append(ate)
				acc_num.append(apn)


	f.close()


	plt.plot(acc_time, acc_num, 'ro', markersize=1)

	ax = plt.gca()
	#ax.yaxis.grid(b=True, which='major', color='black', linestyle='-')
	#ax.yaxis.grid(b=True, which='minor', color='black', linestyle='--') 
	#plt.minorticks_on()

	plt.xlabel('Time (cycle)')
	plt.ylabel('Page Number')
	
	#ax.xaxis.set_ticks_position('none')
	#ax.yaxis.set_major_locator(MaxNLocator(integer=True))
	plt.axis('auto')
	plt.savefig('./eviction_policy/'+b+'_pattern.png',  dpi=300)

'''
'''

i = 2
for b in benchmarks:
	plt.cla()
	plt.clf()
 
	plt.figure(i)
	i+=1
	plt.rc('font', **font)

	file_name = './' + parent_folder + '/' + experiment_folder + '/' + access_folders + '/' + b + '_access_pattern.txt'

	f = open(file_name)

	pg_index_pfr = []
	acc_counter_prf = []

	for line in f:
		t = line.split();
		if t[0] != 'Total':
			#a = int(t[0])
			#b = int(t[1])
		#if a < 1206722:
			pg_index_pfr.append(t[0])
			acc_counter_prf.append(t[1])

	f.close()

	for p,a in zip(pg_index_pfr,acc_counter_prf):
     		plt.plot([p, p], [0, a], color='black', linewidth=0.3)


	ax = plt.gca()
	ax.yaxis.grid(b=True, which='major', color='black', linestyle='-')
	ax.yaxis.grid(b=True, which='minor', color='black', linestyle='--') 
	plt.minorticks_on()

	plt.xlabel('Page Index')
	plt.ylabel('Access Frequency')

	ax.xaxis.set_ticks_position('none')


	plt.savefig('./oversubscription_percentage/' + b + '_access_pattern.png',  dpi=300)






plt.cla()
plt.clf()
 
plt.figure(i)

plt.rc('font', **font)
file_name = './' + parent_folder + '/' + experiment_folder + '/' + thresh_folders + '/' + 'bfs_thresh'

f = open(file_name)

pg_index_pfr = []
acc_counter_prf = []

for line in f:
	t = line.split();
	#if t[0] != 'Total':
		#a = int(t[0])
		#b = int(t[1])
		#if a < 1206722:
	pg_index_pfr.append(t[1])
	acc_counter_prf.append(t[3])

f.close()

for p,a in zip(pg_index_pfr,acc_counter_prf):
	plt.plot([p, p], [0, a], color='black', linewidth=0.8)


ax = plt.gca()
#ax.yaxis.grid(b=True, which='major', color='black', linestyle='-')
#ax.yaxis.grid(b=True, which='minor', color='black', linestyle='--') 
#plt.minorticks_on()

plt.xlabel('Page Index')
plt.ylabel('Thrash Frequency')

ax.xaxis.set_ticks_position('none')
ax.yaxis.set_major_locator(MaxNLocator(integer=True))

plt.savefig('./oversubscription_percentage/bfs_thresh_pattern.png',  dpi=300)
'''
