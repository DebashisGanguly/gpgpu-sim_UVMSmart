#!/usr/bin/env
''' run.py -- An interactive experiment running tool
 
Usage:
	run.py [-a <all>] [-d <default>] [-p <page-size>] [-g <gddr-size>] [-e <eviction-policy>] [-f <free-page-buffer>] [-i <pcie>]  [-x <hardware>]

Options:
	-a  --all				running all experiments
	-d  --default   <default>		running unmanged & managed & managed with prefetch
	-x  --hardware	<hardware>		running with hardware prefetch & without hardware prefetch
 	-p  --page-szie <page-size>		running page size experiments (4kb, 2mb)
	-g  --gddr-size <gddr-size>		running gddr size experiments (90.9%, 80% of the working set)
	-e  --evction   <eviction-policy> 	running different eviction policy (lru, random)
	-f  --free 	<free-page-buffer>	running different free page buffer (5%, 10% of gddr)
	-i  --pcie	<pcie>			running different pcie standard (3.0, 4.0, 5.0)
	
'''
import os
os.system("make clean")
os.system("make -j10")

config_file = './configs/GeForceGTX1080Ti/gpgpusim.config'
list_of_benchmark = ["backprop","bfs","hotspot","nw","pathfinder","srad_v2","stencil"]
size_of_benchmark = [2.4719, 2.496, 3, 32, 38.5284, 2.9541, 4]

managed_dir = 'benchmarks/Managed/'
unmanaged_dir = 'benchmarks/Unmanaged/'
result_dir = 'benchmarks/output_logs/'

import re
import os


def reset_config():
	newText = ''
	with open(config_file) as f:
		for line in f:
			if(line[0] != '#'):
				line = re.sub('-page_size.*', '-page_size 4KB',line)
				line = re.sub('-page_table_walk_latency.*', '-page_table_walk_latency 100',line)
				line = re.sub('-gddr_size.*', '-gddr_size 1GB',line)
				line = re.sub('-eviction_policy.*', '-eviction_policy lru',line)
				line = re.sub('-percentage_of_free_page_buffer.*', '-percentage_of_free_page_buffer 0',line)
				line = re.sub('-pcie_bandwith.*', '-pcie_bandwith 16.0GB/s',line)
				line = re.sub('-hardware_prefetch.*', '-hardware_prefetch 1',line)
			newText = newText + line


	with open(config_file, "w") as f:
		f.write(newText)


def pop_config():
	os.system("cd benchmarks/;./setup_config.sh --cleanup;./setup_config.sh GeForceGTX1080Ti")

def set_config(option, value):
	newText = ''
	with open(config_file) as f:
		for line in f:
			if(line[0] != '#'):
				line = re.sub(option+'.*', option+' '+value,line)
			newText = newText + line


	with open(config_file, "w") as f:
		f.write(newText)

def run_benchmark(name, type):
	dir = ''
	if(type == 'managed' or type == 'prefetch'):
		dir = managed_dir
	elif(type == 'unmanaged'):
		dir = unmanaged_dir
	pref = ''
	if(type == 'prefetch'):
		pref = 'prefetch'
	command = "cd " + dir + name + "; make clean; make" + pref + "; ./run > " + name + ".log"
	os.system(command)
	print command

def save_file(file_dir, bench_name, new_exp, new_name):
	os.system("mkdir "+ result_dir + new_exp)
	os.system("mkdir "+ result_dir + new_exp + "/" + new_name + "/")
	os.system("cp " + file_dir + bench_name +"/" + bench_name + ".log " + result_dir + new_exp + "/" + new_name + "/")
	os.system("cp " + file_dir + bench_name + "/Pcie_trace.txt " + result_dir + new_exp + "/" + new_name + "/"+ bench_name +"_pcie_trace.txt")
	os.system("cp " + file_dir + bench_name + "/Access_pattern_detail.txt " + result_dir + new_exp + "/" + new_name + "/"+ bench_name +"_access_pattern_detail.txt")
	os.system("cp " + file_dir + bench_name + "/Access_pattern.txt " + result_dir + new_exp + "/" + new_name+ "/" + bench_name +"_access_pattern.txt")
	os.system("cp " + file_dir + bench_name + "/access.txt " + result_dir + new_exp + "/" + new_name + "/"+ bench_name +"_access.txt")
	os.system("cp " + config_file + " " + result_dir + new_exp + "/" + new_name + "/sim.config")

def run_default():
	reset_config()

	for ben in list_of_benchmark:
		run_benchmark(ben, 'managed')
		save_file(managed_dir, ben,  'UVM_Basic', 'Managed')

	for ben in list_of_benchmark:
		run_benchmark(ben, 'unmanaged')
		save_file(unmanaged_dir, ben,  'UVM_Basic', 'Unmanaged')

	for ben in list_of_benchmark:
		run_benchmark(ben, 'prefetch')
		save_file(managed_dir, ben,  'UVM_Basic', 'UserPrefetch')	

def run_page_size():
	reset_config()

	set_config('-page_size', '4KB')
	for ben in list_of_benchmark:
		run_benchmark(ben, 'managed')
		save_file(managed_dir, ben,  'PageSize', '4K')

	set_config('-page_size', '2MB')
	set_config('-page_table_walk_latency', '66')
	for ben in list_of_benchmark:
		run_benchmark(ben, 'managed')
		save_file(managed_dir, ben,  'PageSize', '2M')
	

def run_hardware():
	reset_config()

	set_config('-hardware_prefetch', '1')
	for ben in list_of_benchmark:
		run_benchmark(ben, 'managed')
		save_file(managed_dir, ben,  'HardwarePrefetch', 'With')

	set_config('-hardware_prefetch', '0')
	for ben in list_of_benchmark:
		run_benchmark(ben, 'managed')
		save_file(managed_dir, ben,  'HardwarePrefetch', 'Without')

def run_gddr_size():
	reset_config()

	set_config('-eviction_policy', 'lru')

	i = 0
	while i < len(list_of_benchmark):
		run_benchmark(list_of_benchmark[i], 'managed')
		save_file(managed_dir, ben,  'OversubscriptionPercentage', '0')
		i += 1

	i = 0
	while i < len(list_of_benchmark):
		set_config('-gddr_size',str(size_of_benchmark[i]*10.0/11.0) +'MB')
		run_benchmark(list_of_benchmark[i], 'managed')
		save_file(managed_dir, ben,  'OversubscriptionPercentage', '110')
		i += 1

	i = 0
	while i < len(list_of_benchmark):
		set_config('-gddr_size',str(size_of_benchmark[i]*100.0/125.0) +'MB')
		run_benchmark(list_of_benchmark[i], 'managed')
		save_file(managed_dir, ben,  'OversubscriptionPercentage', '125')
		i += 1
	


def run_evcition():

	reset_config()

	set_config('-eviction_policy', 'lru')

	i = 0
	while i < len(list_of_benchmark):
		set_config('-gddr_size',str(size_of_benchmark[i]*10.0/11.0) +'MB')
		run_benchmark(list_of_benchmark[i], 'managed')
		save_file(managed_dir, ben,  'EvictionPolicy', 'LRU')
		i += 1

	set_config('-eviction_policy', 'random')

	i = 0
	while i < len(list_of_benchmark):
		set_config('-gddr_size',str(size_of_benchmark[i]*10.0/11.0) +'MB')
		run_benchmark(list_of_benchmark[i], 'managed')
		save_file(managed_dir, ben,  'EvictionPolicy', 'Random')
		i += 1


def run_free():
	
	reset_config()

	set_config('-eviction_policy', 'lru')

	i = 0
	while i < len(list_of_benchmark):
		set_config('-gddr_size',str(size_of_benchmark[i]*10.0/11.0) +'MB')
		run_benchmark(list_of_benchmark[i], 'managed')
		save_file(managed_dir, ben,  'evictionThresold', '0')
		i += 1

	i = 0
	while i < len(list_of_benchmark):
		set_config('-gddr_size',str(size_of_benchmark[i]*10.0/11.0) +'MB')
		set_config('-percentage_of_free_page_buffer', 0.05)
		run_benchmark(list_of_benchmark[i], 'managed')
		save_file(managed_dir, ben,  'Pre-evictionThresold', '5')
		i += 1

	i = 0
	while i < len(list_of_benchmark):
		set_config('-gddr_size',str(size_of_benchmark[i]*10.0/11.0) +'MB')
		set_config('-percentage_of_free_page_buffer', 0.1)
		run_benchmark(list_of_benchmark[i], 'managed')
		save_file(managed_dir, ben,  'Pre-evictionThresold', '10')
	


def run_pcie():
	reset_config()

	set_config('-pcie_bandwith', '16.0GB/s')
	for ben in list_of_benchmark:
		run_benchmark(ben, 'managed')
		save_file(managed_dir, ben,  'PCIE_Version', '3')

	set_config('-pcie_bandwith', '32.0GB/s')
	for ben in list_of_benchmark:
		run_benchmark(ben, 'managed')
		save_file(managed_dir, ben,  'PCIE_Version', '4')

	set_config('-pcie_bandwith', '64.0GB/s')
	for ben in list_of_benchmark:
		run_benchmark(ben, 'managed')
		save_file(managed_dir, ben,  'PCIE_Version', '5')

try:
	from docopt import docopt
	from os import system
	arguments = docopt(__doc__, version='run.py version 0.0.1')	

	os.system("make clean;make -j10")
	reset_config()
	pop_config()
	os.system("mkdir " + result_dir)

	if arguments['-a']:
		run_default()
		run_page_size()
		run_pcie()
		run_gddr_size()
		run_evcition()
		run_free()
		run_hardware()
	
	if arguments['-d']:
		run_default()

	if arguments['-p']:
		run_page_size()

	if arguments['-i']:
		run_pcie()

	if arguments['-g']:
		run_gddr_size()
	
	if arguments['-e']:
		run_evcition()

	if arguments['-f']:
		run_free()

	if arguments['-x']:
		run_hardware()

except KeyboardInterrupt:
    exit('Interrupt detected! exiting...')
