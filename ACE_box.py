""" Python module for ACE sonar data analysis

"""
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skimage
import skimage.exposure
from skimage.filters import gaussian, median

#########################################################
## Data extraction

def extract_data(filename,max_depth=100):
	df = pd.read_csv(filename, delimiter=',', skipinitialspace=True)
	info_df = df.iloc[:,:13]
	data= np.array(df.iloc[:,13:]).transpose()
	print('------------------------------')
	print('Data matrix size:',data.shape)
	depth_data = compute_depth_data(info_df)
	speed_vector,speed_averaged_vector = extract_speeds(info_df)
	info_df['speed'] = speed_vector
	info_df['speed_averaged'] = speed_averaged_vector
	data_trunc = cut_echogram(data,max_depth,depth_data)
	print('-----------------------------')
	return info_df,data_trunc,depth_data


def cut_echogram(data,depth,depth_data):
	""" Reduce the echogram to the values above 'depth'
		'depth' is the maximal depth in meters
	"""
	cutoff = depth_to_sample(depth,depth_data)
	print('Echogram truncated to the first {} meters ({} pixels).'.format(depth,cutoff))
	return data[:cutoff,:]



#####################################################
## Processing speed
def get_date(ping_date,ping_time,ping_milliseconds):
	microseconds = str(ping_milliseconds)+'000'
	if len(microseconds)==5:
		microseconds = '0' + microseconds
	if len(microseconds)==4:
		microseconds = '00'+ microseconds

	date_str = ping_date + ' ' + ping_time + ' ' + microseconds
	format_str = '%Y-%m-%d %H:%M:%S %f'
	return datetime.strptime(date_str,format_str)


def compute_speed(distance,delta_time,units='km/h'):
	#print(delta_time.total_seconds())
	if units=='km/h':
		return distance/1000/delta_time.total_seconds()*3600
	elif units=='m/s':
		return distance/delta_time.total_seconds()
	else:
		raise ValueError('Bad units. Units allowed: "km/h" or "m/s".')


def speed_at_sample(sample,df):
	miles_to_meter = 1852
	if sample == 0: # the speed will be the one of the next sample
		sample = 1
	distance = (df['Distance_gps'][sample] - df['Distance_gps'][sample-1])*miles_to_meter
	previous_sample_date = get_date(df['Ping_date'][sample-1],df['Ping_time'][sample-1],df['Ping_milliseconds'][sample-1])
	sample_date = get_date(df['Ping_date'][sample],df['Ping_time'][sample],df['Ping_milliseconds'][sample])
	#print(sample_date,next_sample_date)
	return compute_speed(distance,sample_date-previous_sample_date)


def extract_speeds(df):
	speeds = [speed_at_sample(s,df) for s in range(len(df['Ping_index']))]
	from scipy.signal import butter,filtfilt
	order,cutoff=5,0.1
	b, a = butter(order, cutoff, 'low')
	speeds_averaged = filtfilt(b, a, speeds)
	return speeds,speeds_averaged

def plot_speeds(df):
	speeds = df['speeds']
	speeds_filt = df['speeds_averaged']
	plt.figure()
	plt.plot(speeds,label='Raw speed')
	plt.plot(speeds_filt,label='Averaged speed')
	plt.title('Speed of the boat')
	plt.xlabel('Ping index')
	plt.ylabel('Speed (km/h)')
	plt.legend()
	plt.grid()
	plt.show()


#####################
## Depth processing

def depth_variation(df):
	variations = np.std(df['Depth_start'])+np.std(df['Depth_stop'])+np.std(df['Sample_count'])
	if variations > 1:
		print('Warning: there was a change in the depth per pixel!', variations)
		return True
	return False

def get_depth_per_pixel(df):
	if not depth_variation(df):
		print('Start depth (in meters):',df['Depth_start'][0])
		print('Stop depth (in meters):',df['Depth_stop'][0])
		print('Nb of pixels along depth axis:',df['Sample_count'][0])
		pixel_depth =  (df['Range_stop'][0] - df['Range_start'][0])/df['Sample_count'][0]
		print('Depth per pixel (in meters):',pixel_depth)
		return pixel_depth
	else:
		raise ValueError('The calibration of the depth has changed during the recording. Computation stopped.')


def compute_depth_data(df):
	depth_data = {}
	transducer_depth = 8.4 # transducer under the ship, below the see level	
	depth_data['depth_start'] = transducer_depth + df['Range_start'][0]
	depth_data['depth_per_pixel'] = get_depth_per_pixel(df)
	depth_data['depth_stop'] = df['Depth_stop'][0]
	return depth_data

def get_sample_depth(sample,depth_data):
	""" Depth in meters of the sample.
		From the formula given here : 
		http://support.echoview.com/WebHelp/Reference/Algorithms/Calculating_sample_range_and_depth_from_exported_data.htm
	"""
	return depth_data['depth_start'] + depth_data['depth_per_pixel'] * (sample + 0.5)

def depth_to_sample(depth,depth_data):
	""" Convert depth to sample index
	"""
	return int((depth - depth_data['depth_start']) / depth_data['depth_per_pixel'] - 0.5)



#########################################
## Image processing

def fix_contrast(image):
	# Contrast stretching
	p2, p98 = np.percentile(image, (2, 98))
	image_rescale = skimage.exposure.rescale_intensity(image, in_range=(p2, p98))
	return image_rescale

def binary_impulse(Sv, threshold=10):
	'''
	:param Sv: gridded Sv values (dB re 1m^-1)
	:type  Sv: numpy.array
	:param threshold: threshold-value (dB re 1m^-1)
	:type  threshold: float
	return:
	:param image: denoised image   
	'''
	mask = Sv.copy()
	samples,pings = Sv.shape
	nb_noisy = 0
	for sample in range(1, samples-1):
		for ping in range(0, pings):
			a = Sv[sample-1, ping]
			b = Sv[sample, ping]
			c = Sv[sample+1, ping]
			if (b - a > threshold) & (b - c > threshold):
				mask[sample, ping] = (a+c)/2
				nb_noisy +=1
	print('Number of noisy pixels removed: ',nb_noisy)
	return mask


def fast_binary_impulse(image):
	''' remove vertical lines in the echogram 
	'''
	mask = image.copy()
	rshift = np.roll(image,1,axis=1)
	lshift = np.roll(image,-1,axis=1)
	grad = image - rshift + image - lshift
	mask[grad>np.max(grad)/10]=1
	mask[grad<=np.max(grad)/10]=0
	#print('Gradients',np.max(grad),np.min(grad))
	#mask[grad>10]=1
	#mask[grad<=10]=0
	
	nb_noisy = np.sum(mask)
	data_d = image.copy()
	data_d[mask==1] = (rshift[mask==1] + lshift[mask==1]) / 2
	print('Number of noisy pixels removed: ',nb_noisy)
	return data_d

def remove_vertical_lines(image):
	#databi = binary_impulse(image.transpose(), threshold=np.max(image))
	#return databi.transpose()
	data_clean = fast_binary_impulse(image)
	return data_clean

def substract_meanovertime(image):
	""" Substract the mean over time.
	"""
	return image - np.mean(image,1,keepdims=True)

def substract_lin_meanovertime(echogram):
	lin_echogram = 10**(echogram/20)
	data_lin = lin_echogram - np.mean(lin_echogram,1,keepdims=True)
	echogram_f = 20*np.log10(data_lin-np.min(data_lin)+0.00001)
	return echogram_f

def gaussian_filter(image):
	sigma = [15,2]
	gauss_denoised = gaussian(image,sigma)
	return gauss_denoised

def remove_background_noise(data,depth_data):
	alpha = 0.0394177
	ref_depth = depth_data['depth_stop'] # 100m is 537 pixels
	data_trunc_ref = data
	Sv_ref = data_trunc_ref[-1,:]
	offset = np.mean(Sv_ref)- 20* np.log10(ref_depth)-2*alpha*ref_depth
	depth_values = np.linspace(depth_data['depth_start'],ref_depth,len(data_trunc_ref[:,0]))
	bg_noise = offset + 20 * np.log10(depth_values) + 2 * alpha * depth_values
	
	lin_echogram = 10**(data/10)
	bg_noise_matrix = np.tile(bg_noise,(lin_echogram.shape[1],1)).transpose()
	lin_bg_noise = 10**(bg_noise_matrix/10)
	#data_lin = lin_echogram - np.mean(lin_echogram,1,keepdims=True)
	data_lin = lin_echogram - lin_bg_noise
	echogram_f = 10*np.log10(data_lin - np.min(data_lin) + 0.00000000001)
	return echogram_f

###optional##
def denoisewavelet(image):
	from skimage.restoration import denoise_wavelet
	return denoise_wavelet(image, sigma=0.3)
###

def filter_data(data):
	data_rescale = fix_contrast(data)
	data2 = remove_vertical_lines(data_rescale)
	#data2 = median(data_rescale)
	data2 = substract_lin_meanovertime(data2)
	data2 = substract_meanovertime(data2)
	gauss_denoised = gaussian_filter(data2)
	return gauss_denoised

def show_echogram(data,depth_data):
	plt.figure()
	plt.imshow(data,aspect='auto')
	ticks = np.arange(0,data.shape[0],100)
	ticks_labels = [int(get_sample_depth(t,depth_data)) for t in ticks]
	plt.yticks(ticks,ticks_labels)
	plt.xlabel('Ping index')
	plt.ylabel('Depth (m)')
	plt.title('Echogram')
	plt.show()


#####################################################################
## Krill detection

def krill_function(image,threshold=0.5):
	""" Return a binary function 
		where a positive value means the presence of krill in the ping
		The second output value is a control value.
	"""
	energy = np.sqrt(np.sum(image**2,0))
	energy_fluctuation = np.std(energy)
	if energy_fluctuation == 0:
		raise ValueError('The echogram has constant values! Image size:',image.shape)
	normalized_energy = (energy-np.mean(energy))/energy_fluctuation
	binary_signal = normalized_energy.copy()
	binary_signal[binary_signal<threshold] = 0
	binary_signal[binary_signal>threshold] = 100
	return binary_signal,energy_fluctuation

def extract_krillchunks(binary_signal,data):
	""" Return a list of numpy array
	Each array corresponds to a krill swarm
	"""
	minimal_swarm_length = 2
	krill_chunks = []
	data_len = len(binary_signal)
	for idx in range(data_len):
		if binary_signal[idx] >0:
			if idx==0 or binary_signal[idx-1] == 0:
				# start of krill detection
				krill_start = idx
			if idx == data_len-1 or binary_signal[idx+1] == 0:
				# end of krill detection
				# Check if length is too small (artefact)
				if idx - krill_start >= minimal_swarm_length:
					krill_dic_list = process_chunk(data,krill_start,idx)
					# store the krill layers in list
					krill_chunks = krill_chunks + krill_dic_list
	return krill_chunks

def process_chunk(data,krill_start,krill_stop):
	krill_dic = {}
	krill_dic['Ping_start_index'] = krill_start 
	krill_dic['Ping_end_index'] = krill_stop
	krill_dic['data'] = data[:,krill_start:krill_stop+1]
	# Check if several swarms in the data
	#data_t = krill_dic['data'].transpose()
	#binary_signal,energy = krill_function(data_t)
	swarm_list = separate_verticalswarms(krill_dic)
	return swarm_list

def separate_verticalswarms(krill_dic):
	data = krill_dic['data']
	data_t = data.transpose()
	binary_signal,energy = krill_function(data_t)
	min_echo = np.min(data_t)
	#print('min',min_echo)
	data_list = []
	data_len = len(binary_signal)
	isolated_swarm_list = []
	for idx in range(data_len):
		if binary_signal[idx] >0:
			if idx==0 or binary_signal[idx-1] == 0:
				# start of krill swarm
				krill_start_depth = idx
			if idx == data_len-1 or binary_signal[idx+1] == 0:
				# end of krill swarm
				single_swarm_dic = {}
				single_swarm_dic['Ping_start_index'] = krill_dic['Ping_start_index'] 
				single_swarm_dic['Ping_end_index'] = krill_dic['Ping_end_index']
				single_swarm = np.ones(data.shape) * min_echo
				single_swarm[krill_start_depth:idx+1,:] = data[krill_start_depth:idx+1,:]
				#print(single_swarm.shape, krill_start_depth, idx+1)
				single_swarm_dic['data'] = single_swarm				
				#RESIZE SWARM LENGTH HERE
				single_swarm_dic = tighten_swarm_window(single_swarm_dic)
				#print(single_swarm_dic['data'].shape)
				# store krill layer in list
				isolated_swarm_list.append(single_swarm_dic)
	return isolated_swarm_list

def tighten_swarm_window(single_swarm_dic):
	data = single_swarm_dic['data']
	#binary_signal,energy = krill_function(data)
	energy = np.sqrt(np.sum(data**2,0))
	max_brightness = np.max(energy)
	data_len = len(energy)
	rate = 0.1 # rate threshold
	#reduce on the left
	for idx in range(data_len):
		if energy[idx] < max_brightness*rate:
			single_swarm_dic['Ping_start_index'] +=1
			single_swarm_dic['data'] = single_swarm_dic['data'][:,1:]
		else:
			break
	#reduce on the right
	for idx in range(data_len):
		if energy[data_len - 1 - idx] < max_brightness*rate:
			single_swarm_dic['Ping_end_index'] -=1
			single_swarm_dic['data'] = single_swarm_dic['data'][:,:-1]
		else:
			break
	return single_swarm_dic

###########################################################################
## Extraction of krill characteristics

# def swarm_depth(data):
# 	distribution = np.sum(data,axis=1)
# 	# Keep only the highest values above the noise
# 	distribution = distribution - np.min(distribution) # values can be negatives
# 	distribution_thres = distribution.copy()
# 	distribution_thres[distribution<0.5*np.max(distribution)] = 0
# 	density = distribution_thres/np.sum(distribution_thres)
# 	depth_coord = np.arange(0,len(density))
# 	mean_point = np.sum(density*depth_coord) # mean of the distribution
# 	sigma = np.sqrt((np.sum(density*(depth_coord-mean_point)**2))) # standard deviation
# 	height = 4 * sigma
# 	return mean_point,height

# def swarm_depth(data):
# 	data_t = data.transpose()
# 	binary_distribution,energy = krill_function(data_t)
# 	krill_chunks = extract_krillchunks(binary_distribution,data_t)
# 	height_list = []
# 	mean_point_list = []
# 	for chunk in krill_chunks:
# 		top = chunk['Ping_start_index']
# 		bottom = chunk['Ping_end_index']
# 		height_list.append( bottom - top)
# 		mean_point_list.append((bottom + top) / 2)
# 	if len(mean_point_list) > 1:
# 		print('More than one swarm! number:',mean_point_list,height_list)
# 	if len(mean_point_list) == 0:
# 		print('Nothing found in the chunk')
# 		return 0,0
# 	print('Swarm found')
# 	return mean_point_list[0],height_list[0]


def swarm_depth(data):
	data_t = data.transpose()
	binary_signal,energy = krill_function(data_t)
	data_len = len(binary_signal)
	# initialize
	height = 0
	mean_point = 0
	for idx in range(data_len):
		if binary_signal[idx] >0:
			if idx==0 or binary_signal[idx-1] == 0:
				# start of krill swarm
				swarm_top = idx
			if idx == data_len-1 or binary_signal[idx+1] == 0:
				# end of krill swarm
				swarm_bottom = idx
				height = swarm_bottom - swarm_top
				mean_point = (swarm_bottom + swarm_top) / 2
	#if height == 0:
	#	print('Swarm with height = 0 !')
	return mean_point, height



def timeandposition(idx,info_df):
	# Return latitude, longitude, date and time of the ping 'idx'

	latitude = info_df['Latitude'][idx] # info_df.iloc[idx,3]
	longitude = info_df['Longitude'][idx] #info_df.iloc[idx,4]
	date = info_df['Ping_date'][idx] # info_df.iloc[idx,1]
	time = info_df['Ping_time'][idx] # info_df.iloc[idx,2]
	return latitude,longitude,date,time


def swarm_infos(krill_chunk,info_df,depth_data,filename):
	miles_to_meter = 1852
	krill_info = {}
	krill_start = krill_chunk['Ping_start_index']
	krill_stop = krill_chunk['Ping_end_index']
	krill_info['ping_index_start'] = krill_start
	krill_info['ping_index_stop'] = krill_stop
	krill_info['filename'] = filename 
	krill_length_in_pings = krill_stop-krill_start
	krill_length = (info_df['Distance_gps'][krill_stop] -
	 info_df['Distance_gps'][krill_start]) * miles_to_meter
	krill_info['length'] = krill_length
	middle_location = int(krill_start + krill_length_in_pings / 2)
	latitude,longitude,date,time = timeandposition(middle_location, info_df)
	krill_info['latitude'] = latitude
	krill_info['longitude'] = longitude
	krill_info['date'] = date
	krill_info['time'] = time
	krill_info['boat speed'] = info_df['speed_averaged'][middle_location]
	mean_depth,height = swarm_depth(krill_chunk['data'])
	krill_info['depth_in_pixels'] = mean_depth
	krill_info['height_in_pixels'] = height
	krill_info['depth'] = get_sample_depth(mean_depth,depth_data)
	krill_info['height'] = height * depth_data['depth_per_pixel']
	return krill_info

def info_from_swarm_list(swarm_echo_list,echogram,info_df,depth_data,data_filename):
	swarm_info_list = []
	# Correct for the amplification along depth	
	echogram_rectif = remove_background_noise(echogram,depth_data)

	for swarm_echo in swarm_echo_list:
		info_dic = swarm_infos(swarm_echo,info_df,depth_data,data_filename)
		biomass,biomass_per_pixel = krill_brightness(info_dic,echogram_rectif)
		info_dic['biomass'] = biomass
		info_dic['biomass_per_pixel'] = biomass_per_pixel
		swarm_info_list.append(info_dic)
	return swarm_info_list
		


def krill_brightness(swarm_infos_dic,data):
	krill_start = swarm_infos_dic['ping_index_start']
	krill_stop = swarm_infos_dic['ping_index_stop']
	krill_top = int(swarm_infos_dic['depth_in_pixels']-swarm_infos_dic['height_in_pixels']/2)
	krill_bottom = int(swarm_infos_dic['depth_in_pixels']+swarm_infos_dic['height_in_pixels']/2)
	#print(krill_start,krill_stop,krill_top,krill_bottom)
	krill_window = data[krill_top:krill_bottom+1,krill_start:krill_stop]
	#return krill_window
	krill_b = np.sum(10**(krill_window/10))
	window_surface = (krill_stop - krill_start) * (krill_bottom - krill_top)
	if window_surface == 0:
		krill_b_per_pixel = 0
	else:
		krill_b_per_pixel = krill_b/window_surface
	return krill_b,krill_b_per_pixel