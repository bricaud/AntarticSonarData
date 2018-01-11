""" Python module for ACE sonar data analysis

"""
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import skimage
import skimage.exposure

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

def extract_data(filename,max_depth=100):
	df = pd.read_csv(filename, delimiter=',', skipinitialspace=True)
	info_df = df.iloc[:,:13]
	data= np.array(df.iloc[:,13:]).transpose()
	print('Data matrix size:',data.shape)
	depth_data = compute_depth_data(df)
	data_trunc = cut_echogram(data,max_depth,depth_data)
	return info_df,data_trunc,depth_data

######################################

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

def cut_echogram(data,depth,depth_data):
	""" Reduce the echogram to the values above 'depth'
	"""
	cutoff = depth_to_sample(depth,depth_data)
	return data[:cutoff,:]


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
	:param mask: binary mask (0 - noise; 1 - signal)
	:type  mask: 2D numpy.array
	desc: generate threshold mask    
	'''
	mask = np.ones(Sv.shape).astype(int)
	samples,pings = Sv.shape
	for sample in range(1, samples-1):
		for ping in range(0, pings):
			a = Sv[sample-1, ping]
			b = Sv[sample, ping]
			c = Sv[sample+1, ping]
			if (b - a > threshold) & (b - c > threshold):
				mask[sample, ping] = 0
	return mask

def remove_vertical_lines(image):
	databi = binary_impulse(image.transpose(), threshold=np.max(image))
	print('Number of noisy pixels: ',databi.size-np.sum(databi))
	# Replace the noisy pixels by the minimal value of the image (may not be zero)
	return databi.transpose()*image+(1-databi.transpose())*np.min(image)

def substract_meanovertime(image):
	""" Substract the mean over time.
	"""
	return image - np.mean(image,1,keepdims=True)

def gaussian_filter(image):
	from skimage.filters import gaussian
	gauss_denoised = gaussian(image,5)
	return gauss_denoised

###optional##
def denoisewavelet(image):
	from skimage.restoration import denoise_wavelet
	return denoise_wavelet(image, sigma=0.3)
###

def filter_data(data):
	data_rescale = fix_contrast(data)
	data2 = remove_vertical_lines(data_rescale)
	data2 = substract_meanovertime(data2)
	gauss_denoised = gaussian_filter(data2)
	return gauss_denoised

######################## Krill detection #############################
def krill_function(image):
	energy = np.sqrt(np.sum(image**2,0))
	energy_fluctuation = np.std(energy)
	normalized_energy = (energy-np.mean(energy))/energy_fluctuation
	binary_signal = normalized_energy.copy()
	threshold = 0.5
	binary_signal[binary_signal<threshold] = 0
	binary_signal[binary_signal>threshold] = 100
	return binary_signal,energy_fluctuation

def extract_krillchunks(binary_signal,data):
	krill_chunks = []
	krill_dic = {}
	data_len = len(binary_signal)
	for idx in range(data_len):
		if binary_signal[idx] >0:
			if idx==0 or binary_signal[idx-1] == 0:
				# start of krill detection
				krill_start = idx
			if idx == data_len-1 or binary_signal[idx+1] == 0:
				# end of krill detection
				krill_dic['Ping_start_index'] = krill_start 
				krill_dic['Ping_end_index'] = idx
				krill_dic['data'] = data[:,krill_start:idx+1]
				# store krill layer in list
				krill_chunks.append(krill_dic)
				krill_dic = {}
	return krill_chunks

def record_starttimeandposition(krill_dic,idx,info_df):
	# record latitude and longitude
	krill_dic['latitude_start'] = info_df.iloc[idx,3]
	krill_dic['longitude_start'] = info_df.iloc[idx,4]
	krill_dic['date_start'] = info_df.iloc[idx,1]
	krill_dic['time_start'] = info_df.iloc[idx,2]
	return krill_dic

def record_stoptimeandposition(krill_dic,idx,info_df):
	# record latitude and longitude
	krill_dic['latitude_stop'] = df.iloc[idx,3]
	krill_dic['longitude_stop'] = df.iloc[idx,4]
	krill_dic['date_stop'] = df.iloc[idx,1]
	krill_dic['time_stop'] = df.iloc[idx,2]
	return krill_dic
