# Load libraries
import xml.etree.ElementTree as ET
import glob
import urllib.request, json
import time
from socket import timeout

def extract_ewns(file_name, return_name=True):
	# file_name:
	#        kml file name
	# return_name:
	#        controls whether return small image file name
	# return east longitude, west longitude,
	#        north latitude, south latitude
	#        (optional) corresponding image name
	tree = ET.parse(file_name)
	root = tree.getroot()
	name = root[0].find('name').text
	root = root[0].find('LatLonBox')
	e = float(root.find('east').text)
	w = float(root.find('west').text)
	n = float(root.find('north').text)
	s = float(root.find('south').text)
	if return_name:
		return e,w,n,s,name
	return e,w,n,s

def calculate_center(e,w,n,s):
	# return center longitude, center latitude
	return (e+w)/2.0, (n+s)/2.0

def request_fips(lon,lat):
	# check fips by lat & lon online
	url = 'https://geo.fcc.gov/api/census/area?lat=' + \
	      str(lat) + '&lon=' + str(lon) + '&format=json'
	with urllib.request.urlopen(url) as req:
		data = json.loads(req.read().decode())
		if len(data['results']) > 0:
			fips = data['results'][0]['block_fips']
		else:
			fips = '0'
	return fips

FOLDER_PATH = 'City'
OUTPUT_CSV = 'City_fips_index.csv'
OUTPUT_LOG = 'City_fips.log'
# set SLEEP_TIME to 0 if you don't want this
SLEEP_TIME = 0

all_kmls = glob.glob(FOLDER_PATH+'/*.kml')
with open(OUTPUT_CSV,'w') as fout:
	with open(OUTPUT_LOG,'w') as flog:
		for kml in all_kmls:
			e,w,n,s,name = extract_ewns(kml)
			print('File: '+kml)
			lon,lat = calculate_center(e,w,n,s)
			if SLEEP_TIME != 0:
				time.sleep(SLEEP_TIME)
			try:
				fips = request_fips(lon,lat)
				fout.write(name + ',' + fips+'\n')
			except (HTTPError, URLError) as error:
				flog.write('Data of %s not retrieved because %s\nURL: %s\n',
						   name, error, url)
			except timeout:
				flog.write('socket timed out - URL %s\n', url)
			else:
				print(name + ': Success!')
