import os
import numpy as np

def cn():
	# for dirname in os.listdir('images'):
	new_dirs = np.arange(12)
	dirname = 'images'
	if os.path.isdir(dirname):
		for i, filename in enumerate(os.listdir(dirname)):
			if(i%10==0):
				new_dir = str(new_dirs[int(i/10)])
				if not os.path.exists(new_dir):
					os.makedirs(new_dir)
			os.rename(dirname + "/" + filename, new_dir + "/" + str(i) + ".jpg")

if __name__ == '__main__':
	cn()