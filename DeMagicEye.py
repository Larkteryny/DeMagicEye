#from SimpleCV import Image, Display, Color
from multiprocessing import Process, Queue
import numpy as np
import cv2
import copy
import sys


# caclulate the value of a row
# using the integral image
def idxToSum(x1, x2, y, integral):
	# assume x2 > x1
	p3 = integral[y + 1, x2 + 1]
	p2 = integral[y + 1, x1 - 1]
	p1 = integral[y, x2 + 1]
	p0 = integral[y, x1 - 1]
	val = p3 - p2 - p1 + p0
	return val


def integralWindow(x1, y1, x2, y2, integral):
	p3 = integral[y2, x2]
	p2 = integral[y2, x1]
	p1 = integral[y1, x2]
	p0 = integral[y1, x1]
	val = p3 - p2 - p1 + p0
	return val


def findOptimalWindow(img, integral, minSplit=4, maxSplit=16):
	height = img.shape[0]
	width = img.shape[1]

	maxWin = width // minSplit
	minWin = width // maxSplit
	vals = []
	for i in range(minWin, maxWin):
		left = integralWindow(0, 0, i, height, integral)
		right = integralWindow(0, i, 2 * i, height, integral)
		vals.append(np.abs(left - right) / (i * height))
	return np.where(np.array(vals) == np.min(vals))[0][0]


# get depth map
def doMagicEye(img, integral, window, samplesz, queue, width, height):
	height = img.shape[0]
	width = img.shape[1]
	print(width, height, window, samplesz, queue)

	dmap = np.zeros([width - window, height], dtype='int32')
	# really wish I could get rid of this iteration
	for vidx in range(1, height - 1):  # for each row
		if vidx % 10 == 0:
			print("row {0}".format(vidx))
		for hidx in range(1, width - window - 1):  # for each pixel in the row
			# get the sum of a horz chunk
			sample = idxToSum(hidx, hidx + samplesz, vidx, integral)
			# try and grok this, go thru a search window, calc the abs diff of sums
			# between or sample and the test window, toss in a list
			vals = [np.abs(sample - idxToSum(hidx + sidx, hidx + sidx + samplesz, vidx, integral)) for sidx in
					range(int(window * 0.5), window - samplesz)]
			# find the minimum match
			best = np.where(np.array(vals) == np.min(vals))[0]
			# offset is the hidx of the current window
			dmap[hidx][vidx] = best[-1]  # if we get > 1 use the furthest one
	# create the raw out
	queue.put(dmap)


# speed up image processing with parallelism
def parallelizeMatching(numProc, img, integral, window, samplesz):
	height = img.shape[0]
	width = img.shape[1]
	print(f"Integral: {integral.shape}")

	# create queues
	queues = [Queue() for i in range(0, numProc)]
	# spit the images and set up the processes
	processes = [Process(target=doMagicEye,
						 args=(img[i * height // numProc:(i + 1) * height // numProc, :],
							   integral[i * height // numProc:(i + 1) * height // numProc, :],
							   window, samplesz, queues[i], width, height))
				 for i in range(0, numProc)]
	# and go!
	[p.start() for p in processes]
	# get the chunks from the process
	chunks = [q.get() for q in queues]
	dmap = np.zeros([width - window, height], dtype='int32')
	# reassemble the chunks
	for i, chunk in zip(range(0, numProc), chunks):
		dmap[:, i * height // numProc:(i + 1) * height // numProc] = chunk
	# kill the processes
	[p.terminate() for p in processes]
	return dmap


if __name__ == "__main__":
	# check arguments
	if (len(sys.argv) > 3 or len(sys.argv) < 2):
		print("USAGE: DeMagicEye <infile> <outfile_stem> [search_window]")
		exit()

	# get input file, output stem, and optional search window
	ifile = str(sys.argv[1])
	stub = str(sys.argv[2])
	searchWndw = 1.1
	if (len(sys.argv) == 4):
		searchWndw = float(sys.argv[3])

	# open image and create integral image
	img = cv2.imread(ifile)
	height = img.shape[0]
	width = img.shape[1]
	# img = img.scale(1)
	# create the integral image
	print("!!!!!!!!!!!!!!!!!!", type(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))
	integral = cv2.integral(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

	# find our search window and make it big
	window = int(searchWndw * findOptimalWindow(img, integral))
	print(searchWndw)
	print("image: {0}x{1}".format(width, height))
	print("window: {0}".format(window))

	# how big of a signal we match on
	samplesz = window // 10
	print("sample: {0}".format(samplesz))
	numProc = 4
	dmap = parallelizeMatching(numProc, img, integral, window, samplesz)
	print(dmap.flags)
	#result = cv2.imdecode(dmap, cv2.IMREAD_GRAYSCALE)# Image(dmap)
	#result = cv2.imwrite(f"{stub}Equalized.png", sbs)
	cv2.imwrite('{0}RAW.png'.format(stub), dmap)

	# create the cleaned up output
	# read and convert to grayscale again due to opencv unsupported format error
	result = cv2.cvtColor(cv2.imread(f"{stub}RAW.png"), cv2.COLOR_BGR2GRAY)
	cv2.medianBlur(result, 3, result)  # medianFilter
	cv2.equalizeHist(result, result)  # equalize
	result = -result  # invert
	cv2.blur(result, (5, 5), result)  # blur window=(5, 5)
	cv2.imwrite('{0}Equalized.png'.format(stub), result)

	# I don't think side by side is that necessary, could add back later
	#sbs = img.sideBySide(result)
	#sbs.save('{0}.png'.format(stub))
