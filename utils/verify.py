#!/usr/bin/env python

import array
import struct
import sys
import math

DIM = 2
eps = 1e-3

relative_eps = 1e-5

class Verify:
	def __init__(self, answer_file, output_file, input_file):
		self.n_centroids, self.n_data, self.centroid, self.cluster = zip(self.read_output(answer_file), self.read_output(output_file))
		self.input_centroids, self.input_data =  self.read_input_centroids(input_file)
		
	def read_output(self, file_name):
		with open(file_name, 'rb') as input_f:
			cls = struct.unpack('I', input_f.read(struct.calcsize('I')))[0]
			cnt = struct.unpack('I', input_f.read(struct.calcsize('I')))[0]
			centroids = array.array('f')
			data = array.array('i')
			centroids.fromfile(input_f, cls * DIM)
			data.fromfile(input_f, cnt)
		return cls, cnt, centroids, data

	def read_input_centroids(self, file_name):
		with open(file_name, 'rb') as input_f:
			cls = struct.unpack('I', input_f.read(struct.calcsize('I')))[0]
			cnt = struct.unpack('I', input_f.read(struct.calcsize('I')))[0]
			centroids = array.array('f')
			data = array.array('f')
			centroids.fromfile(input_f, cls * DIM)
			data.fromfile(input_f, cnt * DIM)
		return centroids, data;
			
			

	def verify(self):
		assert self.n_centroids[0] == self.n_centroids[1]
		assert self.n_data[0] == self.n_data[1]

		result_c, result_d = True, True

		xy1 = zip(self.centroid[0][0::2], self.centroid[0][1::2])
		xy2 = zip(self.centroid[1][0::2], self.centroid[1][1::2])
		for idx, ((x1, y1), (x2, y2)) in enumerate(zip(xy1, xy2)):
			if math.fabs(x1 - x2) > eps or math.fabs(y1 - y2) > eps:
				pass
				#print 'centroids[%d] : ans=(%e, %e), out=(%e, %e)' % (idx, x1, y1, x2, y2)
				#result_c = False

		for idx, (c1, c2) in enumerate(zip(self.cluster[0], self.cluster[1])):
			if c1 != c2:
				point = self.input_data[2*idx:2*idx+2]
				input_c1 = self.input_centroids[2*c1:2*c1+2]
				input_c2 = self.input_centroids[2*c2:2*c2+2]
				dist1 = (point[0] - input_c1[0]) * (point[0] - input_c1[0]) + (point[1] - input_c1[1]) * (point[1] - input_c1[1])
				dist2 = (point[0] - input_c2[0]) * (point[0] - input_c2[0]) + (point[1] - input_c2[1]) * (point[1] - input_c2[1])
				relative_diff = math.fabs(dist2 - dist1) / math.fabs(dist1)
				if relative_diff > relative_eps:
					print 'partitioned[%d] : ans=%d, out=%d, relative_diff=%f' % (idx, c1, c2, relative_diff)
					result_d = False

		if result_c and result_d:
			print 'Verification Success'
		else:
			print 'Verification Failed'

if __name__ == '__main__':
	if len(sys.argv) < 4:
		print('{0} <answer file> <output file> <input file>'.format(sys.argv[0]))
		sys.exit(1)
	test = Verify(sys.argv[1], sys.argv[2], sys.argv[3])
	test.verify()
