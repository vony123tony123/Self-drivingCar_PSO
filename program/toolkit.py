import numpy as np

class toolkit:
	def rotation_matrix(angle):
		theta = np.radians(angle)
		c, s = np.cos(theta), np.sin(theta)
		rotation_matrix = np.array([[c ,-s], [s, c]])
		return rotation_matrix


	# calculate cross point between vector and horizontal line 
	def calculate_crossPoint_in_horizontalLine(point, vector, y_value):
		if vector[1] == 0:
			return False
		time_of_vector = (y_value - point[1]) / vector[1]
		if time_of_vector < 0:
			return False
		x = vector[0] * time_of_vector + point[0]
		return [x, y_value]

	# calculate cross point between vector and vertical line 
	def calculate_crossPoint_in_verticalLine(point, vector, x_value):
		if vector[0] == 0:
			return False
		time_of_vector = (x_value - point[0]) / vector[0]
		if time_of_vector < 0:
			return False
		y = vector[1] * time_of_vector + point[1]
		return [x_value, y]


	# x = 1 dim matrix
	# m = 1 dim matrix
	# return scalar
	def euclid_distance(x, m):
	    distance = np.linalg.norm(x-m, axis=-1)
	    return distance

	# x = 2 dim matrix
	# m = 2 dim matrix
	# return 2 dim matrix
	def euclid_distance_2d(x, m):
	    x_minus_m_seperate = list(map(lambda a: a-m, x))
	    distance = np.linalg.norm(x_minus_m_seperate, axis=-1)
	    return distance

	def readFile(file):
		dataset = list()
		answers = list()
		with open(file, 'r') as f:
		    lines = f.readlines()
		    for line in lines:
		        data_list = line.split(" ")
		        dataset.append(list(map(float,data_list[:-1])))
		        answers.append(float(data_list[-1]))
		dataset = np.array(dataset)
		answers = np.array(list(map(lambda x: (x+40)/80, answers)))
		return dataset, answers

	def Kmeans(n_clusters, points, max_epochs):
	    randomIntArray = [random.randint(0,len(points)-1) for k in range(n_clusters)]
	    m = points[randomIntArray]
	    for epoch in range(max_epochs):
	        d = toolkit.euclid_distance_2d(points, m)
	        clusters = np.argmin(d, axis=-1)
	        m = [np.mean(points[clusters==k], axis=0) for k in range(n_clusters)]
	        m = np.array(m)
	    std = np.array([np.mean(toolkit.euclid_distance(points[clusters==k],m[k]), axis=0) for k in range(n_clusters)])
	    return m, std, clusters

	def guass_function(x, m, std):
	    return np.exp(-1 * (toolkit.euclid_distance(x, m)**2 / (2 * std**2)))


