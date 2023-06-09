import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QSizePolicy
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle, Circle
from math import tan, radians
import numpy as np
import math as m
from toolkit import toolkit
from rbfn import RBFN
import traceback
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

RADIUS = 3

class drawPlot(FigureCanvas):
	def __init__(self,width=10,height=10,dpi=100):
		# 第一步：創建一個創建Figure
		self.fig = Figure(figsize=(width, height), dpi=dpi)
		# 第二步：在父類中激活Figure窗口
		super(drawPlot, self).__init__(self.fig)  # 此句必不可少，否則不能顯示圖形
		self.ax = self.fig.add_subplot(111)
		plt.xlim((-7, 31))
		plt.ylim((-4,51))
		# 第四步：就是畫圖，可以在此類中畫，也可以在其它類中畫,最好是在別的地方作圖
		FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
		FigureCanvas.updateGeometry(self)
		self.points = list()

	def drawMap(self, goal_points, boarder_points):
		self.points.clear()
		self.goal_points = goal_points
		self.boarder_points = boarder_points


	def drawPoint(self, point, color='b'):
		self.points.append([point[0], point[1], color])
		self.ax.scatter(point[0], point[1], color=color)

	def updatePlot(self):
		self.ax.cla()
		self.ax.plot(self.boarder_points[:,0], self.boarder_points[:,1])
		self.ax.add_patch(Rectangle((self.goal_points[0,0], self.goal_points[1,1]), 
			  width = self.goal_points[1,0] - self.goal_points[0,0], height = self.goal_points[0,1] - self.goal_points[1,1]))
		if self.points:
			point = self.points[-1]
			if point[2] == 'b':
				self.ax.add_patch(Circle(point[:2], RADIUS, fill = False))
			for point_i in self.points:
				self.ax.scatter(point_i[0], point_i[1], color='b')
		for cross_point in self.cross_points:
			self.ax.scatter(cross_point[0], cross_point[1], c=[[0,1,0]])
			distance = toolkit.euclid_distance(cross_point, point[:-1])
			self.ax.text(cross_point[0]-0.5, cross_point[1]+0.5, round(distance,2))
			self.ax.plot([cross_point[0], point[0]], [cross_point[1], point[1]], 'g')
		self.draw()

	def readMapFile(mapFile):
		goal_points = list()
		boarder_points = list()
		with open(mapFile, 'r') as f:
			lines = f.readlines()
			for i,line in enumerate(lines):
				line = list(map(float, line.strip('\n').split(',')))
				if i == 0 :
					original_point = line
				elif i == 1 or i == 2 :
					goal_points.append(line)
				else:
					boarder_points.append(line)
		original_point = np.array(original_point)
		goal_points = np.array(goal_points)
		boarder_points = np.array(boarder_points)
		return original_point, goal_points, boarder_points

	def findCrossPoint(boarder_points, point, vector):
		cross_points = list()
		for i in range(len(boarder_points) - 1):
			result = boarder_points[i] - boarder_points[i+1]
			if result[0] == 0:
				cross_point = toolkit.calculate_crossPoint_in_verticalLine(point, vector, boarder_points[i][0])
				min_y = min(boarder_points[i][1], boarder_points[i+1][1])
				max_y = max(boarder_points[i][1], boarder_points[i+1][1])
				if cross_point == False or cross_point[1] < min_y or cross_point[1] > max_y:
					continue
				cross_points.append(cross_point)
			elif result[1] == 0:
				cross_point = toolkit.calculate_crossPoint_in_horizontalLine(point, vector, boarder_points[i][1])
				min_x = min(boarder_points[i][0], boarder_points[i+1][0])
				max_x = max(boarder_points[i][0], boarder_points[i+1][0])
				if cross_point == False or cross_point[0] < min_x or cross_point[0] > max_x:
					continue
				cross_points.append(cross_point)
			else:
				raise Exception("findCrossPoint error")
		cross_points = np.array(cross_points)
		min_distance_index = np.argmin(toolkit.euclid_distance(cross_points, point))
		return cross_points[min_distance_index]

	def findNextState(point, phi, theta):
		x = point[0]
		y = point[1]
		r_phi = m.radians(phi)
		r_theta = m.radians(theta)

		x = x + m.cos(r_phi + r_theta) + m.sin(r_theta) * m.sin(r_phi)
		y = y + m.sin(r_phi + r_theta) - m.sin(r_theta) * m.cos(r_phi)
		phi = phi - m.degrees(m.asin(2 * m.sin(r_theta) / 6))

		return [x, y], phi

	def getSensorVector(vector, angle):
		center_rotation_matrix = toolkit.rotation_matrix(angle)
		center_vector = np.dot(center_rotation_matrix, vector.T)

		left_rotation_matrix = toolkit.rotation_matrix(angle + 45)
		left_vector = np.dot(left_rotation_matrix, vector.T)

		right_rotation_matrix = toolkit.rotation_matrix(angle - 45)
		right_vector = np.dot(right_rotation_matrix, vector.T)

		return [center_vector, right_vector, left_vector]
	
	def clearPoints(self):
		self.points.clear()

	def distance_point_to_line_segment(point, seg_start, seg_end):
		AB = np.array([seg_end[0] - seg_start[0], seg_end[1] - seg_start[1]])
		AP = np.array([point[0] - seg_start[0], point[1] - seg_start[1]])

		dot = np.dot(AB, AP)
		AB_norm = np.linalg.norm(AB)
		
		if dot <= 0:
			return np.linalg.norm(AP)
		elif dot >= AB_norm * AB_norm:
			return np.linalg.norm(point - seg_end)
		else:
			return np.abs(np.cross(AB, AP)) / AB_norm

	def is_Crash(pt, boarder_points):
		for i in range(1,len(boarder_points)):
			distance = drawPlot.distance_point_to_line_segment(pt, boarder_points[i-1], boarder_points[i])
			if distance < 2.8:
				return True
		return False

	def drawSensor(self, cross_points):
		self.cross_points = cross_points


if __name__ == "__main__":

	datafile = "D:/Project/NNHomework/NN_HW2/train4dAll.txt"
	mapfile = "D:/Project/NNHomework/NN_HW2/軌道座標點.txt"

	RBFN = RBFN()
	dataset, answers = RBFN.readFile(datafile)
	RBFN.train(dataset, answers, max_epochs = 100, lr = 0.01)

	drawPlot = drawPlot()
	original_point, goal_points, boarder_points = readMapFile(mapfile)
	drawPlot.drawMap(goal_points, boarder_points)
	drawPlot.drawPoint(original_point)

	point = original_point[:-1]
	phi = original_point[-1]
	vector = np.array([100,0])
	radius = 6

	while point[1] < 37:
		try:
			drawPlot.drawPoint(point)
			sensor_vectors = getSensorVector(vector, phi)
			sensor_distances = list()
			drawPlot.drawPoint(point)
			for sensor_vector in sensor_vectors:
				cross_point = findCrossPoint(boarder_points, point, sensor_vector)
				drawPlot.drawPoint(cross_point, 'r')
				distance = toolkit.euclid_distance(cross_point, point)
				sensor_distances.append(distance)
			sensor_distances = np.array(sensor_distances).flatten()
			theta = RBFN.predict(sensor_distances)
			print('distance = ',sensor_distances)
			print('point = ',point)
			print('phi =', phi)
			print('theta = ', theta)
			print('---------------')
			point, phi = findNextState(point, phi, theta, radius)
		except Exception as e:
			print(e)
			traceback_output = traceback.format_exc()
			print(traceback_output)
			break


	plt.show()