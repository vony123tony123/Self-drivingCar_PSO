import numpy as np
import random
import matplotlib.pyplot as plt
from toolkit import toolkit as tk

class RBFN:
	def forward(x, m, std, w, delta):
	    guass = tk.guass_function(x,m,std)
	    return np.dot(guass, w) + delta

	def optimize(lr, data, ans, F, guass_list, m_list, std_list, w_list, delta):
		m_gradient = (ans - F) * w_list * guass_list / std_list**2
		m_gradient = np.array([m_gradient]).T
		m_gradient = m_gradient * (data - m_list)
		m_after = m_list + lr * m_gradient

		std_graident = (ans - F) * w_list * guass_list / std_list**3
		std_graident = std_graident * toolkit.euclid_distance([data], m_list)[0]**2
		std_after = std_list + lr * std_graident
		
		w_after = w_list + lr * (ans - F) * guass_list
		delta_after = delta + lr * (ans - F)

		return m_after, std_after, w_after, delta_after

	def train(self, dataset, answers, K=3, lr=0.1, max_epochs=100):
		w_list = np.random.randn(K)
		delta = np.random.randn()
		m_list,std_list ,clusters = tk.Kmeans(K, dataset, 100)

		# print("m_list = ", m_list)
		# print("w_list = ",w_list)
		# print("delta = ", delta)
		# print("std_list = ", std_list)

		for epoch in range(max_epochs):
			ME = 0
			for data, ans in zip(dataset, answers):
				F = RBFN.forward(data, m_list, std_list, w_list, delta)
				E = (ans - F)**2/2
				ME = ME + E

				# print("data = ", data)
				# print("m_list = ", m_list)
				# print("guass_list = ",guass_list)
				# print("w_list = ",w_list)
				# print("delta = ", delta)
				# print("std_list = ", std_list)
				# print("ans = ", ans)
				# print("F = ",F)
				guass_list = tk.guass_function(data, m_list,std_list)
				m_gradient = (ans - F) * w_list * guass_list / std_list**2
				m_gradient = np.array([m_gradient]).T
				m_gradient = m_gradient * (data - m_list)
				m_after = m_list + lr * m_gradient

				std_graident = (ans - F) * w_list * guass_list / std_list**3
				std_graident = std_graident *np.sum((data - m_list)**2, axis = -1)
				std_after = std_list + lr * std_graident
				
				w_after = w_list + lr * (ans - F) * guass_list
				delta_after = delta + lr * (ans - F)

				m_list = m_after
				std_list = std_after
				w_list = w_after
				delta = delta_after

				# print("---------------------------")
				# print("m_list = ", m_list)
				# print("w_list = ",w_list)
				# print("delta = ", delta)
				# print("std_list = ", std_list)
				# break
			ME = ME / len(dataset)
			# if epoch % 100 == 0:
			# 	print("Epoch {} : mean loss = {}".format(epoch, ME))
			print("Epoch {} : mean loss = {}".format(epoch, ME))

		self.m_list = m_list
		self.std_list = std_list
		self.w_list = w_list
		self.delta = delta

	def predict(self, data):
		guass = tk.guass_function(data,self.m_list,self.std_list)
		result = np.dot(guass, self.w_list) + self.delta
		return result

	def setParams(self, delta, w_list, m_list, std_list):
		self.delta =  delta
		self.w_list = w_list
		self.m_list = m_list
		self.std_list = std_list

if __name__ == "__main__":
	read_file = "./train4dAll.txt"

	RBFN = RBFN()

	dataset, answers = toolkit.readFile(read_file)

	RBFN.train(dataset, answers, max_epochs = 1)
