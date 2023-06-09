from rbfn import RBFN
from toolkit import toolkit as tk
import numpy as np
import random
from drawplot import drawPlot

dataset, answers = tk.readFile("./train4dAll.txt")
K=10
reproduction_rate = 0.1
mapData = None
count = 0
Vmin = -20
Vmax = 20

"""
    產生隨機初始個體和速度:
    w_list為隨機產生
    m_list為選擇隨機train dataset之點
    std_list為隨機在5~20間產生
    n: 族群數量
    K: 群聚數量
"""
def initial_particles(num_particles = 50):
    particles = []
    for i in range(num_particles):
        w_list = np.random.randn(K+1)
        randomIntArray = [random.randint(0,len(dataset)-1) for k in range(K)]
        m_list = np.array(dataset[randomIntArray]).flatten()
        std_list = np.array([np.random.uniform(5,20) for i in range(K)])
        genetic = np.concatenate((w_list, m_list, std_list))
        particles.append(genetic)
    particles = np.array(particles)
    velocities = np.random.uniform(Vmin, Vmax, (num_particles, len(particles[0])))  # 初始化粒子的速度
    return particles, velocities


"""
    將基因解碼成RBFN之元素
"""
def decode(genetic):
    genetic = np.array(genetic)
    delta = genetic[0]
    w_list = genetic[1:(1+K)]
    m_list = genetic[K+1:3*K+(K+1)].reshape(-1,3)
    std_list = genetic[3*K+(K+1):]
    return delta, w_list, m_list, std_list


'''
    fitness_function: 讓基因實際去跑map(費時短)
'''
def fitness_function(particle, mapFilePath="map.txt"):
    global mapData
    delta, w, m, std = decode(particle)
    rbfn = RBFN()
    rbfn.setParams(delta, w, m, std)
    if mapData is None:
       mapData  = drawPlot.readMapFile(mapFilePath)
    original_point, goal_points, boarder_points = mapData[0], mapData[1], mapData[2]
    currentPoint = original_point[:-1]
    currentPhi = original_point[-1]
    original_point = original_point[:-1]
    currentVector = np.array([100, 0])
    step = 0
    bonus = 0
    while True:
        try:
            phi = currentPhi
            point = currentPoint
            sensor_vectors = drawPlot.getSensorVector(currentVector, phi)
            sensor_distances = []
            cross_points = []
            for sensor_vector in sensor_vectors:
                cross_point = drawPlot.findCrossPoint(boarder_points, point, sensor_vector)
                distance = tk.euclid_distance(cross_point, point)
                cross_points.append(cross_point)
                sensor_distances.append(distance)
            sensor_distances = np.array(sensor_distances).flatten()
            theta = rbfn.predict(sensor_distances)
            if drawPlot.is_Crash(point, boarder_points) == True:
                raise Exception("touch the wall of map")
            currentPoint, currentPhi = drawPlot.findNextState(point, phi, theta)
            if currentPoint[1] > min(goal_points[:,1]):
                bonus = 1000000
                break
            step +=1
            if step % 1000 == 0:
                break
        except Exception as e:
            break
    if step > 1000:
        return -1
    result = 0.3 * step * -1  + 0.7 * float(tk.euclid_distance(currentPoint, original_point)) + bonus
    if bonus != 0:
        result -= step * 10
    return result

if __name__ == "__main__":
    max_epochs = 100
    num_particles = 50
    particles, velocities = initial_particles(num_particles)
    best_positions = particles.copy()
    best_fitness_values = np.apply_along_axis(fitness_function, 1, particles)

    global_best_index = np.argmax(best_fitness_values)  # 初始化全域最佳位置的索引
    global_best_position = best_positions[global_best_index]  # 初始化全域最佳位置
    global_best_fitness = best_fitness_values[global_best_index]  # 初始化全域最佳適應值

    for i in range(max_epochs):
        cognition_lr = 0.08 * np.random.rand(num_particles, 1)
        social_lr = 0.08 * np.random.rand(num_particles, 1)
        w = 0.08 # 慣性權重控制著速度對於當前速度的重要性

        # 更新速度
        inertia_term = w * velocities # 慣性項（inertia term）用於平衡粒子的移動速度，並保持其過去的運動方向
        cognitive_term = cognition_lr * (best_positions - particles)
        social_term = social_lr  * (global_best_position - particles)
        velocities = inertia_term + cognitive_term + social_term


        # 更新位置
        particles += velocities
        
        # 計算適應值
        fitness_values = np.apply_along_axis(fitness_function, 1, particles)

        # 更新個體最佳解
        update_indices = fitness_values > best_fitness_values
        best_positions[update_indices] = particles[update_indices]
        best_fitness_values[update_indices] = fitness_values[update_indices]
        
        # 更新全域最佳解
        global_best_index = np.argmax(best_fitness_values)
        global_best_position = best_positions[global_best_index]
        global_best_fitness = best_fitness_values[global_best_index]
        

        print(f'{i+1}/{max_epochs}: Loss = {global_best_fitness}')
        print('--------')

    
        