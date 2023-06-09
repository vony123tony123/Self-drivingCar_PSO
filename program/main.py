# -*- coding: utf-8 -*-
import time
import traceback
import threading
from tqdm import tqdm

import matplotlib
from PyQt5.QtCore import pyqtSignal, QThread, QTimer, QEventLoop

import numpy as np
from Windows import Ui_MainWindow
from rbfn import RBFN
from PSO import *
from drawplot import drawPlot
import os
from toolkit import toolkit
# 导入程序运行必须模块
import sys
# PyQt5中使用的基本控件都在PyQt5.QtWidgets模块中
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
matplotlib.use('Qt5Agg')

# 設定gui的功能
class MyMainWindow(QMainWindow, Ui_MainWindow):

    step=0#用來判斷要不要創建plotpicture

    def __init__(self, parent=None):
        super(MyMainWindow, self).__init__(parent)
        self.setupUi(self)
        self.inputfilepath_btn_2.clicked.connect(self.choosefileDialog)
        self.pushButton_2.clicked.connect(self.startCar)
        self.canva = drawPlot()
        self.plot_layout.addWidget(self.canva)

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

    def choosefileDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filename, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                  "All Files (*);;Text Files (*.txt)", options=options)
        if filename:
            self.mapfilepath_edit.setText(filename)
        else:
            self.mapfilepath_edit.setText("")

    def startCar(self):
        mapFilePath = self.mapfilepath_edit.text()
        original_point, self.goal_points, self.boarder_points = drawPlot.readMapFile(mapFilePath)
        self.canva.drawMap(self.goal_points,self.boarder_points)

        num_particles = 100
        max_epochs = 100
        count_goal_epochs = 0
        fitness_values = []

        particles, velocities = initial_particles(num_particles)
        best_positions = particles.copy()
        best_fitness_values = np.apply_along_axis(fitness_function, 1, particles)

        global_best_index = np.argmax(best_fitness_values)  # 初始化全域最佳位置的索引
        global_best_position = best_positions[global_best_index]  # 初始化全域最佳位置
        global_best_fitness = best_fitness_values[global_best_index]  # 初始化全域最佳適應值

        # Training PSO Algorithm
        for i in range(max_epochs):
            cognition_lr = 0.2
            social_lr = 0.2
            w = 0.8 # 慣性權重控制著速度對於當前速度的重要性

            # 更新速度
            inertia_term = w * velocities # 慣性項（inertia term）用於平衡粒子的移動速度，並保持其過去的運動方向
            cognitive_term = cognition_lr * (best_positions - particles)
            social_term = social_lr  * (global_best_position - particles)
            velocities = inertia_term + cognitive_term + social_term
            velocities = np.clip(velocities, Vmin, Vmax)

            # 更新位置
            particles += velocities
            
            # 計算適應值
            fitness_values = np.apply_along_axis(fitness_function, 1, particles)

            # 更新個體最佳解和全域最佳解
            update_indices = fitness_values > best_fitness_values
            best_positions[update_indices] = particles[update_indices]
            best_fitness_values[update_indices] = fitness_values[update_indices]
            
            # 更新全域最佳解
            global_best_index = np.argmax(best_fitness_values)
            global_best_position = best_positions[global_best_index]
            global_best_fitness = best_fitness_values[global_best_index]
            

            print(f'{i+1}/{max_epochs}: best_fitness = {np.max(global_best_fitness)}')
            print(f"fitness_values: {max(fitness_values)}")
            print(f"best_fitness_values: {max(best_fitness_values)}")
            print('--------')
            # early stop epochs
            if max(fitness_values) > 10000:
                with open('goal_result.txt', 'a') as f:
                    f.write(f'{global_best_position}\n')
                count_goal_epochs += 1
            if count_goal_epochs >= 20:
                break
            
        '''
        Feed genes into RBFN network. Use resulting network to guide autonomous vehicle from start and plot travel path.
        '''
        self.RBFN = RBFN()
        delta, w_list, m_list, std_list = decode(global_best_position)
        self.RBFN.setParams(delta, w_list, m_list, std_list)

        self.currentPoint = original_point[:-1]
        self.currentPhi = original_point[-1]
        self.currentVector = np.array([100, 0])
        self.canva.clearPoints()

        self.loop = QEventLoop()
        self.timer = QTimer()
        self.timer.setInterval(1)
        self.timer.timeout.connect(self.updatePlot)
        self.timer.start()
        self.loop.exec_()
   
    def updatePlot(self):
        try:
            phi = self.currentPhi
            point = self.currentPoint
            self.canva.drawPoint(point)
            sensor_vectors = drawPlot.getSensorVector(self.currentVector, phi)
            sensor_distances = []
            cross_points = []
            for sensor_vector in sensor_vectors:
                cross_point = drawPlot.findCrossPoint(self.boarder_points, point, sensor_vector)
                #=self.canva.drawPoint(cross_point, 'r')
                distance = toolkit.euclid_distance(cross_point, point)
                cross_points.append(cross_point)
                sensor_distances.append(distance)
            sensor_distances = np.array(sensor_distances).flatten()
            self.canva.drawSensor(cross_points)
            theta = self.RBFN.predict(sensor_distances)
            if drawPlot.is_Crash(point, self.boarder_points):
                raise Exception("touch the wall of map")
            self.canva.updatePlot()
            self.currentPoint, self.currentPhi = drawPlot.findNextState(point, phi, theta)
            if self.currentPoint[1] > min(self.goal_points[:,1]):
                self.timer.stop()
                self.loop.quit()
        except Exception as e:
            print(e)
            traceback_output = traceback.format_exc()
            #print(traceback_output)
            self.timer.stop()
            self.loop.quit()
            # sys.exit(app.exec_())

if __name__ == "__main__":
    # 固定的，PyQt5程序都需要QApplication对象。sys.argv是命令行参数列表，确保程序可以双击运行
    app = QApplication(sys.argv)
    # 初始化
    myWin = MyMainWindow()

    # 将窗口控件显示在屏幕上
    myWin.show()
    # 程序运行，sys.exit方法确保程序完整退出。
    sys.exit(app.exec_())

