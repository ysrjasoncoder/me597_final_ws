#!/usr/bin/env python3
import sys
import os
import numpy as np
import rclpy
from rclpy.node import Node as rclpy_node
from rclpy.duration import Duration
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Pose, Twist
from std_msgs.msg import Float32
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

from tf2_ros import Buffer, TransformListener
import tf2_py
from geometry_msgs.msg import TransformStamped

import ament_index_python.packages
import time
import heapq
import numpy as np
from copy import copy
from PIL import Image, ImageOps
import numpy as np
import yaml
import pandas as pd    
from collections import deque

import numpy as np

class Queue():
    def __init__(self, init_queue = []):
        self.queue = copy(init_queue)
        self.start = 0
        self.end = len(self.queue)-1

    def __len__(self):
        numel = len(self.queue)
        return numel

    def __repr__(self):
        q = self.queue
        tmpstr = ""
        for i in range(len(self.queue)):
            flag = False
            if(i == self.start):
                tmpstr += "<"
                flag = True
            if(i == self.end):
                tmpstr += ">"
                flag = True

            if(flag):
                tmpstr += '| ' + str(q[i]) + '|\n'
            else:
                tmpstr += ' | ' + str(q[i]) + '|\n'

        return tmpstr

    def __call__(self):
        return self.queue

    def initialize_queue(self,init_queue = []):
        self.queue = copy(init_queue)

    def sort(self,key=str.lower):
        self.queue = sorted(self.queue,key=key)

    def push(self,data):
        self.queue.append(data)
        self.end += 1

    def pop(self):
        p = self.queue.pop(self.start)
        self.end = len(self.queue)-1
        return p

class Node():
    def __init__(self,name):
        self.name = name
        self.children = []
        self.weight = []

    def __repr__(self):
        return self.name

    def add_children(self,node,w=None):
        if w == None:
            w = [1]*len(node)
        self.children.extend(node)
        self.weight.extend(w)

class Tree():
    def __init__(self,name):
        self.name = name
        self.root = 0
        self.end = 0
        self.g = {}


    def add_node(self, node, start = False, end = False):
        self.g[node.name] = node
        if(start):
            self.root = node.name
        elif(end):
            self.end = node.name

    def set_as_root(self,node):
        # These are exclusive conditions
        self.root = True
        self.end = False

    def set_as_end(self,node):
        # These are exclusive conditions
        self.root = False
        self.end = True

import yaml
from PIL import Image
from dataclasses import dataclass
from math import radians
from math import cos, sin, floor
@dataclass
class MapInfo:
    resolution: float
    width: int
    height: int
    origin_x: float
    origin_y: float
    origin_yaw: float = 0.0
    y_down: bool = False

def load_map_info_from_yaml(yaml_path, y_down=False):
    """
    从ROS地图YAML文件中加载地图元数据（resolution, origin, width, height）
    并返回 MapInfo 对象。
    """
    # 1. 解析 YAML 文件
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    image_path = data['image']
    resolution = float(data['resolution'])
    origin = data['origin']          # [x, y, yaw]
    origin_x, origin_y, origin_yaw = origin

    # 2. 打开 .pgm 图像获取宽高
    img = Image.open(yaml_path.replace('yaml','pgm'))
    width, height = img.size  # 注意：Pillow 返回 (width, height)

    # 3. 构建 MapInfo 对象
    map_info = MapInfo(
        resolution=resolution,
        width=width,
        height=height,
        origin_x=origin_x,
        origin_y=origin_y,
        origin_yaw=origin_yaw,  # 通常为0
        y_down=y_down
    )

    return map_info

class Map():
    def __init__(self, map_yaml_path):
        self.map_im, self.map_df, self.limits = self.__open_map(map_yaml_path)
        self.image_array = self.__get_obstacle_map(self.map_im, self.map_df)
        self.mpInfo = load_map_info_from_yaml(map_yaml_path, y_down=True)


    def __open_map(self,map_yaml_path):
        # Open the YAML file which contains the map name and other
        # configuration parameters
        f = open(map_yaml_path, 'r')
        map_df = pd.json_normalize(yaml.safe_load(f))
        # Open the map image
        pgm_file  = map_df.image[0]
        pgm_file_path = map_yaml_path.replace('yaml','pgm')
        im = Image.open(pgm_file_path)
        # size = 200, 200
        # im.thumbnail(size)
        # im = ImageOps.grayscale(im)
        # Get the limits of the map. This will help to display the map
        # with the correct axis ticks.
        xmin = map_df.origin[0][0]
        xmax = map_df.origin[0][0] + im.size[0] * map_df.resolution[0]
        ymin = map_df.origin[0][1]
        ymax = map_df.origin[0][1] + im.size[1] * map_df.resolution[0]

        return im, map_df, [xmin,xmax,ymin,ymax]

    def __get_obstacle_map(self,map_im, map_df):
        img_array = np.reshape(list(self.map_im.getdata()),(self.map_im.size[1],self.map_im.size[0]))
        up_thresh = self.map_df.occupied_thresh[0]*255
        low_thresh = self.map_df.free_thresh[0]*255

        for j in range(self.map_im.size[0]):
            for i in range(self.map_im.size[1]):
                if img_array[i,j] > up_thresh:
                    img_array[i,j] = 255
                else:
                    img_array[i,j] = 0
        return img_array
    
    def _world_to_map_continuous(self, wx: float, wy: float):
        """
        世界坐标 -> 连续栅格坐标（未取整），在 OccupancyGrid 语义下：
        - (0,0) 栅格中心位于 (origin_x, origin_y)
        - 若 origin_yaw != 0，会做逆旋转将世界点转回地图坐标轴
        """
        m = self.mpInfo
        # 平移到以 origin 为参考
        dx = wx - m.origin_x
        dy = wy - m.origin_y

        # 逆旋转（世界 -> 地图坐标轴）
        c, s = cos(-m.origin_yaw), sin(-m.origin_yaw)
        mx_cont = (dx * c - dy * s) / m.resolution
        my_cont = (dx * s + dy * c) / m.resolution

        # 这里的 (0,0) 意味着第 0 个栅格的中心
        return mx_cont, my_cont

    def pose_to_grid(self, wx: float, wy: float, clamp: bool = False):
        """
        世界坐标(米) -> 栅格索引(int)。
        默认采用“向下取整到所在格子”的策略：ix = floor(mx_cont + 0.5)
        因为 (0,0) 指第 0 格中心，所以 +0.5 能把中心对齐到整数格。
        """
        m = self.mpInfo
        mx_cont, my_cont = self._world_to_map_continuous(wx, wy)

        # 把“以格中心为原点”的连续坐标转为“以格索引为整数”
        ix = int(floor(mx_cont + 0.5))
        iy = int(floor(my_cont + 0.5))

        # 如果使用图像坐标（y 向下，左上角 0,0）
        if m.y_down:
            iy = (m.height - 1) - iy

        if clamp:
            ix = max(0, min(m.width - 1, ix))
            iy = max(0, min(m.height - 1, iy))

        # 越界检查（不 clamp 时给出提示）
        if not clamp and (ix < 0 or ix >= m.width or iy < 0 or iy >= m.height):
            raise ValueError(f"grid index out of bounds: ({ix}, {iy})")

        return ix, iy

    def grid_to_pose(self, ix: int, iy: int):
        """
        栅格索引 -> 世界坐标（返回该格**中心**的世界系 (x,y)）。
        """
        m = self.mpInfo
        if ix < 0 or ix >= m.width or iy < 0 or iy >= m.height:
            raise ValueError(f"grid index out of bounds: ({ix}, {iy})")

        # 图像坐标转换回常规地图坐标（如有需要）
        if m.y_down:
            iy = (m.height - 1) - iy

        # 先得到以地图坐标轴为基的连续坐标（以 0 格中心为 0）
        mx_cont = float(ix)
        my_cont = float(iy)

        # 转回以 origin 为参考的米制坐标（先缩放再旋转）
        dx_local = mx_cont * m.resolution
        dy_local = my_cont * m.resolution

        # 旋转到世界坐标轴（地图 -> 世界）
        c, s = cos(m.origin_yaw), sin(m.origin_yaw)
        wx = m.origin_x + (dx_local * c - dy_local * s)
        wy = m.origin_y + (dx_local * s + dy_local * c)

        return wx, wy

class MapProcessor():
    def __init__(self,name):
        self.map = Map(name)
        self.inf_map_img_array = np.zeros(self.map.image_array.shape)
        self.map_graph = Tree(name)

    def __modify_map_pixel(self,map_array,i,j,value,absolute):
        if( (i >= 0) and
            (i < map_array.shape[0]) and
            (j >= 0) and
            (j < map_array.shape[1]) ):
            if absolute:
                map_array[i][j] = value
            else:
                map_array[i][j] += value

    def __inflate_obstacle(self,kernel,map_array,i,j,absolute):
        dx = int(kernel.shape[0]//2)
        dy = int(kernel.shape[1]//2)
        if (dx == 0) and (dy == 0):
            self.__modify_map_pixel(map_array,i,j,kernel[0][0],absolute)
        else:
            for k in range(i-dx,i+dx):
                for l in range(j-dy,j+dy):
                    self.__modify_map_pixel(map_array,k,l,kernel[k-i+dx][l-j+dy],absolute)

    def inflate_map(self,kernel,absolute=True):
        # Perform an operation like dilation, such that the small wall found during the mapping process
        # are increased in size, thus forcing a safer path.
        self.inf_map_img_array = np.zeros(self.map.image_array.shape)
        for i in range(self.map.image_array.shape[0]):
            for j in range(self.map.image_array.shape[1]):
                if self.map.image_array[i][j] == 0:
                    self.__inflate_obstacle(kernel,self.inf_map_img_array,i,j,absolute)
        r = np.max(self.inf_map_img_array)-np.min(self.inf_map_img_array)
        if r == 0:
            r = 1
        self.inf_map_img_array = (self.inf_map_img_array - np.min(self.inf_map_img_array))/r

    def get_graph_from_map(self):
        # Create the nodes that will be part of the graph, considering only valid nodes or the free space
        for i in range(self.map.image_array.shape[0]):
            for j in range(self.map.image_array.shape[1]):
                if self.inf_map_img_array[i][j] == 0:
                    node = Node('%d,%d'%(i,j))
                    self.map_graph.add_node(node)
        # Connect the nodes through edges
        for i in range(self.map.image_array.shape[0]):
            for j in range(self.map.image_array.shape[1]):
                if self.inf_map_img_array[i][j] == 0:
                    if (i > 0):
                        if self.inf_map_img_array[i-1][j] == 0:
                            # add an edge up
                            child_up = self.map_graph.g['%d,%d'%(i-1,j)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_up],[1])
                    if (i < (self.map.image_array.shape[0] - 1)):
                        if self.inf_map_img_array[i+1][j] == 0:
                            # add an edge down
                            child_dw = self.map_graph.g['%d,%d'%(i+1,j)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_dw],[1])
                    if (j > 0):
                        if self.inf_map_img_array[i][j-1] == 0:
                            # add an edge to the left
                            child_lf = self.map_graph.g['%d,%d'%(i,j-1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_lf],[1])
                    if (j < (self.map.image_array.shape[1] - 1)):
                        if self.inf_map_img_array[i][j+1] == 0:
                            # add an edge to the right
                            child_rg = self.map_graph.g['%d,%d'%(i,j+1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_rg],[1])
                    if ((i > 0) and (j > 0)):
                        if self.inf_map_img_array[i-1][j-1] == 0:
                            # add an edge up-left
                            child_up_lf = self.map_graph.g['%d,%d'%(i-1,j-1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_up_lf],[np.sqrt(2)])
                    if ((i > 0) and (j < (self.map.image_array.shape[1] - 1))):
                        if self.inf_map_img_array[i-1][j+1] == 0:
                            # add an edge up-right
                            child_up_rg = self.map_graph.g['%d,%d'%(i-1,j+1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_up_rg],[np.sqrt(2)])
                    if ((i < (self.map.image_array.shape[0] - 1)) and (j > 0)):
                        if self.inf_map_img_array[i+1][j-1] == 0:
                            # add an edge down-left
                            child_dw_lf = self.map_graph.g['%d,%d'%(i+1,j-1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_dw_lf],[np.sqrt(2)])
                    if ((i < (self.map.image_array.shape[0] - 1)) and (j < (self.map.image_array.shape[1] - 1))):
                        if self.inf_map_img_array[i+1][j+1] == 0:
                            # add an edge down-right
                            child_dw_rg = self.map_graph.g['%d,%d'%(i+1,j+1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_dw_rg],[np.sqrt(2)])

    def gaussian_kernel(self, size, sigma=1):
        size = int(size) // 2
        x, y = np.mgrid[-size:size+1, -size:size+1]
        normal = 1 / (2.0 * np.pi * sigma**2)
        g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
        r = np.max(g)-np.min(g)
        sm = (g - np.min(g))*1/r
        return sm

    def rect_kernel(self, size, value):
        m = np.ones(shape=(size,size))
        return m

    def draw_path(self,path):
        path_tuple_list = []
        path_array = copy(self.inf_map_img_array)
        for idx in path:
            tup = tuple(map(int, idx.split(',')))
            path_tuple_list.append(tup)
            path_array[tup] = 0.5
        return path_array

class PID:
    def __init__(self, kp, ki, kd, setpoint, integral_separation_threshold, deadband):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_separation_threshold = integral_separation_threshold
        self.deadband = deadband

        self.prev_error = 0.0
        self.integral = 0.0

    def compute(self, current_value, target_value):
        error = target_value - current_value
        
        if abs(error) < self.integral_separation_threshold:
            self.integral += error
        else:
            self.integral = 0.0
        
        derivative = error - self.prev_error
        
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        
        self.prev_error = error
        
        if abs(error) <= self.deadband:
            output = 0.0
        return output

class AStar():
    def __init__(self, G, is_blocked=None):
        self.G = G                          # Tree
        self.q = Queue([])                  # open set
        self.closed = set()                 # closed set（可选）
        # 预初始化 dist / via / fscore
        self.dist = {name: float('inf') for name in self.G.g.keys()}  # g(n)
        self.via  = {name: None for name in self.G.g.keys()}
        self.fscore = {name: float('inf') for name in self.G.g.keys()} # f(n)

        self.is_blocked = is_blocked

    def __parse_name_to_ij(self, name):
        # name 格式如 "i,j"
        i, j = name.split(',')
        return int(i), int(j)

    def __heuristic(self, a, b):
        # 欧式距离（与 8 邻接 + √2/1 权重一致，且可采纳、一致）
        ai, aj = self.__parse_name_to_ij(a)
        bi, bj = self.__parse_name_to_ij(b)
        di, dj = abs(ai - bi), abs(aj - bj)
        return (di**2 + dj**2) ** 0.5

        # 若想更贴近 1 / √2 的代价，也可用 octile：
        # dmin = min(di, dj); dmax = max(di, dj)
        # return dmax + (2**0.5 - 1) * dmin

    def __get_f_score(self, node):
        # node 是 Node 实例或名字；BFS/Dijkstra 用的是名字，这里统一支持
        name = node.name if hasattr(node, 'name') else node
        return self.fscore[name]

    def solve(self, sn, en):
        # sn、en 均为 Node 实例
        s, t = sn.name, en.name
        self.dist[s] = 0.0
        self.fscore[s] = self.__heuristic(s, t)
        self.q.push(sn)

        while len(self.q) > 0:
            # 取 f 最小的节点（注意 list.sort 是升序）
            self.q.sort(key=self.__get_f_score)
            u = self.q.pop()         # pop 从队头取最小（Queue.pop 用 start=0）
            u_name = u.name

            if u_name == t:
                break
            if u_name in self.closed:
                continue
            self.closed.add(u_name)

            # 扩展邻居
            for i, c in enumerate(u.children):

                if self.is_blocked is not None and self.is_blocked(c.name):
                    continue

                w = u.weight[i]
                tentative_g = self.dist[u_name] + w
                if tentative_g < self.dist[c.name]:
                    self.dist[c.name] = tentative_g
                    self.via[c.name] = u_name
                    self.fscore[c.name] = tentative_g + self.__heuristic(c.name, t)
                    # 放入 open（可能重复入队，但因 dist 更小，后续会被选中）
                    self.q.push(c)

        return self.via

    def reconstruct_path(self, sn, en):
        path = []
        dist = 0.0
        cur = en.name
        if self.via[cur] is None and cur != sn.name:
            # 不可达
            return path, float('inf')

        while cur is not None:
            path.append(cur)
            if cur == sn.name:
                break
            cur = self.via[cur]

        path.reverse()
        dist = self.dist[en.name]
        return path, dist


class PathFinder():
    def __init__(self, node:rclpy_node, map_name = 'sync_classroom_map'):
        package_share_directory = ament_index_python.packages.get_package_share_directory('turtlebot3_gazebo')
        map_yaml_path = os.path.join(package_share_directory, 'maps', map_name + '.yaml')
        self.node = node
        self.mp = MapProcessor(map_yaml_path)
        kr = self.mp.rect_kernel(10,1) # sim: 10
        self.mp.inflate_map(kr,True)
        self.mp.get_graph_from_map()
        self.node.get_logger().info('Map: ' + map_name + 'loaded.')

        self.dynamic_costmap = np.zeros_like(self.mp.inf_map_img_array, dtype=np.uint8)
        # 动态障碍多帧确认计数器
        self.dynamic_confirm = np.zeros_like(self.mp.inf_map_img_array, dtype=np.uint8)

        # 参数（你可以调）
        self.confirm_frames = 3      # 连续命中 3 帧才算障碍
        self.decay_per_frame = 1     # 每帧衰减 1（没命中则减少）
        self.hit_increment = 2       # 命中时增加 2（快一点确认）
        self.inflation_radius = 2        # 动态障碍膨胀半径（格）
        self.inflation_add = 1           # 膨胀圈每格 +1（比中心弱）



    # change the scale of the map, from world to pixel map
    def world_to_pixel(self, x, y, clamp=False)->tuple[int,int]:
        # x_origin, y_origin = 73, 56 # sim: 73, 56 
        # resolution = 1/0.075 # sim: 0.075 
        # pixel_x = x_origin + x * resolution
        # pixel_y = y_origin - y * resolution
        
        # return int(round(pixel_x)), int(round(pixel_y))
        return self.mp.map.pose_to_grid(x, y,clamp=clamp)
    
    def pixel_to_world(self, pixel_x, pixel_y, min_x=-5.4, max_y=4.2, scale_x=0.0745, scale_y=0.0741)->tuple:
        # x_origin, y_origin = 72, 56 
        # resolution = 1/0.075 
        # world_x = (pixel_x - x_origin) / resolution
        # world_y = (y_origin - pixel_y) / resolution

        # return world_x, world_y
        return self.mp.map.grid_to_pose(pixel_x, pixel_y)

    def build_a_pose(self, world_x, world_y):
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.node.get_clock().now().to_msg()
        pose.pose.position.x = world_x
        pose.pose.position.y = world_y
        return pose

    def find(self, start_pose, end_pose):
        path = Path()
        path.header.frame_id = 'map'
        path.header.stamp = self.node.get_clock().now().to_msg()

        start_pose_ = self.build_a_pose(start_pose.pose.position.x, start_pose.pose.position.y)
        path.poses.append(start_pose_)
        ##Fixed important bug
        # world → pixel（强制 clamp，防越界）
        si, sj = self.world_to_pixel(
            start_pose.pose.position.x,
            start_pose.pose.position.y,
            clamp=True
        )
        gi, gj = self.world_to_pixel(
            end_pose.pose.position.x,
            end_pose.pose.position.y,
            clamp=True
        )

        # 注意：map_graph 用的是 "row,col" = "i,j"
        start_ij = (sj, si)
        goal_ij  = (gj, gi)

        # ---- 修正起点 ----
        if (self.dynamic_costmap[start_ij] != 0 or
            self.mp.inf_map_img_array[start_ij] != 0):
            fixed = self.find_nearest_free_cell(start_ij)
            if fixed is None:
                raise ValueError("No free start cell found")
            self.node.get_logger().warn(
                f"Start blocked, moved from {start_ij} to {fixed}"
            )
            start_ij = fixed

        # ---- 修正终点 ----
        if (self.dynamic_costmap[goal_ij] != 0 or
            self.mp.inf_map_img_array[goal_ij] != 0):
            fixed = self.find_nearest_free_cell(goal_ij)
            if fixed is None:
                raise ValueError("No free goal cell found")
            self.node.get_logger().warn(
                f"Goal blocked, moved from {goal_ij} to {fixed}"
            )
            goal_ij = fixed

        # 写回 graph
        self.mp.map_graph.root = f"{start_ij[0]},{start_ij[1]}"
        self.mp.map_graph.end  = f"{goal_ij[0]},{goal_ij[1]}"

        self.node.get_logger().info('Start: {}, End: {}'.format(self.mp.map_graph.root, self.mp.map_graph.end))
        #self.node.get_logger().info('Start: {}, End: {}'.format(self.mp.map_graph.root, self.mp.map_graph.end))
        
        def is_blocked(name: str) -> bool:
            i, j = map(int, name.split(','))  # name 形式是 "i,j" = "row,col"
            # 边界保护（理论上不会越界，但习惯性防一手）
            if i < 0 or i >= self.mp.inf_map_img_array.shape[0]:
                return True
            if j < 0 or j >= self.mp.inf_map_img_array.shape[1]:
                return True
            # 只看动态 costmap：1 表示有障碍
            return self.dynamic_costmap[i, j] != 0

        as_maze = AStar(self.mp.map_graph, is_blocked=is_blocked)
        start = time.time()
        as_maze.solve(self.mp.map_graph.g[self.mp.map_graph.root],self.mp.map_graph.g[self.mp.map_graph.end])
        self.node.get_logger().info('A* time: {:.4f} seconds'.format(time.time() - start))
        path_as,dist_as = as_maze.reconstruct_path(self.mp.map_graph.g[self.mp.map_graph.root],self.mp.map_graph.g[self.mp.map_graph.end])
        if len(path_as) == 0:
            raise ValueError('A* failed to find a path!')
        
        for node in path_as:
            world_x, world_y = self.pixel_to_world(int(node.split(',')[1]), int(node.split(',')[0]))
            pose = self.build_a_pose(world_x, world_y)
            path.poses.append(pose)

        end_pose_ = self.build_a_pose(end_pose.pose.position.x, end_pose.pose.position.y)
        path.poses.append(end_pose_)

        return path
    def local_replan(self, current_pose, global_path, lookahead_distance=3.0):
        """
        局部重规划：仅重新规划机器人周围受影响的区域
        
        Args:
            current_pose: 当前机器人位置
            global_path: 全局路径
            lookahead_distance: 局部规划的前瞻距离（米）
        
        Returns:
            融合后的新路径，如果局部规划失败则返回原路径
        """
        if not global_path or not global_path.poses:
            return global_path
        
        # 1. 找到当前位置在全局路径中的索引
        current_x = current_pose.pose.position.x
        current_y = current_pose.pose.position.y
        
        min_dist = float('inf')
        current_idx = 0
        for i, pose in enumerate(global_path.poses):
            dx = pose.pose.position.x - current_x
            dy = pose.pose.position.y - current_y
            dist = np.hypot(dx, dy)
            if dist < min_dist:
                min_dist = dist
                current_idx = i
        
        # 2. 找到前瞻距离内的路径终点
        lookahead_idx = current_idx
        accumulated_dist = 0.0
        for i in range(current_idx, len(global_path.poses) - 1):
            p1 = global_path.poses[i]
            p2 = global_path.poses[i + 1]
            dx = p2.pose.position.x - p1.pose.position.x
            dy = p2.pose.position.y - p1.pose.position.y
            segment_dist = np.hypot(dx, dy)
            accumulated_dist += segment_dist
            
            if accumulated_dist >= lookahead_distance:
                lookahead_idx = i + 1
                break
        else:
            lookahead_idx = len(global_path.poses) - 1
        
        # 3. 检查局部路径段是否有障碍
        has_obstacle = False
        for i in range(current_idx, min(lookahead_idx + 1, len(global_path.poses))):
            pose = global_path.poses[i]
            px, py = self.world_to_pixel(pose.pose.position.x, pose.pose.position.y, clamp=True)
            
            # # 检查该点及其周围是否有动态障碍
            # check_radius = 1  # 检查半径
            # for dy in range(-check_radius, check_radius + 1):
            #     for dx in range(-check_radius, check_radius + 1):
            #         check_y = py + dy
            #         check_x = px + dx
            #         if (0 <= check_y < self.dynamic_costmap.shape[0] and 
            #             0 <= check_x < self.dynamic_costmap.shape[1]):
            #             if self.dynamic_costmap[check_y, check_x] != 0:
            #                 has_obstacle = True
            #                 break
            #     if has_obstacle:
            #         break
            # if has_obstacle:
            #     break
            if self.dynamic_costmap[py, px] != 0:
                has_obstacle = True
                break
        
        # 4. 如果没有障碍，直接返回原路径
        if not has_obstacle:
            self.node.get_logger().info('Local path clear, no replanning needed')
            return global_path
        
        # 5. 执行局部重规划
        self.node.get_logger().info(f'Obstacle detected, local replanning from idx {current_idx} to {lookahead_idx}')
        
        local_start_pose = self.build_a_pose(current_x, current_y)
        local_end_pose = global_path.poses[lookahead_idx]
        
        # 局部A*规划
        local_path = self.find(local_start_pose, local_end_pose)
        
        # 6. 融合路径：current -> local_end + 保留后续全局路径
        if not local_path or len(local_path.poses) < 2:
            self.node.get_logger().warn('Local replanning failed, keeping original path')
            return global_path
        
        # 构建新路径
        new_path = Path()
        new_path.header = global_path.header
        new_path.header.stamp = self.node.get_clock().now().to_msg()
        
        # 添加局部规划的路径（去掉最后一个点，避免重复）
        new_path.poses.extend(local_path.poses[:-1])
        
        # 添加全局路径中lookahead之后的部分
        new_path.poses.extend(global_path.poses[lookahead_idx:])
        
        self.node.get_logger().info(f'Local replan success: {len(local_path.poses)} local + {len(global_path.poses)-lookahead_idx} global waypoints')
        
        return new_path


    def find_nearest_free_cell(self, start_ij, max_radius=10):
        """
        从 start_ij=(i,j) 出发，搜索最近的 free cell
        使用 BFS，保证最近
        """
        h, w = self.dynamic_costmap.shape
        si, sj = start_ij

        # 如果自己就是 free，直接返回
        if self.dynamic_costmap[si, sj] == 0 and self.mp.inf_map_img_array[si, sj] == 0:
            return si, sj

        visited = set()
        q = deque()
        q.append((si, sj, 0))
        visited.add((si, sj))

        while q:
            i, j, d = q.popleft()
            if d > max_radius:
                break

            for di, dj in [(-1,0),(1,0),(0,-1),(0,1),
                        (-1,-1),(-1,1),(1,-1),(1,1)]:
                ni, nj = i + di, j + dj
                if not (0 <= ni < h and 0 <= nj < w):
                    continue
                if (ni, nj) in visited:
                    continue

                # 判定：静态 free + 动态 free
                if self.dynamic_costmap[ni, nj] == 0 and self.mp.inf_map_img_array[ni, nj] == 0:
                    return ni, nj

                visited.add((ni, nj))
                q.append((ni, nj, d + 1))

        return None  # 找不到


def euler_from_quaternion(quaternion):
    x, y, z, w = quaternion
    #  Roll 
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)

    # Pitch 
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)  # 防止数值误差超出 [-1, 1]
    pitch = np.arcsin(t2)

    # Yaw 
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)

    return roll, pitch, yaw
def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    """
    将 quaternion 转成 3x3 旋转矩阵
    不依赖 tf_transformations
    """
    # 归一化（防止数值问题）
    norm = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm

    R = np.array([
        [1 - 2*(qy*qy + qz*qz),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),     1 - 2*(qx*qx + qz*qz),     2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),         2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)]
    ])
    return R

def add_hit_with_inflation(conf, static_free, cy, cx, hit_inc, r, add_inc):
    """
    在 conf 上对 (cy,cx) 命中做累加，并对半径 r 做膨胀累加。
    static_free: 静态free掩码（True 表示可通行）
    """
    h, w = conf.shape

    def sat_add(y, x, inc):
        v = int(conf[y, x]) + int(inc)
        conf[y, x] = 255 if v > 255 else v

    # 中心
    if 0 <= cy < h and 0 <= cx < w and static_free[cy, cx]:
        sat_add(cy, cx, hit_inc)

    # 膨胀圈（简单方形，你也可以换成圆形条件 dx*dx+dy*dy<=r*r）
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            ny, nx = cy + dy, cx + dx
            if not (0 <= ny < h and 0 <= nx < w):
                continue
            if not static_free[ny, nx]:
                continue
            if dy == 0 and dx == 0:
                continue
            # 可选：用圆形膨胀
            # if dx*dx + dy*dy > r*r: continue
            sat_add(ny, nx, add_inc)

class Navigation(rclpy_node):
    """! Navigation node class.
    This class should serve as a template to implement the path planning and
    path follower components to move the turtlebot from position A to B.
    """

    def __init__(self, node_name='task2'):
        """! Class constructor.
        @param  None.
        @return An instance of the Navigation class.
        """
        super().__init__(node_name)
        # Path planner/follower related variables
        self.path = Path()
        self.goal_pose = PoseStamped()
        self.ttbot_pose = PoseStamped()
        self.start_time = 0.0

        # Subscribers
        self.create_subscription(PoseStamped, '/move_base_simple/goal', self.__goal_pose_cbk, 10)
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.__ttbot_pose_cbk, 10)

        self.create_subscription(LaserScan, '/scan', self.__laser_cbk, 10)

        # Publishers
        self.path_pub = self.create_publisher(Path, 'global_plan', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.calc_time_pub = self.create_publisher(Float32, 'astar_time',10) #DO NOT MODIFY
        self.costmap_marker_pub = self.create_publisher(MarkerArray, 'dynamic_costmap_markers', 1)
        # Node rate
        self.rate = self.create_rate(10)

        #Load the map
        map_name = 'sync_classroom_map' #sim: 'sync_classroom_map'
        self.PathFinder = PathFinder(self, map_name)
        
        #Implement the PID controller
        # self.speed_pid = PID(0.3, 0.001, 0.0005, 0.35, 0.2, 0.15)
        # self.heading_pid = PID(0.3, 0, 0, 0.1, 0.2, 0.03)
        self.speed_pid = PID(
            kp=0.5,      # 提高比例增益
            ki=0.002,    # 稍微增大积分
            kd=0.001,    # 稍微增大微分
            setpoint=0.35,
            integral_separation_threshold=0.2,
            deadband=0.1  # 减小死区，更早开始移动
        )
        
        self.heading_pid = PID(
            kp=0.6,      # 提高转向响应
            ki=0.01,     # 加入小的积分项
            kd=0.02,     # 加入微分项减少震荡
            setpoint=0.1,
            integral_separation_threshold=0.2,
            deadband=0.02
        )
        
        self.is_goal_set = False

        self._current_path = None
        self._last_plan_time = None
        self.replan_interval =4
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.map_frame = "odom"
# 常见 scan frame: "base_scan" / "laser" / msg.header.frame_id
    def __goal_pose_cbk(self, data):
        """! Callback to catch the goal pose.
        @param  data    PoseStamped object from RVIZ.
        @return None.
        """
        self.goal_pose = data
        self.get_logger().info('goal_pose: {:.4f}, {:.4f}'.format(self.goal_pose.pose.position.x, self.goal_pose.pose.position.y))
        self.is_goal_set = True

    def __ttbot_pose_cbk(self, data):
        """! Callback to catch the position of the vehicle.
        @param  data    PoseWithCovarianceStamped object from amcl.
        @return None.
        """
        self.ttbot_pose = data.pose
        #self.get_logger().info('ttbot_pose: {:.4f}, {:.4f}'.format(self.ttbot_pose.pose.position.x, self.ttbot_pose.pose.position.y))

    def __laser_cbk(self, msg: LaserScan):
        if self.ttbot_pose is None:
            return

        scan_frame = msg.header.frame_id  # 很重要：不要假设
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame,
                msg.header.frame_id,
                rclpy.time.Time(),   # ✅ 用最新可用，而不是 msg.header.stamp
                timeout=Duration(seconds=0.05)
            )

        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}")
            return
        t = tf.transform.translation
        q = tf.transform.rotation
        # 把 TF 转成 4x4 矩阵
        R = quaternion_to_rotation_matrix(q.x, q.y, q.z, q.w)  # 3x3

        T = np.eye(4, dtype=np.float64)                     
        T[0:3, 0:3] = R
        T[0, 3] = float(t.x)
        T[1, 3] = float(t.y)
        T[2, 3] = float(t.z)


        # ---- 多帧确认版本里，你可以在这里先做 conf 衰减 ----
        dyn = self.PathFinder.dynamic_costmap
        conf = self.PathFinder.dynamic_confirm
        static = self.PathFinder.mp.inf_map_img_array

        static_free = (static == 0)
            # 衰减
        dp = self.PathFinder.decay_per_frame
        conf[:] = np.where(conf > dp, conf - dp, 0).astype(np.uint8)

        angle = msg.angle_min
        for r in msg.ranges:
            if np.isinf(r) or np.isnan(r) or r < msg.range_min or r > msg.range_max:
                angle += msg.angle_increment
                continue

            x_s = r * np.cos(angle)
            y_s = r * np.sin(angle)

            p = np.array([x_s, y_s, 0.0, 1.0], dtype=np.float64)
            pw = T @ p
            x_w, y_w = float(pw[0]), float(pw[1])

            pix_x, pix_y = self.PathFinder.world_to_pixel(x_w, y_w, clamp=True)

            # ✅ 只在静态free区域才当作动态障碍候选（避免把墙写进动态层）
            if static_free[pix_y, pix_x]:
                add_hit_with_inflation(
                    conf=conf,
                    static_free=static_free,
                    cy=pix_y,
                    cx=pix_x,
                    hit_inc=self.PathFinder.hit_increment,
                    r=self.PathFinder.inflation_radius,
                    add_inc=self.PathFinder.inflation_add
                )

            angle += msg.angle_increment

        # ---- 4) 阈值化输出 dyn（二值给A*用）----
        dyn[:] = (conf >= self.PathFinder.confirm_frames).astype(np.uint8)

        self.publish_dynamic_costmap_markers()

    def publish_dynamic_costmap_markers(self):
        dyn = self.PathFinder.dynamic_costmap
        h, w = dyn.shape

        ma = MarkerArray()

        # 用一个 CUBE_LIST marker 把所有障碍画出来
        m = Marker()
        m.header.frame_id = 'map'
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'dynamic_costmap'
        m.id = 0
        m.type = Marker.CUBE_LIST  # 一堆小方块
        m.action = Marker.ADD

        # 尺寸：用地图分辨率
        res = float(self.PathFinder.mp.map.map_df.resolution[0])
        m.scale.x = res
        m.scale.y = res
        m.scale.z = 0.02  # 薄薄一层就行

        # 颜色：红色，半透明一点
        m.color.r = 1.0
        m.color.g = 0.0
        m.color.b = 0.0
        m.color.a = 0.6

        # 用你自己的 pixel_to_world 来处理坐标，这里 Y 翻转自然跟你当前程序一致
        for i in range(h):       # i = row = 像素 y
            for j in range(w):   # j = col = 像素 x
                if dyn[i, j] == 0:
                    continue

                world_x, world_y = self.PathFinder.pixel_to_world(j, i)

                p = Point()
                p.x = float(world_x)
                p.y = float(world_y)
                p.z = 0.01  # 稍微抬一点
                m.points.append(p)

        ma.markers.append(m)
        self.costmap_marker_pub.publish(ma)
    def a_star_path_planner(self, start_pose, end_pose):
        """! A Start path planner.
        @param  start_pose    PoseStamped object containing the start of the path to be created.
        @param  end_pose      PoseStamped object containing the end of the path to be created.
        @return path          Path object containing the sequence of waypoints of the created path.
        """
        #path = Path()
        self.get_logger().info(
            'A* planner.\n> start: {},\n> end: {}'.format(start_pose.pose.position, end_pose.pose.position))
        self.start_time = self.get_clock().now().nanoseconds*1e-9 #Do not edit this line (required for autograder)
        # TODO: IMPLEMENTATION OF THE A* ALGORITHM
        # path.poses.append(start_pose)
        # path.poses.append(end_pose)
        path = self.PathFinder.find(start_pose, end_pose)

        # Do not edit below (required for autograder)
        self.astarTime = Float32()
        self.astarTime.data = float(self.get_clock().now().nanoseconds*1e-9-self.start_time)
        self.calc_time_pub.publish(self.astarTime)
        self.path_pub.publish(path)
        return path

    def _pose_xy(self, pose):
        return np.array([pose.pose.position.x, pose.pose.position.y], dtype=np.float64)

    def _project_point_to_segment(self, p, a, b):
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom < 1e-12:
            return a, 0.0
        t = float(np.dot(p - a, ab) / denom)
        t = max(0.0, min(1.0, t))
        proj = a + t * ab
        return proj, t

    def _turn_angle_at(self, path, k):
        """
        返回路径在 k 处的转角（弧度），k 是 waypoint 索引
        用 (k-1)->k 和 k->(k+1) 的夹角衡量拐角程度
        """
        n = len(path.poses)
        if k <= 0 or k >= n - 1:
            return 0.0
        p0 = self._pose_xy(path.poses[k-1])
        p1 = self._pose_xy(path.poses[k])
        p2 = self._pose_xy(path.poses[k+1])
        v1 = p1 - p0
        v2 = p2 - p1
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 < 1e-9 or n2 < 1e-9:
            return 0.0
        c = float(np.dot(v1, v2) / (n1 * n2))
        c = max(-1.0, min(1.0, c))
        return float(np.arccos(c))  # 0=直线，越大越拐

    def _segment_hits_obstacle(self, p_world, g_world, step=0.05):
        """
        检查 world 直线段 p->g 是否穿过障碍（静态膨胀层或动态层）
        step: 世界坐标采样间隔（米），建议 0.03~0.08
        """
        # 依赖：PathFinder.world_to_pixel、PathFinder.mp.inf_map_img_array、PathFinder.dynamic_costmap
        p = np.array(p_world, dtype=np.float64)
        g = np.array(g_world, dtype=np.float64)
        d = g - p
        L = float(np.linalg.norm(d))
        if L < 1e-6:
            return False

        static = self.PathFinder.mp.inf_map_img_array  # 0=free, >0=膨胀障碍
        dyn = self.PathFinder.dynamic_costmap          # 1=动态障碍

        steps = max(2, int(L / step))
        for i in range(steps + 1):
            t = i / steps
            x = float(p[0] + d[0] * t)
            y = float(p[1] + d[1] * t)
            # world->grid（你这里返回的是 (ix,iy)，注意你后面用 pix_y,pix_x 访问数组）
            ix, iy = self.PathFinder.world_to_pixel(x, y, clamp=True)
            # 数组索引是 [row=y, col=x]
            if static[iy, ix] != 0:
                return True
            if dyn[iy, ix] != 0:
                return True

        return False
    def _reset_tracker_on_new_path(self, path):
    # 用一个轻量的“路径签名”判断是否真变了
        n = len(path.poses)
        if n < 2:
            self._path_sig = None
            return

        def xy(i):
            return (round(path.poses[i].pose.position.x, 3),
                    round(path.poses[i].pose.position.y, 3))

        sig = (n, xy(0), xy(min(5, n-1)), xy(n-1))

        if getattr(self, "_path_sig", None) != sig:
            self._path_sig = sig
            # ✅ 重置跟踪内部状态：让 get_path_idx 重新从机器人附近找段
            self._path_progress_seg = 0
            self._lookahead = 0.6
            self._force_global_search = 8   # 接下来 8 次循环用更稳的全局搜索


    def get_path_idx(self, path, vehicle_pose):
        if path is None or len(path.poses) < 2:
            return 0

        n = len(path.poses)

        # ---- 可调参数（控制层）----
        MIN_LH = 0.25
        MAX_LH = 1.20
        GROW = 1.15
        SHRINK = 0.60
        SAFE_STEP = 0.05

        BIG_TURN = np.deg2rad(35)
        SMALL_TURN = np.deg2rad(10)

        # 维护可变 lookahead
        if not hasattr(self, "_lookahead"):
            self._lookahead = 0.6
        lh = float(np.clip(self._lookahead, MIN_LH, MAX_LH))

        # 机器人位置
        p = np.array([vehicle_pose.pose.position.x,
                    vehicle_pose.pose.position.y], dtype=np.float64)

        # ✅ 永远先定义 last_seg，避免 force 分支未定义
        last_seg = getattr(self, "_path_progress_seg", 0)
        last_seg = max(0, min(last_seg, n - 2))

        # 选搜索窗口
        force = getattr(self, "_force_global_search", 0)
        if force > 0:
            seg_start, seg_end = 0, n - 2
            self._force_global_search = force - 1
        else:
            back, ahead = 3, 40
            seg_start = max(0, last_seg - back)
            seg_end   = min(n - 2, last_seg + ahead)

        # ---- 1) 找最近投影段 ----
        best_seg = seg_start
        best_proj = None
        best_d2 = float("inf")

        for i in range(seg_start, seg_end + 1):
            a = self._pose_xy(path.poses[i])
            b = self._pose_xy(path.poses[i+1])
            proj, _ = self._project_point_to_segment(p, a, b)
            d2 = float(np.dot(p - proj, p - proj))
            if d2 < best_d2:
                best_d2 = d2
                best_seg = i
                best_proj = proj

        # 极端保护（理论不该发生）
        if best_proj is None:
            self._path_progress_seg = 0
            return 0

        # 单调进度（最多允许回退1段）
        best_seg = max(best_seg, last_seg - 1)
        self._path_progress_seg = best_seg

        # ---- 2) 转角自适应 ----
        k = min(n - 2, best_seg + 1)
        turn = self._turn_angle_at(path, k)

        if turn < SMALL_TURN:
            lh = min(MAX_LH, lh * GROW)
        elif turn > BIG_TURN:
            lh = max(MIN_LH, lh * SHRINK)

        # ---- 3) 从投影点沿路径前进 lookahead ----
        def advance_idx_from(seg, proj_point, lookahead):
            remaining = float(lookahead)

            b = self._pose_xy(path.poses[seg+1])
            dist_to_b = float(np.linalg.norm(b - proj_point))
            if remaining <= dist_to_b:
                target = proj_point + (b - proj_point) * (remaining / max(dist_to_b, 1e-9))
                return seg + 1, target

            remaining -= dist_to_b
            for j in range(seg + 1, n - 1):
                p1 = self._pose_xy(path.poses[j])
                p2 = self._pose_xy(path.poses[j+1])
                seglen = float(np.linalg.norm(p2 - p1))
                if seglen < 1e-9:
                    continue
                if remaining <= seglen:
                    t = remaining / seglen
                    target = p1 + (p2 - p1) * t
                    return j + 1, target
                remaining -= seglen

            return n - 1, self._pose_xy(path.poses[-1])

        idx, target_xy = advance_idx_from(best_seg, best_proj, lh)

        # ---- 4) 直线段碰撞检查：不安全就缩短 lookahead ----
        robot_xy = p
        while lh > MIN_LH:
            if not self._segment_hits_obstacle(robot_xy, target_xy, step=SAFE_STEP):
                break
            lh = max(MIN_LH, lh * SHRINK)
            idx, target_xy = advance_idx_from(best_seg, best_proj, lh)

        self._lookahead = lh

        # 末端锁定
        if idx >= n - 3:
            return n - 1

        return idx


    def path_follower(self, vehicle_pose, current_goal_pose, max_speed = 0.5):
        """! Path follower.
        @param  vehicle_pose           PoseStamped object containing the current vehicle pose.
        @param  current_goal_pose      PoseStamped object containing the current target from the created path. This is different from the global target.
        @return path                   Path object containing the sequence of waypoints of the created path.
        """
        speed = 0.0
        heading = 0.0
        # TODO: IMPLEMENT PATH FOLLOWER

        distance = np.sqrt((current_goal_pose.pose.position.x-vehicle_pose.pose.position.x)**2
                           + (current_goal_pose.pose.position.y-vehicle_pose.pose.position.y)**2)
        speed = np.clip(-self.speed_pid.compute(distance,0.0), 0.0, max_speed)
        
        target_yaw = np.arctan2(current_goal_pose.pose.position.y-vehicle_pose.pose.position.y,
                                current_goal_pose.pose.position.x-vehicle_pose.pose.position.x)
        current_yaw = euler_from_quaternion([vehicle_pose.pose.orientation.x, vehicle_pose.pose.orientation.y,
                                             vehicle_pose.pose.orientation.z, vehicle_pose.pose.orientation.w])[2]
        angle_diff = target_yaw - current_yaw
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi # Normalize angle to [-pi, pi]
        heading = np.clip(self.heading_pid.compute(current_yaw, current_yaw + angle_diff), -0.5, 0.5)

        return speed, heading

    def move_ttbot(self, speed, heading):
        """! Function to move turtlebot passing directly a heading angle and the speed.
        @param  speed     Desired speed.
        @param  heading   Desired yaw angle.
        @return path      object containing the sequence of waypoints of the created path.
        """
        cmd_vel = Twist()
        # TODO: IMPLEMENT YOUR LOW-LEVEL CONTROLLER
        # Turn first
        if abs(heading) > np.pi/30:
            cmd_vel.angular.z = heading
            cmd_vel.linear.x = 0.0
        else:
            cmd_vel.angular.z = heading
            cmd_vel.linear.x = speed

        self.cmd_vel_pub.publish(cmd_vel)

    def _loop_once(self):
        now = self.get_clock().now()

        # 1. 收到新的 goal：从当前位置到新 goal 规划一次
        if self.is_goal_set:
            self.is_goal_set = False
            path = self.a_star_path_planner(self.ttbot_pose, self.goal_pose)
            self._current_path = path
            self._reset_tracker_on_new_path(path)
            self._last_plan_time = now

        # 2. 没有 goal 就啥也不干
        if not hasattr(self, "goal_pose") or self.goal_pose is None:
            return

        # 3. 如果已经到 goal 附近了，就停下
        goal_dx = self.goal_pose.pose.position.x - self.ttbot_pose.pose.position.x
        goal_dy = self.goal_pose.pose.position.y - self.ttbot_pose.pose.position.y
        goal_dist = np.hypot(goal_dx, goal_dy)
        if goal_dist < 0.3:  # 10cm 以内认为到达
            self.get_logger().info('Goal reached! Distance: {:.3f}m'.format(goal_dist))
            self.move_ttbot(0.0, 0.0)
            self.goal_pose = None
            self._current_path = None   
            return

        # 4. 周期性重规划（利用最新 amcl_pose + costmap）
        if self._last_plan_time is None:
            need_replan = True
        else:
            dt = (now.nanoseconds - self._last_plan_time.nanoseconds) * 1e-9
            need_replan = dt > self.replan_interval

        if need_replan:
            self.move_ttbot(0.0, 0.0) # stop before replan  
            try:
                path = self.PathFinder.local_replan(
                    self.ttbot_pose, 
                    self._current_path,
                    lookahead_distance=2
                )
                if(len(path.poses) <= 2): ## 非常粗糙的修改，后续最好改为异常处理
                    self.get_logger().info('No path found to the goal! Replanning in 2 seconds.')
                    self._last_plan_time = now - Duration(seconds=2.0) 
                else:
                    self._current_path = path
                    self._reset_tracker_on_new_path(path)
                    self._last_plan_time = now
                    self.path_pub.publish(path)
            
            except Exception as e:
                self.get_logger().info('Exception occurred while planning path: {}'.format(e))
                self._last_plan_time = now - Duration(seconds=2.0)
            # 执行局部重规划
            
            

        # 5. 按当前路径跟踪
        path = self._current_path
        if path and path.poses:
            idx = self.get_path_idx(path, self.ttbot_pose)
            current_goal = path.poses[idx]
            speed, heading = self.path_follower(self.ttbot_pose, current_goal)
            self.move_ttbot(speed, heading)

    def run(self):
        """! Main loop of the node. You need to wait until a new pose is published, create a path and then
        drive the vehicle towards the final pose.
        @param none
        @return none
        """
        self.timer = self.create_timer(0.05, self._loop_once)
        rclpy.spin(self)


def main(args=None):
    rclpy.init(args=args)
    nav = Navigation(node_name='task2')

    try:
        nav.run()
    except KeyboardInterrupt:
        pass
    finally:
        nav.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()