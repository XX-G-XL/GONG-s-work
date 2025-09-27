#The below import are all that should be used in this assignment. Additional libraries are not allowed.
import numpy as np 
import math 
import scipy.sparse.csgraph 
import matplotlib.pyplot as plt 
import random 
import argparse
import collections
import sys
import csv

'''
==============================
The code below here is for your occupancy grid solution
==============================
'''

def read_map_from_file(filename):
    '''
    This functions reads a csv file describing a map and returns the map data
    Inputs:
        - filename (string): name of the file to read
    Outputs:
        - map (tuple): A map is a tuple of the form (grid_size, start_pos, goal_pos, [obstacles])
            grid_size is an tuple (length, height) representing the size of the map
            start_pos is a tuple (x, y) representing the x,y coordinates of the start position
            goal_pos is a tuple (x, y) representing the x,y coordinate of the goal position
            obstacles is a list of tuples. Each tuple represents a single  circular obstacle and is of the form (x, y, radius).
                x is an integer representing the x coordinate of the obstacle
                y is an integer representing the y coordinate of the obstacle
                radius is an integer representing the radius of the obstacle
    '''

    #Your code goes here
    with open(filename, 'r') as file:# 'r'只读模式,with 确保文件操作后关闭
        txt = file.read()#从文件中读取整个文件内容为一个字符串
    lines = txt.split('\n')  #用字符串的split（）方法，将txt的字符串用\n分割成行，每一行文本成为列表中一个元素
    # lines=txt.split('\n')按换行符分割文件内容，得到每行作为一个列表的元素
    # 读取网格大小、起点和终点坐标
    x_dimension, y_dimension = map(int, lines[0].split(','))# 读取文件第一行,为网格的宽度和高度
    start_x, start_y = map(int, lines[1].split(','))# 读取第二行为起点坐标
    goal_x, goal_y = map(int, lines[2].split(','))# 读取第三行为目标点坐标
    # 读取障碍物信息
    obstacles = []#创建一个空列表，用于存放障碍物的信息
    for line in lines[3:]:# 跳过前三行从第四行开始（第四行后为障碍物的信息）
        if not line.strip():# 检查是否为空
            continue
        x, y, r = map(int, line.split(','))# 将三个信息用逗号隔开并提取
        # map(int, ...) 将字符串解析为整数的快捷方式。
        obstacles.append((x, y, r))# 装入obstacles列表
        # obstacles 是存放所有障碍物 (x, y, r) 的列表，用于记录场景中所有障碍物信息
    return (x_dimension, y_dimension), (start_x, start_y), (goal_x, goal_y), obstacles

def is_intersect(rect_left, rect_top, rect_right, rect_bottom, circle_x, circle_y, circle_radius):
    '''
    该函数判断一个矩形区域是否与圆形障碍物相交。
    输入：
        - rect_left, rect_top, rect_right, rect_bottom: 矩形的左、上、右、下边界坐标
        - circle_x, circle_y, circle_radius: 圆心坐标及半径
    输出：
        - True: 如果矩形与圆形相交
        - False: 如果矩形不与圆形相交
    '''
    # 确定矩形的四个边界
    x = rect_left# 矩阵左边界的x坐标
    while x <= rect_right:# 矩阵终止点为矩阵右边界
        y = rect_top# 矩阵上边界y坐标
        while y <= rect_bottom:# 终止点为矩阵下边界
            # 判断当前点（x，y）是否在圆的范围内
            if (x - circle_x) ** 2 + (y - circle_y) ** 2 <= circle_radius ** 2:
                # 圆的标准方程，当距离的平方小于等于半径的平方时，点（x，y）在圆内
                return True# 如果找到相交点，，返回ture，说明圆与矩形相交
            y += 0.1# 步长为0.1
        x += 0.1# 每0.1步长进行一次检查
    return False

def make_occupancy_grid_from_map(map_data, cell_size=5):
    '''
    This function takes a map and a cell size (the physical size of one "cell" in the grid) and returns a 2D numpy array, 
    with each cell containing a '1' if it is occupied and '0' if it is empty
    Inputs: map (tuple) - see read_map_from_file for description.
    Outputs: occupancy_grid - 2D numpy array
    '''

    #Your code goes here
    grid_size, _, _, obstacles = map_data# 只提取网格尺寸和障碍物信息

    # 计算单元格的宽度和高度
    grid_width = math.ceil(grid_size[0] / cell_size)
    # cell_size表示每个单元格边长，计算横向的单元格数量，`math.ceil` 确保向上取整
    grid_height = math.ceil(grid_size[1] / cell_size)# 计算纵向单元格数量
    # 创建一个二维数组 `occupancy_grid'，大小为横竖单元格数量相乘，初始化占用网格，所有单元格默认设为 0（空闲）
    occupancy_grid = np.zeros((grid_height, grid_width), dtype=int)

    # 遍历每个单元格，检查是否被障碍物占据
    for row in range(grid_height):# 从上到下逐行经过单元格
        for col in range(grid_width):# 从左到右逐行经过单元格
            rect_left = col * cell_size# 左边顶点坐标，col为从左到右的顺序
            rect_top = row * cell_size# 顶部顶点坐标，row为从上到下的顺序
            rect_right = (col + 1) * cell_size# 右边顶点坐标
            rect_bottom = (row + 1) * cell_size# 底部顶点坐标
            intersect = False
            # 遍历所有的障碍物，检查是否与当前单元格相交
            for cx, cy, cr in obstacles:# cx, cy, cr` 是障碍物的圆心x、y坐标和半径。
                # 使用 is_intersect 函数检查该单元格与障碍物是否相交
                if is_intersect(rect_left, rect_top, rect_right, rect_bottom, cx, cy, cr):
                    #is_intersect函数判断圆心到矩形最近距离，如果小于原半径则为相交
                    intersect = True# ture表示有相交
            if intersect:
                occupancy_grid[row][col] = 1# 表示已经被障碍物占用

    return occupancy_grid


def make_adjacency_matrix_from_occupancy_grid(occupancy_grid):
    '''
    This function converts an occupancy grid into an adjacency matrix. We assume that cells are connected to their neighbours unless the neighbour is occupied. 
    We also assume that the cost of moving from a cell to a neighbour is always '1' and allow only horizontal and vertical connections (i.e. no diagonals allowed).
    Inputs: occupancy_grid - a 2D (NxN) numpy array. An element with value '1' is occupied, while those with value '0' are empty.
    Outputs: A 2D (MxM where M=NxN) array. Element (i,j) contains the cost of travelling from node i to node j in the occupancy grid. 
    '''

    #Your code goes here
    # 定义四个可能的移动方向：上、下、左、右对应位移（dy，dx）
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # 通过occupancy_grid.shape获取占用网格的高度（行数）和宽度（列数）
    height, width = occupancy_grid.shape #通过占用单元格的形状，提取行数和列数
    num_cells = height * width# 计算总体单元格数量
    # 构建大小为`(num_cells, num_cells)` 的二维数组adjacency_matrix，初始情况下，所有网格单元都设为初始值 0，表示它们是空闲状态
    adjacency_matrix = np.zeros((num_cells, num_cells), dtype=int)
    # 该矩阵用于记录两个单元格之间的连通关系

    # 遍历占用网格的每个单元格
    for y in range(height):
        for x in range(width):# 使用嵌套循环，逐行逐列经过每个单元格（y，x）
            if occupancy_grid[y, x] == 1: # 如果当前单元格被占用，则跳过
                continue

            # 将二维网格坐标 (y, x) 转换为一维数组的线性索引 current
            current = y * width + x# y行乘上总列数（前y行所有的单元格），加上最后一行的第x个

            # 遍历四个方向的相邻单元格
            for dx, dy in directions:# directions是前文的上下左右的位移量
                nx, ny = x + dx, y + dy# 当前行加上dy垂直方向上的偏移量，当前列加上dx水平方向上的位移量
                # 确保邻居单元格在网格范围内，并且没有被占用
                if 0 <= nx < width and 0 <= ny < height and occupancy_grid[ny, nx] == 0:# 确保相邻单元格是否在网格范围内，同时检查是否是空闲状态，是否可通行
                    neighbor = ny * width + nx# 计算相邻单元格的线性索引
                    # 设定在二维数组adjacency_matrix中，从当前单元格到邻居单元格的步数为 1
                    adjacency_matrix[current, neighbor] = 1

    return adjacency_matrix

def get_path_from_predecessors(predecessors, map_data, cell_size=5):
    '''
    This function takes a predecessors matrix, map_data and cell_size as input and returns the path from start to goal position. 
    We take the mid-point of each cell as the (x, y) coordinate for the path.
    Inputs: predecessors - a 2D numpy array (size = M = NxN, where N is the length of an occupancy grid) produced by scipy's implementation of Dijkstra's algorithm.
            Each element i tells us the index of the node we should travel to if we are in node j. 
            map_data -  (tuple) see read_map_from_file for description.
            cell_size - (integer) the physical size corresponding to a single cell in the grid.
    Outputs: path - A list of tuples (x, y), where (x, y) are the coordinates of a position we can travel to in the map. 
    '''

    #Your code goes here
# 将map——data里无量纲的数据变成了实际的位置坐标。从终点索引回溯到起点索引，生成路径的实际坐标序列
    # 从 map_data 中获取网格大小、起点坐标、终点坐标和障碍物信息
    grid_size, start_pos, goal_pos, obstacles = map_data
    grid_width = math.ceil(grid_size[0] / cell_size)# 根据单元格大小计算网格宽度，math.ceil向上取整确保覆盖
    #grid_size[0] 是地图的宽度，cell_size 是单元格的边长

    # 计算起点在网格中的索引
    start_index = (start_pos[1] // cell_size) * grid_width + (start_pos[0] // cell_size)
    # 计算终点在网格中的索引
    goal_index = (goal_pos[1] // cell_size) * grid_width + (goal_pos[0] // cell_size)#‘//’表示整除
    # index = 行号 * 网格宽度 + 列号
    # 行号 = (y 坐标 // cell_size)
    # 列号 = (x 坐标 // cell_size)

    path = []# 用于储存终点回溯到起点的路径
    current = goal_index# 初始化索引 goal_index，表示从终点开始回溯
    # 通过前驱矩阵回溯路径，直到回溯到起点或遇到 -9999（表示无效索引）
    while current != start_index and current != -9999:
        # 计算当前索引对应的实际坐标（取单元格中心点）
        xx = (current % grid_width) * cell_size + cell_size // 2# %取余
        yy = (current // grid_width) * cell_size + cell_size // 2
        path.append((xx, yy))  # 将当前坐标加入路径列表
        current = predecessors[start_index, current]# 从 predecessors 中获取上一个节点索引。

    # 由于路径是从终点回溯到起点的，因此需要反转路径
    path.reverse()
    return path

def plot_map(ax, map_data):
    '''
    This function plots a map given a description of the map
    Inputs:
    ax (matplotlib axis) - the axis the map should be drawn on
    map_data - a tuple describing the map. See definition in read_map_from_file function for details.
    '''
    if map_data:
        start_pos = map_data[1]
        goal_pos = map_data[2]
        obstacles = map_data[3]

        ax.plot(goal_pos[0], goal_pos[1], 'r*')  #绘制终点
        ax.plot(start_pos[0], start_pos[1], 'b*')  # 绘制起点

        for obstacle in obstacles:
            #Obstacle[0] is x position, [1] is y position and [2] is radius
            c_patch = plt.Circle((obstacle[0], obstacle[1]), obstacle[2], color='red')
            ax.add_patch(c_patch) # 绘制圆形障碍物
    else:
        print("No map data provided- have you implemented read_map_from_file?")

def plot_path(ax, path):
    '''
    This function plots the path found by your occupancy grid solution.
    Inputs: ax (matplotlib axis) - the axis object where the path will be drawn
            path (list of tuples) - a list of points (x, y) representing the spatial co-ordinates of a path.
    '''
    if path:
        xs, ys = zip(*path)# path 的作用是将 path 中每个点（如 (x, y)）拆解为单独的坐标。zip(*path) 将点的集合进行解包并转置，
        ax.plot(xs, ys, 'b-')  # 画出路径

def test_make_occupancy_grid():
    
    map0 = ((10, 10), (1, 1), (9, 9), [])
    assert np.array_equal(make_occupancy_grid_from_map(map0, cell_size=1), np.zeros((10, 10))), "Test 1 - checking map 0 with cell size 10"
    
    map1 = ((10, 10), (1, 1), (9, 9), [(5, 5, 2)])
    assert np.array_equal(make_occupancy_grid_from_map(map1, cell_size=10), np.array([[1]])), "Test 1 - checking map 1 with cell size 10"
    assert np.array_equal(make_occupancy_grid_from_map(map1, cell_size=5), np.array([[1, 1], [1, 1]])), "Test 2 - checking map 1 with cell size 5"

    map1_cell_size_2_answer = np.array([[0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0]])
    assert np.array_equal(make_occupancy_grid_from_map(map1, cell_size=2), map1_cell_size_2_answer), "Test 3 - checking map 1 with cell size 2"

    map2 = ((100, 100), (1, 1), (9, 9), [(10, 10, 5), (90, 90, 5)])
    map2_answer = np.array([[1, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1]])

    occupancy_grid1 = np.array([[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]])

    adjacency_matrix1 = np.array([[0., 1., 0., 1., 0., 0., 0., 0., 0.],
        [1., 0., 1., 0., 1., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 1., 0., 0., 0.],
        [1., 0., 0., 0., 1., 0., 1., 0., 0.],
        [0., 1., 0., 1., 0., 1., 0., 1., 0.],
        [0., 0., 1., 0., 1., 0., 0., 0., 1.],
        [0., 0., 0., 1., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 1., 0., 1., 0., 1.],
        [0., 0., 0., 0., 0., 1., 0., 1., 0.]])

    assert np.array_equal(make_adjacency_matrix_from_occupancy_grid(occupancy_grid1), adjacency_matrix1)

    occupancy_grid2 = np.array([[1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]])

    adjacency_matrix2 = np.zeros((occupancy_grid2.size, occupancy_grid2.size))

    assert np.array_equal(make_adjacency_matrix_from_occupancy_grid(occupancy_grid2), adjacency_matrix2)

    occupancy_grid3 = np.array([[0, 1, 0],
        [0, 1, 0],
        [0, 1, 0]])

    adjacency_matrix3 = np.array([[0., 0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 1.],
        [0., 0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0.]])

    assert np.array_equal(make_adjacency_matrix_from_occupancy_grid(occupancy_grid3), adjacency_matrix3)

    test_get_path()

def test_get_path():
    assert len(occupancy_grid('.venv/lib/map2.csv')) == 36# 测试结果长度是否等于36

def test_occupancy_grid():
    test_make_occupancy_grid()

def occupancy_grid(file, cell_size=5):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_axis_off()

    print(file)
    map_data = read_map_from_file(file)
    print(map_data)
    plot_map(ax, map_data)
    grid = make_occupancy_grid_from_map(map_data, cell_size)
    adjacency_matrix = make_adjacency_matrix_from_occupancy_grid(grid)
    #You'll need to edit the line below to use Scipy's shortest graph function to find the path for us
    predecessors = scipy.sparse.csgraph.shortest_path(scipy.sparse.csr_array(adjacency_matrix), directed=False, return_predecessors=True)[1]
    path = get_path_from_predecessors(predecessors, map_data, cell_size)
    plot_path(ax, path)

    plt.show()

    return path

'''
==============================
The code below here is for your RRT solution
==============================
'''


def euclidean_distance(p1, p2):
    # 计算两点间的距离
    """
    :param p1: 点1的坐标 (x, y)
    :param p2: 点2的坐标 (x, y)
    :return: 两点之间的距离
    """
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
#欧几里得距离公式：距离 = √((x2 - x1)^2 + (y2 - y1)^2)

def get_nearest_node(nodes, points):
    """
    在已生成的节点中找到离目标点最近的节点
    :param nodes: 已生成的节点列表
    :param points: 目标点
    :return: 距离目标点最近的节点
    """
    return min(nodes, key=lambda node: euclidean_distance(points, node))
# nodes表示已经存在的节点，points目标点，
# min()用于找到 最小值 元素。通过参数 key 指定一个函数，该函数会应用于 nodes 中的每个元素，其结果值将用于比较。
#lambda 是匿名函数，用法是将每个节点 node 作为输入，计算其与目标点 points 的欧几里得距离
# euclidean_distance(points, node) 来计算点 points 和 node 的欧几里得距离。（与上面公式相等，p1p2换成了points和node）

def is_point_valid(point, obstacles):# 定义一个函数 is_point_valid，用来判断某个点是否位于障碍物范围外。
    """
    判断某个点是否在障碍物内
    :param point: 需要检测的点坐标 (x, y)
    :param obstacles: 障碍物列表，每个障碍物是 (x, y, radius)
    :return: 如果点在障碍物外，返回 True，否则返回 False
    """
    for ox, oy, radius in obstacles:# 经过每个障碍物并提取圆心坐标和半径
        if euclidean_distance(point, (ox, oy)) <= radius:
            # 通过比较圆心和当前点的欧几里得距离与障碍物半径的大小关系，判断点是否在障碍物范围内
            return False
    return True# 如果所有点都在障碍物外，则为ture

def is_line_collision_free(from_node, to_point, obstacles):
    """
    检查从一个点到另一个点的直线路径是否与障碍物发生碰撞
    :param from_node: 起始点 (x, y)
    :param to_point: 目标点 (x, y)
    :param obstacles: 障碍物列表
    :return: 如果路径无碰撞返回 True，否则返回 False
    """
    from_x, from_y = from_node
    to_x, to_y = to_point
    # 将起点和终点坐标赋值

    num_steps = max(abs(to_x - from_x), abs(to_y - from_y))
    # 比较x、y坐标差值，选取其中大的那个
    for i in range(num_steps + 1):# 经过路径上所有的点，i是一个循环变量，从0开始逐步加1
        x = from_x + i * (to_x - from_x) / num_steps
        y = from_y + i * (to_y - from_y) / num_steps
        #计算当前点的x、y坐标
        if not is_point_valid((x, y), obstacles):
            # 检查当前点（x、y）是否与障碍物碰撞
            return False
    return True

def steer(from_node, to_point, step_size):
    #定义了一个函数 steer，目的是从给定的起始点 from_node 向目标点 to_point 以给定的步长 step_size 移动，并返回计算出的新点坐标 (x, y)
    """
    计算从一个节点朝向目标点移动 step_size 的新点
    :param from_node: 起始点 (x, y)
    :param to_point: 目标点 (x, y)
    :param step_size: 迈出的步长
    :return: 计算出的新点 (x, y)
    """
    from_x, from_y = from_node
    to_x, to_y = to_point
    # 分解起始点和目标点的坐标
    direction = np.arctan2(to_y - from_y, to_x - from_x)
    # np.arctan2 函数用于计算两点之间形成的线段与 x 轴正方向之间的夹角（以弧度为单位）
    new_x = int(from_x + step_size * np.cos(direction))
    new_y = int(from_y + step_size * np.sin(direction))
    #p.cos(direction) 和 np.sin(direction) 分别得到方向角的余弦值和正弦值，可以用于分解移动距离到 x 和 y 方向。
    #step_size * np.cos(direction) 表示在 x 轴方向上的移动距离。
    #step_size * np.sin(direction) 表示在 y 轴方向上的移动距离。
    return new_x, new_y

def rrt(map_file, step_size=10, num_points=100):
    """
    使用快速随机树 (RRT) 进行路径规划
    :param map_file: 地图数据文件 (CSV)
    :param step_size: 每次扩展的步长
    :param num_points: 迭代生成的最大点数
    :return: 计算出的路径列表 [(x1, y1), (x2, y2), ...]
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_axis_off()

    # 读取地图数据
    grid_size, start_pos, goal_pos, obstacles = read_map_from_file(map_file)
    plot_map(ax, (grid_size, start_pos, goal_pos, obstacles))

    path = []# 储存起点到终点的路径
    tree = {}# 储存rrt的树结构，新节点由最近节点扩展而来

    # 检查起始点是否有效
    valid = is_point_valid(start_pos, obstacles)

    # 树的扩展过程
    if valid:
        # 存储已生成的点，初始点为起点
        points = [start_pos]
        for _ in range(num_points):
            while True:
                trial_point = (random.randint(0, grid_size[0]), random.randint(0, grid_size[1]))# 随机生成点
                nearest_node = get_nearest_node(points, trial_point)#  找最近节点
                new_point = steer(nearest_node, trial_point, step_size)# 用steer函数，
                # 检测路线是否触碰
                if not is_line_collision_free(nearest_node, new_point, obstacles):
                    continue
                if new_point in tree:
                    continue
                # 添加新点到已生成点集
                points.append(new_point)
                tree[new_point] = nearest_node
                x_coords, y_coords = zip(*[nearest_node, new_point])
                ax.plot(x_coords, y_coords, linestyle='-', color='green')
                ax.plot(trial_point[0], trial_point[1], 'yx')
                break
        # 找到离目标点最近的节点
        nearest_node = get_nearest_node(points, goal_pos)
        # 检查从最近点到目标点是否可行
        if not is_line_collision_free(nearest_node, goal_pos, obstacles):
            valid = False#如果不能无触碰连接，说明规划失败
        else:
            x_coords, y_coords = zip(*[nearest_node, goal_pos])
            ax.plot(x_coords, y_coords, linestyle='-', color='green')
            tree[goal_pos] = nearest_node
    # 如果整体规划有效，就从目标点开始回溯，直到找到起点
    if valid:
        path = [goal_pos]
        while path[0] != start_pos:
            path.insert(0, tree[path[0]])

    # 绘制最终路径
    plot_path(ax, path)

    plt.show()

    return path



def test_rrt():
    #Your test functions / statements go here.
    path = rrt('.venv/lib/map2.csv')
    assert len(path) > 0
    assert path[0] == (5, 5)
    assert path[-1] == (95, 95)

'''
==============================
The code below here is used to read arguments from the terminal, allowing us to run different parts of your code.
You should not need to modify this
==============================
'''

def main():

    parser = argparse.ArgumentParser(description=" Path planning Assignment for CPA 2024/25")
    parser.add_argument('--rrt', action='store_true')
    parser.add_argument('-test_rrt', action='store_true')
    parser.add_argument('--occupancy', action='store_true')
    parser.add_argument('-test_occupancy', action='store_true')
    parser.add_argument('-file')
    parser.add_argument('-cell_size', type=int)

    args = parser.parse_args()

    if args.occupancy:
        if args.file is None:
            print("Error - Occupancy grid requires a map file to be provided as input with -file <filename>")
            exit()
        else:
            if args.cell_size:
                occupancy_grid(args.file, args.cell_size)
            else:
                occupancy_grid(args.file)

    if args.test_occupancy:
        print("Testing occupancy_grid")
        test_occupancy_grid()

    if args.test_rrt:
        print("Testing RRT")
        test_rrt()
    
    if args.rrt:
        if args.file is None:
            print("Error - RRT requires a map file to be provided as input with -file <filename>")
            exit()
        else:
            rrt(args.file)

if __name__ == "__main__":
    main()
    #The test code is not complete.
    # Some cells in the chart are not rendered, which sometimes makes errors hard to detect
    #Missing error handling: the map data returned by read_map_from_file is not properly validated, potentially resulting in unexpected runtime failures

    # The program continues to run even if the file format does not meet the required specifications