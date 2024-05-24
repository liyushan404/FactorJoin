# 打开txt文档
with open('/home/lrr/Documents/FactorJoin/checkpoints/postgres_sub_stats.txt', 'r') as file:
    # 逐行读取文档内容
    lines = file.readlines()

# 初始化总规划时间的变量
total_planning_time = 0

# 循环遍历每一行
for line in lines:
    # 如果行包含'planning_time'字段
    if 'planning_time' in line:
        # 解析行中的浮点数并加到总规划时间变量上
        total_planning_time += float(line.split()[-6])
        # print(float(line.split()[-6]))

# 打印总规划时间
print('总规划时间为:', total_planning_time)
