import re
import numpy as np
import random
import math
import time  

# LaTeX FTC From-To Chart
# LATEX_MATRIX = """
# \\begin{bmatrix}
#  0 & 1 & 2 & 3 & 4 & 5 & 6 & 7 & 8 & 9 & 10 & \\\\
#  1 & 1 & 0 & 2 & 3 & 0 & 0 & 4 & 0 & 5 & 6 & \\\\
#  2 & 1 & 7 & 3 & 4,5 & 6 & 2 & 0 & 8 & 9 & 10 & \\\\
#  3 & 1 & 0 & 3 & 0 & 4 & 2 & 0 & 5 & 0 & 6 & \\\\
#  4 & 1 & 2 & 0 & 0 & 5,6,7 & 0 & 3 & 4 & 0 & 8 & \\\\
#  5 & 1 & 0 & 2 & 3 & 5,6 & 4 & 7 & 0 & 8 & 9 & \\\\
#  6 & 1 & 0 & 6,7 & 0 & 2,3,4 & 0 & 5 & 0 & 0 & 8 & \\\\
#  7 & 1 & 2,5 & 3 & 7 & 6 & 4 & 0 & 0 & 8 & 9 & \\\\
#  8 & 1 & 4 & 0 & 6 & 5 & 2 & 3 & 0 & 0 & 7 & \\\\
#  9 & 1 & 3,4 & 2 & 0 & 0 & 0 & 0 & 5,6 & 0 & 7 & \\\\
#  10 & 1 & 5 & 3 & 0 & 4 & 2 & 0 & 0 & 6 & 7 & \\\\
#  11 & 1 & 0 & 2,3 & 0 & 4,5 & 0 & 0 & 7,8 & 6 & 9 &
# \\end{bmatrix}
# """

LATEX_MATRIX = """
\\begin{bmatrix}
 0 & 1 & 2 & 3 & 4 & 5 & 6 & 7 & 8 & 9\\
 1 & 1 & 0 & 0 & 4 & 2 & 5 & 6 & 3,7 & 8\\
 2 & 1 & 2 & 0 & 0 & 4 & 3 & 6 & 5 & 7\\
 3 & 1 & 0 & 4 & 0 & 5 & 0 & 2 & 3 & 6\\
 4 & 1 & 0 & 0 & 0 & 3 & 4,5 & 6 & 2 & 7\\
 5 & 1 & 0 & 2,4,6 & 5 & 3 & 0 & 0 & 7 & 8\\
 6 & 1 & 4,5 & 7 & 6 & 0 & 2 & 3 & 0 & 8\\
 7 & 1 & 0 & 2 & 0 & 3 & 0 & 0 & 0 & 4\\
 8 & 1 & 3 & 0 & 0 & 0 & 2 & 4 & 0 & 5\\
 9 & 1 & 0 & 2,5 & 4 & 0 & 0 & 0 & 3 & 6\\
 10 & 1 & 0 & 2,4 & 0 & 0 & 5 & 0 & 3 & 6\\
\\end{bmatrix}
"""

def parse_matrix(latex_matrix) -> list:
    """解析LaTeX矩陣為Python列表，並動態計算部門數量"""
    rows = re.findall(r'(?:[0-9]+(?:\s*&\s*[^\\]+)+)', latex_matrix)
    matrix = []
    for row in rows:
        clean_row = row.replace('\\', '').strip()
        parts = [cell.strip() for cell in clean_row.split('&')]
        matrix.append([x for x in parts if x])
    return matrix

def get_part_routes(matrix) -> list:
    """獲取每個零件的加工路線"""
    part_routes = []
    for row in matrix[1:]:
        step_dept_map = {}
        step_dept_map[1] = 1  # 第一步驟固定是部門1

        for dept_idx, cell in enumerate(row[1:], start=1):
            if cell == '0' or not cell:
                continue
            try:
                steps = [int(x) for x in cell.split(',') if x.strip()]
                for step in steps:
                    step_dept_map[step] = dept_idx
            except ValueError:
                continue

        if step_dept_map:
            route = []
            max_step = max(step_dept_map.keys())
            for i in range(1, max_step + 1):
                if i in step_dept_map:
                    route.append(step_dept_map[i])
            part_routes.append(route)
    return part_routes

def calculate_flow_matrix(part_routes, n_departments) -> np.ndarray:
    """計算部門間的流量矩陣，部門編號從1到9"""
    flow_matrix = np.zeros((n_departments, n_departments), dtype=int)

    for route in part_routes:
        for i in range(len(route) - 1):
            from_dept = route[i] - 1  # 轉換為0-based索引
            to_dept = route[i + 1] - 1
            flow_matrix[from_dept][to_dept] += 1

    return flow_matrix

def calculate_total_distance(flow_matrix, department_order) -> int:
    """計算給定部門排列的總移動距離，部門1和部門10固定在頭和尾"""
    total_distance = 0
    n = len(department_order)

    for i in range(n):
        for j in range(n):
            if i != j:
                # 計算部門i和j之間的實際距離
                distance = abs(department_order[i] - department_order[j])
                # 乘以流量
                total_distance += flow_matrix[i][j] * distance

    return total_distance

def find_optimal_sa(flow_matrix) -> list:
    """使用模擬退火算法(Simulated Annealing)尋找最佳部門排列，部門1和部門9固定在頭和尾"""
    start_time = time.time()  # 記錄開始時間

    n = len(flow_matrix)
    temperature = 10000.0  # 增加初始溫度
    cooling_rate = 0.99  # 降低冷卻速率
    min_temperature = 0.01
    max_iterations = 10000  # 增加迭代次數

    current_solution = list(range(1, n-1))  # 排除部門1和部門9
    random.shuffle(current_solution)
    current_solution = [0] + current_solution + [n-1]  # 部門1在頭，部門9在尾
    current_distance = calculate_total_distance(flow_matrix, current_solution)
    best_solution = current_solution.copy()
    best_distance = current_distance

    iteration = 0
    while temperature > min_temperature and iteration < max_iterations:
        i, j = random.sample(range(1, n-1), 2)  # 只在中間部門中交換
        new_solution = current_solution.copy()
        new_solution[i], new_solution[j] = new_solution[j], new_solution[i]

        new_distance = calculate_total_distance(flow_matrix, new_solution)
        delta_e = new_distance - current_distance

        if delta_e < 0 or random.random() < math.exp(-delta_e / temperature):
            current_solution = new_solution
            current_distance = new_distance
            if current_distance < best_distance:
                best_solution = current_solution.copy()
                best_distance = current_distance

        temperature *= cooling_rate
        iteration += 1

    execution_time = time.time() - start_time  # 計算執行時間
    return best_solution, best_distance, execution_time

def find_optimal_greedy(flow_matrix) -> list:
    """使用貪婪演算法(Greedy Algorithm)尋找最佳部門排列，部門1和部門9固定在頭和尾"""
    start_time = time.time()  # 記錄開始時間

    n = len(flow_matrix)
    total_flow = np.sum(flow_matrix, axis=0) + np.sum(flow_matrix, axis=1)
    sorted_depts = sorted(range(1, n-1), key=lambda x: total_flow[x], reverse=True)  # 排除部門1和部門9

    arrangement = [0] + [-1] * (n-2) + [n-1]  # 部門1在頭，部門9在尾

    for i in range(len(sorted_depts)):
        best_pos = 1  # 從第二個位置開始
        best_distance = float('inf')

        for pos in range(1, n-1):  # 只在中間部門中選擇
            if arrangement[pos] == -1:
                arrangement[pos] = sorted_depts[i]
                distance = calculate_total_distance(flow_matrix, arrangement)
                if distance < best_distance:
                    best_distance = distance
                    best_pos = pos
                arrangement[pos] = -1

        arrangement[best_pos] = sorted_depts[i]

    execution_time = time.time() - start_time  # 計算執行時間
    return arrangement, calculate_total_distance(flow_matrix, arrangement), execution_time

def find_optimal_center(flow_matrix) -> list:
    """使用中心放置法(Center Placement)尋找最佳部門排列，部門1和部門9固定在頭和尾"""
    start_time = time.time()  # 記錄開始時間

    n = len(flow_matrix)
    total_flow = np.sum(flow_matrix, axis=0) + np.sum(flow_matrix, axis=1)
    sorted_depts = sorted(range(1, n-1), key=lambda x: total_flow[x], reverse=True)  # 排除部門1和部門9

    arrangement = [0] + [-1] * (n-2) + [n-1]  # 部門1在頭，部門9在尾
    mid = (n - 2) // 2 + 1
    left = mid - 1
    right = mid

    for i, dept in enumerate(sorted_depts):
        if i % 2 == 0 and right < n-1:
            arrangement[right] = dept
            right += 1
        elif left >= 1:
            arrangement[left] = dept
            left -= 1

    execution_time = time.time() - start_time  # 計算執行時間
    return arrangement, calculate_total_distance(flow_matrix, arrangement), execution_time

def find_optimal_genetic(flow_matrix, population_size=200, generations=500):
    """使用基因演算法(Genetic Algorithm)尋找最佳部門排列，部門1和部門9固定在頭和尾"""
    start_time = time.time()  # 記錄開始時間

    n = len(flow_matrix)

    def create_individual():
        middle = random.sample(range(1, n-1), n-2)  # 排除部門1和部門9
        return [0] + middle + [n-1]  # 部門1在頭，部門9在尾

    def crossover(parent1, parent2):
        point = random.randint(1, n-3)
        child = parent1[:point+1]
        for x in parent2:
            if x not in child and x != 0 and x != n-1:
                child.append(x)
        return [0] + child + [n-1]

    def mutate(individual):
        if random.random() < 0.3:  # 增加突變率
            i, j = random.sample(range(1, n-1), 2)  # 只在中間部門中交換
            individual[i], individual[j] = individual[j], individual[i]
        return individual

    # 初始化種群
    population = [create_individual() for _ in range(population_size)]
    best_solution = None
    best_distance = float('inf')

    for _ in range(generations):
        # 評估適應度
        fitness = [(ind, calculate_total_distance(flow_matrix, ind)) for ind in population]
        fitness.sort(key=lambda x: x[1])

        if fitness[0][1] < best_distance:
            best_solution = fitness[0][0].copy()
            best_distance = fitness[0][1]

        # 選擇精英
        new_population = [fitness[0][0]]

        # 生成新一代
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(fitness[:population_size//4], 2)  # 只從前25%選擇
            child = crossover(parent1[0], parent2[0])
            child = mutate(child)
            new_population.append(child)

        population = new_population

    execution_time = time.time() - start_time  # 計算執行時間
    return best_solution, best_distance, execution_time

def find_optimal_hybrid(flow_matrix):
    """使用混合演算法：先用貪婪演算法找到初始解，再用模擬退火算法優化，部門1和部門9固定在頭和尾"""
    start_time = time.time()

    # 先用貪婪演算法找到初始解
    greedy_solution, _, _ = find_optimal_greedy(flow_matrix)

    # 再用模擬退火算法優化
    n = len(flow_matrix)
    temperature = 10000.0
    cooling_rate = 0.99
    min_temperature = 0.01
    max_iterations = 10000

    current_solution = greedy_solution.copy()
    current_distance = calculate_total_distance(flow_matrix, current_solution)
    best_solution = current_solution.copy()
    best_distance = current_distance

    iteration = 0
    while temperature > min_temperature and iteration < max_iterations:
        i, j = random.sample(range(1, n-1), 2)  # 只在中間部門中交換
        new_solution = current_solution.copy()
        new_solution[i], new_solution[j] = new_solution[j], new_solution[i]

        new_distance = calculate_total_distance(flow_matrix, new_solution)
        delta_e = new_distance - current_distance

        if delta_e < 0 or random.random() < math.exp(-delta_e / temperature):
            current_solution = new_solution
            current_distance = new_distance
            if current_distance < best_distance:
                best_solution = current_solution.copy()
                best_distance = current_distance

        temperature *= cooling_rate
        iteration += 1

    execution_time = time.time() - start_time
    return best_solution, best_distance, execution_time

def calculate_part_distances(part_routes, department_order):
    """計算每個零件在特定部門排列下的移動距離"""
    distances = {}
    for i, route in enumerate(part_routes, start=1):
        total_distance = 0
        for j in range(1, len(route)):
            # 使用部門排列映射來計算實際距離
            from_dept = department_order[route[j-1]-1]
            to_dept = department_order[route[j]-1]
            total_distance += abs(to_dept - from_dept)
        distances[i] = total_distance
    return distances

def generate_random_ftc_matrix(n_parts, n_departments, max_steps=None, max_parallel=None, seed=None) -> str:
    """
    隨機生成 FTC From-To Chart 矩陣
    
    參數:
    n_parts: 零件數量
    n_departments: 部門數量
    max_steps: 每個零件的最大加工步驟數
    max_parallel: 每個步驟的最大並行加工數
    seed: 隨機數種子，用於生成可重現的結果
    
    返回:
    LaTeX 格式的矩陣字串
    """
    if max_steps is None:
        max_steps = 10
    if max_parallel is None:
        max_parallel = 3
    if seed is not None:
        random.seed(seed)  # 設置隨機數種子
    # 生成矩陣標題行（部門編號）
    matrix = []
    header = ["0"]
    header.extend([str(i) for i in range(1, n_departments + 1)])  # 部門編號從1開始
    header.append("\\\\")
    matrix.append(header)

    # 生成每個零件的加工路線
    for part in range(1, n_parts + 1):  # 零件編號從1開始
        row = [str(part)]  # 第一列是零件編號

        # 為每個零件生成不重複的加工步驟
        steps = random.randint(1, max_steps)
        step_assignments = {}

        # 生成所有可能的步驟
        all_steps = list(range(1, steps + 1))
        random.shuffle(all_steps)  # 打亂步驟順序

        # 為每個部門分配步驟
        for dept in range(1, n_departments + 1):
            # 隨機決定是否在這個部門有加工步驟
            if random.random() < 0.5 and all_steps:  # 50% 的機率有加工步驟，且還有未分配的步驟
                # 隨機選擇1到max_parallel個步驟
                n_steps = random.randint(1, min(max_parallel, len(all_steps)))
                if n_steps > 0:
                    # 從剩餘的步驟中選擇前n_steps個
                    selected_steps = all_steps[:n_steps]
                    all_steps = all_steps[n_steps:]  # 移除已選擇的步驟
                    step_assignments[dept] = selected_steps
                else:
                    step_assignments[dept] = []
            else:
                step_assignments[dept] = []

        # 將步驟分配到矩陣行
        for dept in range(1, n_departments + 1):
            if dept in step_assignments and step_assignments[dept]:
                row.append(','.join(map(str, sorted(step_assignments[dept]))))  # 步驟按順序排列
            else:
                row.append('0')

        row.append("\\\\")
        matrix.append(row)

    # 轉換為 LaTeX 格式
    latex = "\\begin{bmatrix}\n"
    for row in matrix:
        latex += " & ".join(row) + "\n"
    latex += "\\end{bmatrix}"

    return latex

def calculate_objective_function(part_routes, department_order, quantities) -> int:
    """
    計算目標函數值
    
    參數:
    part_routes: 零件加工路線
    department_order: 部門排列順序
    quantities: 零件數量列表
    
    返回:
    目標函數值
    """
    total_distance = 0
    for i, route in enumerate(part_routes):
        part_distance = 0
        for j in range(1, len(route)):
            # 使用部門排列映射來計算實際距離
            from_dept = department_order[route[j-1]-1]
            to_dept = department_order[route[j]-1]
            part_distance += abs(to_dept - from_dept)
        total_distance += part_distance * quantities[i]
    return total_distance

def main() -> None:
    # 使用預設矩陣或生成隨機矩陣
    use_random = input("是否使用隨機生成的矩陣？(y/n): ").lower() == 'y'

    if use_random:
        n_parts = int(input("請輸入零件數量: "))
        n_departments = int(input("請輸入部門數量: "))

        # 處理最大加工步驟數的輸入
        max_steps_input = input("請輸入每個零件的最大加工步驟數: (不輸入預設10)")
        max_steps = int(max_steps_input) if max_steps_input.strip() else None

        # 處理最大並行加工數的輸入
        max_parallel_input = input("請輸入每個步驟的最大並行加工數: (不輸入預設3)")
        max_parallel = int(max_parallel_input) if max_parallel_input.strip() else None

        # 輸入隨機數種子
        seed = input("請輸入隨機數種子: (不輸入預設None)")
        seed = str(seed) if seed.strip() else None

        latex_matrix = generate_random_ftc_matrix(n_parts, n_departments, max_steps, max_parallel, seed)

        filename = f"ftc_matrix_{n_parts}parts_{n_departments}depts.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(latex_matrix)
        print(f"\n矩陣已保存到檔案：{filename}")
    else:
        # 使用預設的 LaTeX 矩陣
        latex_matrix = LATEX_MATRIX
        print("\n使用的矩陣：")
        print(latex_matrix)

    matrix = parse_matrix(latex_matrix)
    part_routes = get_part_routes(matrix)
    n_departments = len(matrix[0]) - 1  # 減去標題列
    n_parts = len(matrix) - 1  # 減去標題列

    # 計算流量矩陣
    flow_matrix = calculate_flow_matrix(part_routes, n_departments)

    # 輸入每個零件的數量
    quantities = []
    for i in range(n_parts):
        q = int(input(f"請輸入零件 {i+1} 的數量："))
        quantities.append(q)

    # 計算原始排列的目標函數值
    original_order = list(range(n_departments))
    original_value = calculate_objective_function(part_routes, original_order, quantities)

    print("\n比較不同演算法的效能：")
    print("=" * 60)
    print(f"原始目標函數值：{original_value}")
    print("=" * 60)

    algorithms = [
        ("模擬退火算法", find_optimal_sa),
        ("貪婪演算法", find_optimal_greedy),
        ("中心放置法", find_optimal_center),
        ("基因演算法", find_optimal_genetic),
        ("混合演算法", find_optimal_hybrid)
    ]

    best_algorithm = None
    best_value = float('inf')

    for name, algorithm in algorithms:
        print(f"\n{name}：")
        print("-" * 40)

        order, distance, exec_time = algorithm(flow_matrix)
        value = calculate_objective_function(part_routes, order, quantities)
        improvement = (original_value - value) / original_value * 100

        print(f"部門排列：{' -> '.join(str(x+1) for x in order)}")
        print(f"目標函數值：{value}")
        print(f"改善幅度：{improvement:.2f}%")
        print(f"執行時間：{exec_time:.2f} 秒")

        if value < best_value:
            best_value = value
            best_algorithm = name

    print("\n" + "=" * 60)
    print(f"最佳演算法：{best_algorithm}")
    print(f"最佳改善幅度：{(original_value - best_value) / original_value * 100:.2f}%")
    print(f"最佳目標函數值：{best_value}")
    
# 主程式
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n程式中斷。")
    except Exception as e:
        print(f"發生錯誤：{e}")
    finally:
        print("程式結束！")