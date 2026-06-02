import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import (
    parse_matrix,
    get_part_routes,
    calculate_flow_matrix,
    calculate_total_distance,
    calculate_objective_function,
    calculate_part_distances,
    generate_random_ftc_matrix,
    find_optimal_sa,
    find_optimal_greedy,
    find_optimal_center,
    find_optimal_genetic,
    find_optimal_hybrid,
    LATEX_MATRIX,
)

# ── 測試常數 ──────────────────────────────────────────────────────────────────

SIMPLE_LATEX = r"""
\begin{bmatrix}
 0 & 1 & 2 & 3\\
 1 & 1 & 2 & 3\\
\end{bmatrix}
"""

# 3個部門、1個零件，路線 1→2→3
SIMPLE_ROUTES = [[1, 2, 3]]
SIMPLE_FLOW = np.array([
    [0, 1, 0],
    [0, 0, 1],
    [0, 0, 0],
])

# 5個部門的流量矩陣（供演算法測試用，保證有2個以上中間部門可交換）
ALGO_FLOW = np.array([
    [0, 5, 0, 2, 0],
    [0, 0, 4, 0, 1],
    [0, 0, 0, 3, 0],
    [0, 0, 0, 0, 6],
    [0, 0, 0, 0, 0],
], dtype=int)

N_ALGO = 5  # ALGO_FLOW 的部門數


# ── 輔助函式 ──────────────────────────────────────────────────────────────────

def is_valid_arrangement(arrangement, n):
    """驗證排列合法：長度正確、頭尾固定為0與n-1、無重複"""
    return (
        len(arrangement) == n
        and arrangement[0] == 0
        and arrangement[-1] == n - 1
        and sorted(arrangement) == list(range(n))
    )


# ── parse_matrix ──────────────────────────────────────────────────────────────

class TestParseMatrix:
    def test_returns_list_of_lists(self):
        result = parse_matrix(SIMPLE_LATEX)
        assert isinstance(result, list)
        assert all(isinstance(row, list) for row in result)

    def test_header_row(self):
        result = parse_matrix(SIMPLE_LATEX)
        assert result[0] == ["0", "1", "2", "3"]

    def test_data_row(self):
        result = parse_matrix(SIMPLE_LATEX)
        assert result[1] == ["1", "1", "2", "3"]

    def test_row_count(self):
        result = parse_matrix(SIMPLE_LATEX)
        assert len(result) == 2  # 1 header + 1 零件


# ── get_part_routes ───────────────────────────────────────────────────────────

class TestGetPartRoutes:
    def test_sequential_route(self):
        # 部門1=步驟1, 部門2=步驟2, 部門3=步驟3 → 路線 [1,2,3]
        matrix = [["0","1","2","3"], ["1","1","2","3"]]
        assert get_part_routes(matrix) == [[1, 2, 3]]

    def test_skip_zero_department(self):
        # 部門2無步驟 → 路線 [1,3]
        matrix = [["0","1","2","3"], ["1","1","0","2"]]
        assert get_part_routes(matrix) == [[1, 3]]

    def test_multiple_parts(self):
        matrix = [
            ["0","1","2","3"],
            ["1","1","2","3"],
            ["2","1","0","2"],
        ]
        routes = get_part_routes(matrix)
        assert len(routes) == 2
        assert routes[0] == [1, 2, 3]
        assert routes[1] == [1, 3]

    def test_parallel_steps_same_dept(self):
        # 部門2同時有步驟2和3 → route包含 dept2 兩次
        matrix = [["0","1","2","3"], ["1","1","2,3","4"]]
        routes = get_part_routes(matrix)
        assert routes[0][0] == 1
        assert routes[0].count(2) == 2  # 步驟2和3都在部門2
        assert routes[0][-1] == 3

    def test_step_one_always_dept_one(self):
        matrix = [["0","1","2","3"], ["1","1","2","3"]]
        assert get_part_routes(matrix)[0][0] == 1


# ── calculate_flow_matrix ─────────────────────────────────────────────────────

class TestCalculateFlowMatrix:
    def test_sequential_flow(self):
        result = calculate_flow_matrix(SIMPLE_ROUTES, 3)
        np.testing.assert_array_equal(result, SIMPLE_FLOW)

    def test_shape(self):
        result = calculate_flow_matrix(SIMPLE_ROUTES, 3)
        assert result.shape == (3, 3)

    def test_multi_part_counts(self):
        routes = [[1, 2, 3], [1, 3]]
        result = calculate_flow_matrix(routes, 3)
        assert result[0][1] == 1  # 1→2: 1次
        assert result[0][2] == 1  # 1→3: 1次
        assert result[1][2] == 1  # 2→3: 1次

    def test_no_self_flow(self):
        result = calculate_flow_matrix(SIMPLE_ROUTES, 3)
        for i in range(3):
            assert result[i][i] == 0


# ── calculate_total_distance ──────────────────────────────────────────────────

class TestCalculateTotalDistance:
    def test_original_order(self):
        # flow: 0→1=1次, 1→2=1次；order[0,1,2]各相差1 → 距離=2
        assert calculate_total_distance(SIMPLE_FLOW, [0, 1, 2]) == 2

    def test_spread_order_increases_distance(self):
        # order [0,2,1]：dept0在0, dept1在2, dept2在1
        # flow[0][1]=1: |0-2|=2; flow[1][2]=1: |2-1|=1 → 3
        assert calculate_total_distance(SIMPLE_FLOW, [0, 2, 1]) == 3

    def test_zero_flow_no_contribution(self):
        zero_flow = np.zeros((3, 3), dtype=int)
        assert calculate_total_distance(zero_flow, [0, 1, 2]) == 0


# ── calculate_objective_function（README核心公式 min z = Σ m_i × Q_i） ─────────

class TestCalculateObjectiveFunction:
    def test_single_part_qty_one(self):
        # m_1=2（1→2→3移動2步）, Q_1=1 → Z=2
        assert calculate_objective_function(SIMPLE_ROUTES, [0, 1, 2], [1]) == 2

    def test_quantity_scales_linearly(self):
        # Q_1=5 → Z=10，驗證 Q_i 乘數正確
        assert calculate_objective_function(SIMPLE_ROUTES, [0, 1, 2], [5]) == 10

    def test_multi_part_sum(self):
        # 零件1: m=2, Q=1 → 2; 零件2: 路線1→3, m=2, Q=2 → 4; Z=6
        routes = [[1, 2, 3], [1, 3]]
        assert calculate_objective_function(routes, [0, 1, 2], [1, 2]) == 6

    def test_zero_quantity(self):
        assert calculate_objective_function(SIMPLE_ROUTES, [0, 1, 2], [0]) == 0

    def test_same_dept_step_no_distance(self):
        # 路線 1→2→2→3：步驟3和4在同一部門，移動距離=0
        routes = [[1, 2, 2, 3]]
        dist_with_repeat = calculate_objective_function(routes, [0, 1, 2], [1])
        dist_without = calculate_objective_function([[1, 2, 3]], [0, 1, 2], [1])
        assert dist_with_repeat == dist_without  # 同部門不加距離

    def test_readme_formula(self):
        # 直接對應 README：min z = Σ m_i × Q_i
        routes = [[1, 2, 3], [1, 3]]
        quantities = [3, 2]
        result = calculate_objective_function(routes, [0, 1, 2], quantities)
        # 手算：零件1 m=2, Q=3 → 6；零件2 m=2, Q=2 → 4；Z=10
        assert result == 10


# ── calculate_part_distances ──────────────────────────────────────────────────

class TestCalculatePartDistances:
    def test_returns_dict(self):
        result = calculate_part_distances(SIMPLE_ROUTES, [0, 1, 2])
        assert isinstance(result, dict)

    def test_keyed_by_part_number(self):
        result = calculate_part_distances(SIMPLE_ROUTES, [0, 1, 2])
        assert 1 in result

    def test_distance_value(self):
        result = calculate_part_distances(SIMPLE_ROUTES, [0, 1, 2])
        assert result[1] == 2

    def test_multiple_parts(self):
        routes = [[1, 2, 3], [1, 3]]
        result = calculate_part_distances(routes, [0, 1, 2])
        assert result[1] == 2
        assert result[2] == 2


# ── generate_random_ftc_matrix ────────────────────────────────────────────────

class TestGenerateRandomFtcMatrix:
    def test_returns_string(self):
        assert isinstance(generate_random_ftc_matrix(3, 4), str)

    def test_contains_bmatrix(self):
        result = generate_random_ftc_matrix(3, 4)
        assert r"\begin{bmatrix}" in result
        assert r"\end{bmatrix}" in result

    def test_seed_reproducibility(self):
        # README 要求：偽隨機數功能（相同 seed 產生相同結果）
        r1 = generate_random_ftc_matrix(5, 6, seed="42")
        r2 = generate_random_ftc_matrix(5, 6, seed="42")
        assert r1 == r2

    def test_different_seeds_differ(self):
        r1 = generate_random_ftc_matrix(5, 6, seed="42")
        r2 = generate_random_ftc_matrix(5, 6, seed="99")
        assert r1 != r2

    def test_output_parseable(self):
        # README 要求：統一資料格式（隨機矩陣可被 parse）
        latex = generate_random_ftc_matrix(4, 5, seed="1")
        matrix = parse_matrix(latex)
        assert len(matrix) > 0


# ── 演算法有效性（共用 fixture） ─────────────────────────────────────────────

@pytest.fixture
def algo_inputs():
    return ALGO_FLOW, N_ALGO


class TestSimulatedAnnealing:
    def test_returns_three_tuple(self, algo_inputs):
        fm, _ = algo_inputs
        assert len(find_optimal_sa(fm)) == 3

    def test_valid_arrangement(self, algo_inputs):
        fm, n = algo_inputs
        arrangement, _, _ = find_optimal_sa(fm)
        assert is_valid_arrangement(arrangement, n)

    def test_distance_non_negative(self, algo_inputs):
        fm, _ = algo_inputs
        _, distance, _ = find_optimal_sa(fm)
        assert distance >= 0

    def test_distance_consistent_with_arrangement(self, algo_inputs):
        fm, _ = algo_inputs
        arrangement, distance, _ = find_optimal_sa(fm)
        assert distance == calculate_total_distance(fm, arrangement)

    def test_execution_time_non_negative(self, algo_inputs):
        fm, _ = algo_inputs
        _, _, t = find_optimal_sa(fm)
        assert t >= 0


class TestGreedyAlgorithm:
    def test_returns_three_tuple(self, algo_inputs):
        fm, _ = algo_inputs
        assert len(find_optimal_greedy(fm)) == 3

    def test_valid_arrangement(self, algo_inputs):
        fm, n = algo_inputs
        arrangement, _, _ = find_optimal_greedy(fm)
        assert is_valid_arrangement(arrangement, n)

    def test_distance_consistent_with_arrangement(self, algo_inputs):
        fm, _ = algo_inputs
        arrangement, distance, _ = find_optimal_greedy(fm)
        assert distance == calculate_total_distance(fm, arrangement)

    def test_deterministic(self, algo_inputs):
        fm, _ = algo_inputs
        arr1, dist1, _ = find_optimal_greedy(fm)
        arr2, dist2, _ = find_optimal_greedy(fm)
        assert arr1 == arr2 and dist1 == dist2


class TestCenterPlacement:
    def test_returns_three_tuple(self, algo_inputs):
        fm, _ = algo_inputs
        assert len(find_optimal_center(fm)) == 3

    def test_valid_arrangement(self, algo_inputs):
        fm, n = algo_inputs
        arrangement, _, _ = find_optimal_center(fm)
        assert is_valid_arrangement(arrangement, n)

    def test_distance_consistent_with_arrangement(self, algo_inputs):
        fm, _ = algo_inputs
        arrangement, distance, _ = find_optimal_center(fm)
        assert distance == calculate_total_distance(fm, arrangement)

    def test_deterministic(self, algo_inputs):
        fm, _ = algo_inputs
        arr1, dist1, _ = find_optimal_center(fm)
        arr2, dist2, _ = find_optimal_center(fm)
        assert arr1 == arr2 and dist1 == dist2


class TestGeneticAlgorithm:
    def test_returns_three_tuple(self, algo_inputs):
        fm, _ = algo_inputs
        assert len(find_optimal_genetic(fm)) == 3

    def test_valid_arrangement(self, algo_inputs):
        fm, n = algo_inputs
        arrangement, _, _ = find_optimal_genetic(fm)
        assert is_valid_arrangement(arrangement, n)

    def test_distance_consistent_with_arrangement(self, algo_inputs):
        fm, _ = algo_inputs
        arrangement, distance, _ = find_optimal_genetic(fm)
        assert distance == calculate_total_distance(fm, arrangement)

    def test_execution_time_non_negative(self, algo_inputs):
        fm, _ = algo_inputs
        _, _, t = find_optimal_genetic(fm)
        assert t >= 0


class TestHybridAlgorithm:
    def test_returns_three_tuple(self, algo_inputs):
        fm, _ = algo_inputs
        assert len(find_optimal_hybrid(fm)) == 3

    def test_valid_arrangement(self, algo_inputs):
        fm, n = algo_inputs
        arrangement, _, _ = find_optimal_hybrid(fm)
        assert is_valid_arrangement(arrangement, n)

    def test_distance_consistent_with_arrangement(self, algo_inputs):
        fm, _ = algo_inputs
        arrangement, distance, _ = find_optimal_hybrid(fm)
        assert distance == calculate_total_distance(fm, arrangement)


# ── 整合測試：在專案預設矩陣上執行全部演算法 ─────────────────────────────────

class TestIntegration:
    def test_all_algorithms_on_default_matrix(self):
        """README 要求：多演算法比較，全部能在預設矩陣上執行完畢"""
        matrix = parse_matrix(LATEX_MATRIX)
        n_depts = len(matrix[0]) - 1
        part_routes = get_part_routes(matrix)
        fm = calculate_flow_matrix(part_routes, n_depts)

        for algo in [find_optimal_sa, find_optimal_greedy, find_optimal_center,
                     find_optimal_genetic, find_optimal_hybrid]:
            arrangement, distance, _ = algo(fm)
            assert is_valid_arrangement(arrangement, n_depts), \
                f"{algo.__name__} 回傳無效排列：{arrangement}"
            assert distance >= 0

    def test_objective_consistent_across_flow_and_routes(self):
        """calculate_total_distance 與 calculate_objective_function 在 qty=1 時結果一致"""
        routes = [[1, 2, 3], [1, 3]]
        fm = calculate_flow_matrix(routes, 3)
        order = [0, 1, 2]
        flow_dist = calculate_total_distance(fm, order)
        obj_dist = calculate_objective_function(routes, order, [1, 1])
        assert flow_dist == obj_dist
