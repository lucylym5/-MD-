from pathlib import Path

import numpy as np

from classes import Atomsystem, MDParameters   # 导入参数类和原子体系类

"""计算两粒子间的最小距离，返回形状为(3,)的向量"""
def minimum_displacement(
        displacement: np.ndarray,
        cell: np.ndarray,
        inverse_cell: np.ndarray    # 盒子矩阵的逆
) -> np.ndarray:
    frac_displacement = displacement @ inverse_cell    # 将距离坐标 转换为 相对于盒子尺寸的分数坐标
    frac_displacement -= np.rint(frac_displacement)    # rint：取最近整数
    return frac_displacement @ cell                    # 转换回普通坐标再输出


"""给原子 atom_index 追加一个近邻 neighbor_index"""
def append_neighbor(atoms: Atomsystem, atom_index: int, neighbor_index: int) -> None:
    next_slot = int(atoms.nlist[atom_index])    # 下一个近邻编号存储位置
    atoms.neighbor_list[atom_index, next_slot] = neighbor_index    # 将新近邻编号存入
    atoms.nlist[atom_index] += 1    # 近邻数量 +1


"""检查体系内每对原子，如果它们的最小距离不超过 Verlet 近邻半径，将它们互相加入对方的近邻表"""
def build_neighbor_list(atoms: Atomsystem, parameters: MDParameters):
    inverse_cell = np.linalg.inv(atoms.cell)       # 算一次盒子矩阵的逆，用于计算最小距离
    rv_square = parameters.neighbor_radius ** 2    # Verlet 近邻半径平方

    # 更新前，清空近邻表
    atoms.nlist.fill(0)
    atoms.neighbor_list.fill(-1)

    for i in range(parameters.natom-1):            # 不重复地遍历原子对
        for j in range(i+1, parameters.natom):
            displacement = atoms.positions[j]-atoms.positions[i]
            min_displacement = minimum_displacement(
                displacement,    # 传入原始位移
                atoms.cell,      # 传入盒子矩阵
                inverse_cell     # 传入盒子矩阵的逆
            )
            d_square = float(np.dot(min_displacement, min_displacement))    # 距离平方（向量点乘）

            if d_square <= rv_square:         # 判断距离是否在 Verlet 半径内
                append_neighbor(atoms, j, i)  # 将 i 加到 j 的近邻表内
                append_neighbor(atoms, i, j)  # 将 j 加到 i 的近邻表内


"""输出目标原子近邻表文件"""
def write_neighbor_list(
        atoms: Atomsystem,        # 原子系
        output_index: int,        # 输出原子编号
        output_file: str | Path   # 输出文件路径
) -> None:
    atom_index = output_index - 1
    output = atoms.output_neighbors(atom_index)    # 读取目标原子的近邻
    output_file = Path(output_file)
    with output_file.open("w", encoding="utf-8", newline="\n") as file:
        for i in output:
            x, y, z = atoms.positions[i, :]
            file.write(
                f"{int(i) + 1}\t{x:.12f}\t{y:.12f}\t{z:.12f}\n"    # 分别输出原子编号、x、y、z
            )
