""" 
    完成 RDF 统计 
    径向分布函数是所有原子的平均
"""


import numpy as np
from dataclasses import dataclass

from classes import Atomsystem, MDParameters 
from neighbor_list import minimum_displacement


""" 将周围空间分为 rdf_rcut/dr 各球壳，球壳信息 """
@dataclass
class RDFAccumulator:
    edges: np.ndarray      # 球壳边界，851个数
    centers: np.ndarray    # 球壳中心
    counts: np.ndarray     # 各球壳内目前累计原子数
    nframes: int           # 已经统计了多少帧

""" 准备球壳信息，模拟开始前调用一次即可 """
def initialize_rdf(
        parameters: MDParameters,
) -> RDFAccumulator:
    RDFprepare = RDFAccumulator()

    n = parameters.rdf_rcut / parameters.rdf_dr
    edges = np.linspace(0.0, parameters.rdf_rcut, n + 1)    # linspace 默认首尾都包含
    centers = 0.5 * (edges[:-1] + edges[1:])                # 中心：内外边界平均值
    counts = np.zeros(n, dtype=np.int64)

    return RDFprepare(
        edges = edges,
        centers = centers, 
        counts = counts,
        nframes = 0,  
    )

""" 统计当前帧，各球壳内累计原子数 """
def accumulate_rdf(
        state: RDFAccumulator,
        atoms: Atomsystem, 
        parameters: MDParameters,
) -> None:
    inverse_cell = np.linalg.inv(atoms.cell)
    rcut2 = parameters.rdf_rcut ** 2
    dr = parameters.rdf_dr

    if parameters.rdf_rcut > parameters.rcut:
        for i in range(atoms.natom):
            for j in range(i):
                displacement = minimum_displacement(
                    atoms.positions[j] - atoms.positions[i],
                    atoms.cell,
                    inverse_cell,
                )
                distance2 = float(
                    np.dot(displacement, displacement)
                )
                
                if distance2 <= rcut2:
                    bin_index = int(np.sqrt(distance2) / dr)
                    state.count[bin_index] += 2
    else:
        for i in range(atoms.natom):
            for j in atoms.neighbor_list[i]:
                displacement = minimum_displacement(
                    atoms.positions[j] - atoms.positions[i],
                    atoms.cell,
                    inverse_cell,
                )
                distance2 = float(
                    np.dot(displacement, displacement)
                )
                                
                if distance2 <= rcut2:
                    bin_index = int(np.sqrt(distance2) / dr)
                    state.count[bin_index] += 1

    state.nframe += 1

""" 将计数转换为g(r) """
def finalize_rdf(
        state: RDFAccumulator, 
        atoms: Atomsystem,
)-> tuple:
    r = state.centers.copy()

    volume = abs(np.linalg.det(atoms.cell))
    density = atoms.natom / volume
    shell_volume = 4/3 * np.pi * (state.edges[1:]**3 - state.edges[:-1]**3)
    g_r = state.counts / (state.nframes * density * shell_volume)

    return r, g_r


