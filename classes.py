from pathlib import Path
import numpy as np
import pandas as pd

from dataclasses import dataclass

"""input参数类"""
@dataclass(frozen = True)  # frozen: 对象创建完成后，不可重新给属性赋值
class MDParameters:
    atomid: str
    natom: int          # 原子总数
    mass: float         # 原子摩尔质量
    geo_dir: str        # 构型文件路径
    rcut: float         # 势的阶段半径
    neighbor_r: float   # 近邻表额外截断半径
    neighbor_n: int     # 最大近邻原子数
    epsilon: float
    sigma: float
    dt: float
    nstep: int
    step_out: int
    step_search: int
    temperature_init: float
    ensamble: str        # 系综类型
    ensamble_T: float
    tau: float
    nraise: int
    thermostat: int      # 恒温器类型

    # 计算Verlet搜索半径
    @property  # 把方法变得“看起来像属性”，调用时不需要写圆括号。（没有额外保存一个值）
    def neighbor_radius(self) -> float:
        return self.rcut + self.neighbor_r

    @classmethod  # 从 read_input 的输出创建 MDParameters 类对象
    def from_input(cls, parameters: dict[str, int | float | str]):
        return cls(
            atomid = str(parameters["id"]),
            natom = int(parameters["natom"]),
            mass = float(parameters["mass"]),
            geo_dir = str(parameters["geo_dir"]),
            rcut = float(parameters["rcut"]),
            neighbor_r = float(parameters["neighbor_r"]),
            neighbor_n = int(parameters["neighbor_n"]),
            epsilon = float(parameters["epsilon"]),
            sigma = float(parameters["sigma"]),
            dt = float(parameters["dt"]),
            nstep = int(parameters["nstep"]),
            step_out = int(parameters["step_out"]),
            step_search = int(parameters["step_search"]),
            temperature_init = float(parameters["temperature_init"]),
            ensamble = str(parameters["ensamble"]),
            ensamble_T = float(parameters["ensamble_T"]),
            tau = float(parameters["tau"]),
            nraise = int(parameters["nraise"]),
            thermostat = int(parameters["thermostat"]),
        )


"""原子类"""
@dataclass()
class Atomsystem:
    natom: int
    elements: np.ndarray      # (natom, 1)
    positions: np.ndarray     # (natom, 3)
    velocities: np.ndarray    # (natom, 3)
    cell: np.ndarray          # 周期性盒子矩阵 (3, 3)
    nlist: np.ndarray         # 每个原子的近邻数量
    neighbor_list: np.ndarray # 每个原子的近邻表（编号）

    # 输出某原子近邻表
    # 原子编号从0开始
    def output_neighbors(self, atom_index: int) -> np.ndarray:    
        neighbor_num = int(self.nlist[atom_index])
        return self.neighbor_list[atom_index, :neighbor_num]
        

    @classmethod    # 从 read_geometry 的输出创建 Atomsystem 类对象
    def from_geometry(
        cls, 
        cell: np.ndarray,
        position_table: pd.DataFrame,
        velocity_table: pd.DataFrame,
        input_natom: int,
        neighbors_n: int
    ):  
        positions = position_table[["x", "y", "z"]].to_numpy(dtype=np.float64)
        natom = positions.shape[0]     # shape[0]：ndarray行数，shape[1]：ndarray列数
        velocities = velocity_table[["x", "y", "z"]].to_numpy(dtype=np.float64)

        if natom != input_natom:
            raise ValueError(f"预期 {input_natom} 个原子，实际读取到 {natom} 个，错误")

        nlist = np.zeros(natom, dtype = np.int32)    # 初始化近邻数量为0向量
        neighbor_list = np.full((natom, neighbors_n), -1, dtype = np.int32)    #初始化近邻表为一所有元素均为-1的矩阵

        return cls(
            natom = natom,   
            elements = position_table["element"].to_numpy(dtype=str),
            positions = positions,
            velocities = velocities,
            cell = np.asarray(cell, dtype = np.float64),
            nlist = nlist,
            neighbor_list = neighbor_list
        )