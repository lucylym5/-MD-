import numpy as np

from classes import Atomsystem, MDParameters
from neighbor_list import build_neighbor_list           # 用于重建近邻表
from potential import calculate_lj, lj_stress_tensor    # 计算势能、力、应力

BOLTZMANN_CONSTANT = 1.38064852e-23    # J/K
NA = 6.02214086e23    # /mol
E = 1.602176634e-19   # C
UNIT_AMU_TO_KG = 1.66053906660e-27
UNIT_AI_TO_M = 1e-10
UNIT_PS_TO_S = 1e-12
UNIT_J_TO_EV = 6.241509e18
UNIT_KINETIC_TO_EV = UNIT_AMU_TO_KG * UNIT_AI_TO_M**2 / UNIT_PS_TO_S**2 * UNIT_J_TO_EV
BOLTZMANN_CONSTANT *= UNIT_J_TO_EV


"""初始化随机速度"""
def initialize_velocities(
        atoms: Atomsystem,
        parameters: MDParameters,
        seed: int,
) -> None:                          # 无输出，将初始速度直接存入atom
    
    # 1.均匀分布 初始化每个原子速度
    rng = np.random.default_rng(seed)
    velocities = rng.uniform(
        -0.5,
        0.5,
        size=(atoms.natom, 3)
    )

    # 2.消去质心速度
    center_velocity = np.mean(velocities, axis=0)
    velocities -= center_velocity

    # 3.缩放速度，使温度与指定温度匹配
    current_kinetic = (
        0.5
        * parameters.mass
        * np.sum(velocities**2)
        * UNIT_KINETIC_TO_EV
    )
    target_kinetic = (
        1.5
        * atoms.natom
        * BOLTZMANN_CONSTANT
        * parameters.temperature_init
    )
    velocities *= np.sqrt(target_kinetic / current_kinetic)
    atoms.velocities[:] = velocities    # 加索引[:]，赋值；不加，将左侧指向右侧


"""计算动能和温度"""
def Ekinetic_and_T(
        atoms:Atomsystem,
        parameters: MDParameters,
) -> tuple[float, float]:
    kinetic_energy = (
        0.5
        * parameters.mass
        * np.sum(atoms.velocities**2)
        * UNIT_KINETIC_TO_EV
    )
    temperature = (                     # 能均分：粒子平均动能 3kBT/2
        2/3 
        * kinetic_energy / atoms.natom
        / BOLTZMANN_CONSTANT
    )                              
    return kinetic_energy, temperature


"""将更新后位置包回周期盒"""
def wrap_positions(
    atoms: Atomsystem,
) -> None:
    inverse_cell = np.linalg.inv(atoms.cell)    # 盒子矩阵的逆
    fractional_positions = (
        atoms.positions @ inverse_cell          # (natom,3) @ (3,3)
    )
    fractional_positions -= np.floor(fractional_positions)  # 归一到 [0,1] 之间
    atoms.positions[:] = (
        fractional_positions @ atoms.cell
    )


"""一步Velocity Verlet算法"""
def velocity_verlet_step(
    atoms: Atomsystem,
    parameters: MDParameters,
    forces: np.ndarray,
    step: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    
    # 1.更新位置
    atoms.positions += (
        atoms.velocities * parameters.dt
        + forces / 2 / parameters.mass * parameters.dt**2 
        / UNIT_KINETIC_TO_EV
    )
    wrap_positions(atoms)   # 将位置归一化到盒子中

    # 2.按步数重建近邻表
    if step % parameters.step_search == 0:
        build_neighbor_list(atoms, parameters)

    # 3.计算势能、力和维里应力
    potential_energy, forces_new, virial_tensor = (
        calculate_lj(atoms,parameters)
    )

    # 4.更新速度
    atoms.velocities += (
        (forces + forces_new)
        / 2 / parameters.mass * parameters.dt / UNIT_KINETIC_TO_EV
    )
    stress_tensor = lj_stress_tensor(
        atoms,
        parameters,
        virial_tensor,
    )
    
    return potential_energy, forces_new, stress_tensor