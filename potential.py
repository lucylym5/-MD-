import numpy as np

from classes import Atomsystem, MDParameters
from neighbor_list import minimum_displacement

"""由距离 计算一对原子的 Leonard-Jones 势"""
def lj_pair_energy(
        distance2: float,
        epsilon: float,
        sigma: float
) -> float:
    sigma6_over_distance6 = sigma**6 / distance2**3
    return 4 * epsilon * (
        sigma6_over_distance6**2 - sigma6_over_distance6
    )

"""由距离向量 计算原子受力"""
def lj_pair_force(
        displacement: np.ndarray,
        epsilon: float,
        sigma: float
) -> np.ndarray:
    displacement_square = np.dot(displacement, displacement)
    sigma6_over_distance6 = sigma**6 / displacement_square**3

    force = 24 * epsilon / displacement_square * (
        2 * sigma6_over_distance6**2 - sigma6_over_distance6) * displacement
    return force


"""体系总势能、力"""
def calculate_lj(
        atoms: Atomsystem,
        parameters: MDParameters
)-> tuple[float, np.ndarray, np.ndarray]:

    inverse_cell = np.linalg.inv(atoms.cell)    # 盒子矩阵的逆
    rcutoff_square = parameters.rcut**2

    # 待计算量
    total_energy = 0.0
    forces = np.zeros_like(atoms.positions)
    virial_tensor = np.zeros((3,3))

    # 计算截断势能
    cutoff_energy = lj_pair_energy(
                        rcutoff_square,
                        parameters.epsilon, 
                        parameters.sigma
                    )

    # 遍历每个原子的近邻表，计算总能、每个原子受力和应力
    for i in range(atoms.natom):
        for neighbor in atoms.output_neighbors(i):
            j = int(neighbor)

            if j >= i:      
                continue

            # 计算近邻原子对的最小距离
            displacement = atoms.positions[j] - atoms.positions[i]
            displacement = minimum_displacement(
                displacement,
                atoms.cell,
                inverse_cell,
            )
            displacement_square = np.dot(displacement, displacement)

            # 计算总势能
            if displacement_square <= rcutoff_square:
                total_energy += lj_pair_energy(
                    displacement_square,
                    parameters.epsilon, 
                    parameters.sigma
                ) - cutoff_energy

                # 计算每个原子受力
                force_ij = lj_pair_force(     # 原子 j 受到 i 的力
                    displacement,
                    parameters.epsilon,
                    parameters.sigma,
                )
                forces[i] -= force_ij
                forces[j] += force_ij

                # 计算维里张量
                virial_tensor += np.outer(
                    displacement,
                    force_ij,
                )

    return total_energy, forces, virial_tensor


"""计算应力张量"""
def lj_stress_tensor(
        atoms: Atomsystem,
        parameters: MDParameters,
        virial_tensor: np.ndarray,
):
    # 温度贡献
    volume = abs(np.linalg.det(atoms.cell))
    masses_amu = np.full(     # 单位：AMU
        atoms.natom,
        parameters.mass,
        dtype=np.float64
    )
    velocities = atoms.velocities

    # 去除体系整体平动
    center_of_mass_velocity = (
        np.sum(masses_amu[:, None] * velocities, axis=0)     # None: 把一维数组变 (natom,1) 矩阵；axis=0：按列求和
        / np.sum(masses_amu)
    )
    thermal_velocities = velocities - center_of_mass_velocity

    kinetic_tensor = (
        1.036426965e-4        # 单位换算
        * np.einsum(
            "i,ia,ib->ab",    # 用爱因斯坦求和约定 做张量计算
            masses_amu,
            thermal_velocities,
            thermal_velocities,
        )
    )
 
    stress = (                        # 单位：eV/A^3
        kinetic_tensor + virial_tensor
    ) / volume

    return stress
