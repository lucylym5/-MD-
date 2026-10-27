from pathlib import Path

import numpy as np

from classes import Atomsystem, MDParameters            # 导入参数类和原子体系类
from read import read_geometry, read_input              # 导入读取函数
from neighbor_list import build_neighbor_list           # 建立初始近邻表
from potential import calculate_lj, lj_stress_tensor    # 计算第0步物理量
from dynamics import velocity_verlet_step, Ekinetic_and_T

def output_file(
        atoms: Atomsystem,
        forces: np.ndarray,
        stress_tensor: np.ndarray,
        step: int,
        time: float,
        potential_energy: float,
        kinetic_energy: float,
        temperature: float,
        output_position: str | Path,
        output_velocity: str | Path,
        output_force: str | Path,
        output_stress: str | Path,
        output_log: str | Path,
):
    total_energy = (
        potential_energy + kinetic_energy
    )

    with output_position.open("a", encoding="utf-8", newline="\n") as file:
        file.write(
            f"% step {step} "
            f"time {time:.2f} ps\n"
        )
        for index in range(atoms.natom):
            x, y, z = atoms.positions[index]
            file.write(
                f"{index + 1}\t"
                f"{atoms.elements[index]}\t"
                f"{x:.10f}\t{y:.10f}\t{z:.10f}\n"
            )

    with output_velocity.open("a", encoding="utf-8", newline="\n") as file:
        file.write(
            f"% step {step} "
            f"time {time:.2f} ps\n"
        )
        for index in range(atoms.natom):
            x, y, z = atoms.velocities[index]
            file.write(
                f"{index + 1}\t"
                f"{atoms.elements[index]}\t"
                f"{x:.10f}\t{y:.10f}\t{z:.10f}\n"
            )

    with output_force.open("a", encoding="utf-8", newline="\n") as file:
        file.write(
            f"% step {step} "
            f"time {time:.2f} ps\n"
        )    
        for index in range(atoms.natom):
            fx, fy, fz = forces[index]
            file.write(
                f"{index + 1}\t"
                f"{atoms.elements[index]}\t"
                f"{fx:.10f}\t{fy:.10f}\t{fz:.10f}\n"
            )

    with output_stress.open("a", encoding="utf-8", newline="\n") as file:
        file.write(
            f"% step {step} "
            f"time {time:.8f} ps\n"
        )
        np.savetxt(
            file,
            stress_tensor,
            fmt="%.10f",
            delimiter="\t"
        )

    with output_log.open("a", encoding="utf-8", newline="\n") as file:
        file.write(
            f"{step}\t"
            f"{time:.8f}\t"
            f"{total_energy:.12f}\t"
            f"{potential_energy:.12f}\t"
            f"{kinetic_energy:.12f}\t"
            f"{temperature:.8f}\n"
        )


def main():   
    """读取geo和参数输入"""
    base_path = Path(__file__).resolve().parent   # 读取根目录
    input_file = base_path / "input.in"
    parameters_dict = read_input(input_file) 
    parameters = MDParameters.from_input(parameters_dict)

    geo_file =  base_path / parameters.geo_dir
    cell, position_table, velocity_table = read_geometry(geo_file)
    atoms = Atomsystem.from_geometry(cell, position_table, velocity_table, parameters.natom, parameters.neighbor_n)

    """输出准备"""
    output1 = base_path / "position.txt"
    output2 = base_path / "velocity.txt"
    output3 = base_path / "force.txt"
    output4 = base_path / "stress.txt"
    output5 = base_path / "run.log"

    output1.open("w", encoding="utf-8", newline="\n")
    output2.open("w", encoding="utf-8", newline="\n")
    output3.open("w", encoding="utf-8", newline="\n")
    output4.open("w", encoding="utf-8", newline="\n")
    with output5.open("w", encoding="utf-8", newline="\n") as file:
            file.write(
                "step\t"
                "time\t"
                "total_energy\t"
                "potential_energy\t"
                "kinetic_energy\t"
                "temperature\n"
            )

    """建立初始近邻表"""
    build_neighbor_list(atoms, parameters)

    """计算第0步"""
    potential_0, forces, virial_tensor = calculate_lj(atoms, parameters)
    stress_tensor = lj_stress_tensor(atoms, parameters, virial_tensor)
    kinetic_energy, temp = Ekinetic_and_T(atoms,parameters)

    output_file(
        atoms,
        forces,
        stress_tensor,
        0,
        0,
        potential_0,
        kinetic_energy,
        temp,
        output1,
        output2,
        output3,
        output4,
        output5,
    )


    """计算后续时间步"""
    for i in range(parameters.nstep):
        step = i + 1
        potential_energy, forces_new, stress_tensor = (
            velocity_verlet_step(atoms, parameters, forces, step)
        )
        forces = forces_new
        kinetic_energy, temp = Ekinetic_and_T(atoms,parameters)
        if step % parameters.step_out == 0:
            output_file(
                atoms,
                forces_new,
                stress_tensor,
                step,
                step * parameters.dt,
                potential_energy,
                kinetic_energy,
                temp,
                output1,
                output2,
                output3,
                output4,
                output5,
            )


if __name__ == "__main__":
    main() 