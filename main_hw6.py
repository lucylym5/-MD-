from dataclasses import replace
from pathlib import Path

import numpy as np

from classes import Atomsystem, MDParameters            # 导入参数类和原子体系类
from read import read_geometry, read_input              # 导入读取函数
from neighbor_list import build_neighbor_list           # 建立初始近邻表
from potential import calculate_lj, lj_stress_tensor    # 计算第0步物理量
from dynamics import initialize_velocities, velocity_verlet_step, Ekinetic_and_T
from thermotstat import berendsen_thermotstat, andersen_thermotstat

import output_hw6 as op
import rdf


SEED = 2026

def main():
    base_path = Path(__file__).resolve().parent
    input_file = base_path / "input_rdf.in"
    parameters_dict = read_input(input_file)
    parameters = MDParameters.from_input(parameters_dict)

    # 初始化粒子构型
    geo_file = base_path / parameters.geo_dir
    cell, position_table, velocity_table = read_geometry(
        geo_file, 
        parameters.read_velocity,
    )
    atoms = Atomsystem.from_geometry(
        cell,
        position_table,
        velocity_table,
        parameters.natom,
        parameters.neighbor_n,
    )
    if parameters.read_velocity == "n":
        initialize_velocities(
            atoms,
            parameters,
            SEED,
        ) 
    rdf_state = rdf.initialize_rdf(parameters)

    rng = np.random.default_rng(SEED)

    # 结果文件夹
    result_dir = base_path / f"result_hw6"
    log_path = result_dir / f"run.log"
    fig_path = result_dir / f"rdf.png"
    result_dir.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8", newline="\n") as file:
        file.write(
            "step\t"
            "time\t"
            "total_energy\t"
            "potential_energy\t"
            "kinetic_energy\t"
            "temperature\n"
        )

    # 建立初始近邻表，计算并输出第 0 步。
    build_neighbor_list(atoms, parameters)
    potential_0, forces, virial_tensor = calculate_lj(atoms, parameters)
    stress_tensor = lj_stress_tensor(atoms, parameters, virial_tensor)
    kinetic_energy, temp = Ekinetic_and_T(atoms, parameters)

    op.output_log(
        0,
        0,
        potential_0,
        kinetic_energy,
        temp,
        log_path,
    )

    # 第1步及之后推进
    for step in range(1, parameters.nstep + 1):
        potential_energy, forces, stress_tensor = velocity_verlet_step(
            atoms, parameters, forces, step
        )
        kinetic_energy, temp = Ekinetic_and_T(atoms, parameters)

        if step % parameters.step_out == 0:
            op.output_log(
                step,
                step * parameters.dt,
                potential_energy,
                kinetic_energy,
                temp,
                log_path,
            )

        if parameters.thermotstat == 1:
            berendsen_thermotstat(atoms, parameters, temp)
        else:
            andersen_thermotstat(atoms, parameters, rng)

        if step >= parameters.rdf_start and step <= parameters.rdf_stop:
            rdf.accumulate_rdf(rdf_state, atoms, parameters)

    r, g_r = rdf.finalize_rdf(rdf_state, atoms)
    ref_file = base_path / f"ref_RDF.txt"
    op.plot_rdf_profile(r, g_r, ref_file, fig_path)


if __name__ == "__main__":
    main()
