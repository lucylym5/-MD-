from dataclasses import replace
from pathlib import Path

import numpy as np

from classes import Atomsystem, MDParameters            # 导入参数类和原子体系类
from read import read_geometry, read_input              # 导入读取函数
from neighbor_list import build_neighbor_list           # 建立初始近邻表
from potential import calculate_lj, lj_stress_tensor    # 计算第0步物理量
from dynamics import initialize_velocities, velocity_verlet_step, Ekinetic_and_T
from thermotstat import berendsen_thermotstat, andersen_thermotstat

import output as op

SEED = 2026

def main():
    base_path = Path(__file__).resolve().parent
    input_file = base_path / "input_rdf.in"
    parameters_dict = read_input(input_file)
    base_parameters = MDParameters.from_input(parameters_dict)

    for thermotstat_id, name in ((1, "berendsen"), (2, "andersen")):
        # 参数类是 frozen dataclass，用 replace 创建本次模拟的独立参数。
        parameters = replace(base_parameters, thermotstat=thermotstat_id)

        # 每次重新读取初始构型和速度
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

        rng = np.random.default_rng(SEED)

        # 两种热库的结果分别保存到项目根目录下的不同文件夹。
        result_dir = base_path / f"result_{name}"
        result_dir.mkdir(parents=True, exist_ok=True)
        log_path = result_dir / "run.log"
        fig_path = result_dir / f"temperature_{name}.png"

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
            atoms,
            forces,
            stress_tensor,
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
                    atoms,
                    forces,
                    stress_tensor,
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

        if parameters.thermotstat == 1:
            label = fr"Berendsen, $\tau={parameters.tau:g}~\mathrm{{ps}}$"
        else:
            label = f"Andersen, nraise = {parameters.nraise}"

        op.plot_temperatureTime(log_path, fig_path, parameters, label)
        print(f"{name.capitalize()} completed: {result_dir}")


if __name__ == "__main__":
    main()
