from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np

from classes import Atomsystem, MDParameters  


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



def output_log(
        atoms: Atomsystem,
        forces: np.ndarray,
        stress_tensor: np.ndarray,
        step: int,
        time: float,
        potential_energy: float,
        kinetic_energy: float,
        temperature: float,
        output_log: str | Path,
):
    total_energy = (
        potential_energy + kinetic_energy
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


def plot_temperatureTime(
        output_log: Path,
        save_fig: Path,
        parameters: MDParameters,
        label: str,
):
    data = np.loadtxt(output_log, skiprows=1)

    time_ps = data[:, 1]
    temperature = data[:, 5]

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(
        time_ps,
        temperature,
        linewidth=1.5,
        label=label,
    )

    ax.axhline(
        parameters.ensamble_T,
        color="black",
        linestyle="--",
        linewidth=1,
        label=f"Ensamble temperature: {parameters.ensamble_T:g} K",
    )

    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Temperature (K)")
    ax.set_title("Temperature vs. time")
    ax.legend()
    ax.grid(alpha=0.25)

    # 4. 调整布局并保存图片
    fig.tight_layout()
    fig.savefig(save_fig, dpi=300)
    plt.close(fig)