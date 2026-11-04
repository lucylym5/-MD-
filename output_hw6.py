from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np

from classes import Atomsystem, MDParameters  


def output_log(
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


def plot_rdf_profile(
        r: np.ndarray,
        g_r: np.ndarray,
        ref_file: Path,
        save_fig: Path,
):

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(
        r,
        g_r,
        linewidth=1.5,
        label="MD",
    )

    # 参考RDF绘制
    ref_data = np.loadtxt(ref_file)
    ref_r = ref_data[:, 0]
    ref_g_r = ref_data[:, 1]

    ax.plot(
        ref_r,
        ref_g_r,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Reference",
    )

    ax.set_xlabel(fr"r \, (\text{{\AA}})")
    ax.set_ylabel(fr"g(r)")
    ax.set_title("RDF, Considering Berendsen Thermostat")
    ax.legend()
    ax.grid(alpha=0.25)

    # 4. 调整布局并保存图片
    fig.tight_layout()
    fig.savefig(save_fig, dpi=300)
    plt.close(fig)