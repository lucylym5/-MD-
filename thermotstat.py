import numpy as np
from classes import Atomsystem, MDParameters

from dynamics import BOLTZMANN_CONSTANT, UNIT_KINETIC_TO_EV

"""根据当前温度，缩放 atoms.velocities""" 
def berendsen_thermostat(atoms, parameters, current_temperature):
    dt = parameters.dt
    tau = parameters.tau
    target_T = parameters.ensamble_T

    if dt <= 0 or tau < dt:
        raise ValueError("需满足 dt > 0 且 tau >= dt")
    
    lambda_square = 1 + dt / tau * (target_T/current_temperature- 1)
    atoms.velocities *= np.sqrt(lambda_square)


"""用 Box–Muller 变换生成高斯随机数"""
def gaussian_random(
    rng, 
    mean = 0, 
    std = 1, 
    size = None
):
    if std < 0:
        raise ValueError("标准差不能为负")

    u1 = 1 - rng.random(size=size)    # (0, 1] 均匀分布
    u2 = rng.random(size=size)        # [0, 1) 均匀分布
    
    z = (
            np.sqrt(-2.0 * np.log(u1))
            * np.cos(2.0 * np.pi * u2)
        )
    
    return mean + std * z


"""随机选择发生碰撞的原子，原地更新其速度"""
def andersen_thermostat(atoms, parameters, rng):
    # nraise：每个原子两次碰撞间平均步数  p_collision：每一步碰撞概率
    p_collision = 1 / parameters.nraise     

    # 热库温度对应的 速度方差
    sigma_v = np.sqrt(
            BOLTZMANN_CONSTANT * parameters.ensamble_T
            / (parameters.mass * UNIT_KINETIC_TO_EV)
        )

    # 决定一个原子是否发生碰撞，如碰撞，更新速度
    for i in range(atoms.natom):    
        if rng.random() < p_collision:
            atoms.velocities[i, :] = gaussian_random(
                rng, mean=0.0, std=sigma_v, size=3
            )

    


