from pathlib import Path
from io import StringIO
import pandas as pd
import numpy as np

def read_geometry(geo_file: str | Path)-> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    geo_file = Path(geo_file)
    
    # （按行）读取文件为列表
    lines = geo_file.read_text(encoding="utf-8").splitlines()

    # 找到盒子参数
    cell_start = next(
        i+1
        for i, line in enumerate(lines)
        if "CELL_PARAMETER" in line.upper()
    )
    # 创建列表储存盒子参数
    cell_read: list[list[float]] = []
    # 盒子参数读取
    for i in range(cell_start, cell_start+3):
        values = lines[i].split()                   # 每行读取为一个str列表
        cell_read.append([float(value) for value in values])
    
    cell = np.asarray(cell_read, dtype=np.float64)  # 列表嵌套 → 浮点数组

    # 找到位置参数
    position_start = next(
        i+1
        for i, line in enumerate(lines)
        if "%ATOMIC_POSTION" in line.upper()
    )
    position_end = next(
        i
        for i in range(position_start, len(lines)+1)
        if lines[i].strip() == ""
    )
    # 提取位置坐标
    atomic_position = "\n".join(lines[position_start:position_end])  # 使用换行符重新连接这些行
    position = pd.read_csv(
        StringIO(atomic_position),
        sep = "\t",
        header = None,
        names = ["element","x","y","z"]
    )

    # 找到速度参数
    velocity_start = next(
        i+1
        for i, line in enumerate(lines)
        if "%ATOMIC_VELOCITY" in line.upper()
    )
    velocity_end = next(
        i
        for i in range(velocity_start, len(lines)+1)
        if lines[i].strip() == ""
    )
    # 提取速度
    atomic_velocity = "\n".join(lines[velocity_start:velocity_end])  # 使用换行符重新连接这些行
    velocity = pd.read_csv(
        StringIO(atomic_velocity),
        sep = "\t",
        header = None,
        names = ["element","x","y","z"]
    )

    return cell, position, velocity

def read_input(input_file: Path | str) -> dict[str, int | float | str]:
    inputfile = Path(input_file)

    lines = inputfile.read_text(encoding="utf-8").splitlines()
    parameters: dict[str, int | float | str] = {}

    for index, line in enumerate(lines):    # enumerate默认从0开始计数
        content = line.strip()              # strip去除行首、行尾空格
        if content == "":
            continue
        params = content.split()
        key, value_o = params      # 提取每行的参数名和对应值
        # 尝试将参数转换为整数/浮点数
        try:
            value = int(value_o)
        except ValueError:
            try:
                value = float(value_o)
            except ValueError:
                value = value_o
        parameters[key] = value    # 将参数名和对应值存入字典
    return parameters