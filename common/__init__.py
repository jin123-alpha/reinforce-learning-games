"""
common/ —— 与游戏、与 RL 算法都无关的公共工具。

包含：
    paths.py   全项目唯一的路径解析入口（按游戏名分目录）
    cli.py     四个入口脚本共享的 CLI 工具（KEY=VALUE 解析、配置加载）

依赖约束：本包**只允许**依赖 Python 标准库。
不得 import numpy / torch / stable_baselines3 / pygame,
否则 human_play.py「只依赖 games」的架构保证就断了。
"""
from __future__ import annotations
