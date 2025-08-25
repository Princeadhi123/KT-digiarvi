"""
Curriculum sequencing RL package.

This package supports both package and script-mode imports via try/except
relative-import fallbacks inside modules like `dqn.py`, `a3c.py`, `ppo.py`,
and `q_learning.py`.
"""

__all__ = [
    "env",
    "evaluation",
    "q_learning",
    "dqn",
    "a2c",
    "a3c",
    "ppo",
    "curriculum_rl_experiments",
]
