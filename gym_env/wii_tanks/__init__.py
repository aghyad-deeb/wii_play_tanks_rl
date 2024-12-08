from gymnasium.envs.registration import register

register(
    id="wii_tanks/GridWorld-v0",
    entry_point="wii_tanks.envs:GridWorldEnv",
)
