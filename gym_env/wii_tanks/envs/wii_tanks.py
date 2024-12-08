import math
import main as main

import gymnasium as gym
from gymnasium import spaces

MAIN_TANK = 0

MOVE_TOP = 5
MOVE_RIGHT = 7
MOVE_BOTTOM = 3
MOVE_LEFT = 1
MOVE_TOP_RIGHT = 8
MOVE_BOTTOM_RIGHT = 6
MOVE_TOP_LEFT = 2
MOVE_BOTTOM_LEFT = 0


class WiiTanks(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}
    
    def __init__(self, render_mode=None, size=10, level=1):
        self.size = size
        self.level = level
        # TODO: map better
        # Assuming number of targets == number of levels
        self.num_tanks = level + 1

        self.observation_space = spaces.Dict(
            {
                 f"tank[i]": spaces.Box(0, size - 1, shape=(2,), dtype=int)
                 for i in range(self.num_tanks)
            }
            #{
            #    "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            #    # Need to make num of targets dependent on the level
            #    "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            #    # Need to add observations for obstacles
            #}
        )
        
        # TODO: add 
        # One for each of 8 directions + one for not moving
        movement_actions = 9
        # One for each of 8 directions + one for not shooting
        aim_actions = 9
        self.action_space =  spaces.MultiDiscrete(
            [movement_actions, aim_actions]
        )

        # TODO: something similar to their action to distribution thing

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    # For calculations we want to access often
    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def _get_tanks_locations(self):
        return [np.array([tank.x, tank.y]) for tank in main.tanks]

    def reset(self, seed=None, options=None):
        super().reset()

        # Maybe just call his functions here
        # TODO: import function
        # TODO: pass appropriate stage
        #main.goto_stage()
        main.assets = {}
        main.assets.update(load_tiles())
        main.assets.update(load_effects())
        main.assets.update(load_tanks())
        main.assets.update(load_extra())

        main.lives = PLAYER_LIVES

        main.redraw_bg = render_bg_layer
        main.redraw_shadow = render_shadow_layer
        main.draw_to_bg = draw_to_bg

        transition_surf = Surface((SCREENWIDTH, SCREENHEIGHT))
        transition_surf.fill((0, 0, 0))

        transition_counter = -1

        self._tanks_locations = self._get_tanks_locations()

        ## TODO: import main
        #agent_tank = main.tanks[0]
        #assert agent_tank.type == "white"
        ## TODO: see if we need to quantize location
        ##     I think we do, check classes, he has a bunch of + 0.5
        ##     maybe changing that would help?
        #self._agent_location = np.array([agent_tank.x, agent_tank.y])


        # TODO: get the parsed state of the map
#        map_dir = make_map_1()
#        self.map = [
#            [None for y in range(FIELDHEIGHT)] for x in range(FIELDWIDTH)
#        ]
#        self.tanks = [Tank("white", 0, 0, no_ai=True, stop=True)]
#        self.effects = []
#        self.mines = []
#        self.bullets = []
#        # TODO: import `load_stage()`
#        self.map, self.tanks = load_stage(map_dir, self.map, self.tanks)
#
#        if self.render_mode is not None:
#            # TODO: import the function
#            render_bg_layer()
#            render_shadow_layer()
#
#        # TODO: delete if not necessary
#        text_cache.clear()

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def _action_to_mouse_position(shoot_action):
        """
        From shoot action get the appropriate value for mouse position. 
        Returns positive value and needs to be adjusted.
        """
        return max(0.125 * (shoot_action % 4), 0.5)

    def _action_to_movement(move_action):
        if move_action % 2 == 1:
            if move_action > 4:
                speed = main.tanks[0].speed
            if move_action < 4:
                speed = -1 * main.tanks[0].speed

            if move_action == MOVE_TOP and move_action == MOVE_BOTTOM:
                main.tanks[MAIN_TANK].move_x(speed)
            if move_action == MOVE_RIGHT and move_action == MOVE_LEFT:
                main.tanks[MAIN_TANK].move_y(speed)

        elif move_action % 2 == 0:
            speed = main.tanks[0].speed / math.sqrt(2)
            if move_action == MOVE_TOP_RIGHT:
                main.tanks[MAIN_TANK].move_x(speed)
                main.tanks[MAIN_TANK].move_y(speed)
            if move_action == MOVE_BOTTOM_RIGHT:
                main.tanks[MAIN_TANK].move_x(speed)
                main.tanks[MAIN_TANK].move_y(-1 * speed)
            if move_action == MOVE_TOP_LEFT:
                main.tanks[MAIN_TANK].move_x(-1 * speed)
                main.tanks[MAIN_TANK].move_y(speed)
            if move_action == MOVE_BOTTOM_LEFT:
                main.tanks[MAIN_TANK].move_x(-1 * speed)
                main.tanks[MAIN_TANK].move_y(-1 * speed)

    def step(self, action):
        # TODO: make the action take effect by editing the agent's location
        # TODO: use clip to make sure we don't leave the map
        # TODO: add terminated based on whether the enemy is dead
        # TODO: calculated reward

        move_action, shoot_action = action

        # Handle movement
        if move_action != 4:
            self._action_to_movement(move_action)

        # Handle shooting
        if aim_action != 0:
            negate_aim = False if aim_action < 4 else True
            new_aim = (
                (-1 if negate_aim else 1)
                * self._action_to_mouse_position(aim_action)
            )
            main.tanks[MAIN_TANK].aim_target = new_aim

        # TODO: write a function to get the location of tanks and call it here
        # and other places
        # < 0: top; > 0: bottom; right: 0; left: +- 0.5; top: -0.25; bottim: 0.25

        obseration = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, False, info

    def render(self):
        # TODO: implement
        return

    def _render_frame(self):
        # TODO: implement (maybe just call appropriate function)
        return

    def close(self):
        # TODO: close game
        return
