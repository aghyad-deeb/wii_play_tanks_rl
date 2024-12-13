import numpy as np
import os, sys

os.environ['SDL_VIDEO_CENTERED'] = "1"
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"

from settings import *
from classes import *
from loader import *
import training_map
import main, sound, ai_math

from pygame import *
import pygame.freetype
from math import *

#from gym_main.env.wii_tanks.envs.wii_tanks import WiiTanks

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

seed = 42
init()

# main.TRAINING = False
main.TRAINING = True

if not main.TRAINING:
	screen = display.set_mode((SCREENWIDTH, SCREENHEIGHT))
	display.set_caption("Python Play Tanks!")
	display.set_icon(load_icon())
	secret_font = pygame.freetype.Font("assets/fonts/TopSecret.ttf", 80)
	type_font = pygame.freetype.Font("assets/fonts/TravelingTypewriter.ttf", 80)
	pixel_font = pygame.freetype.Font("assets/fonts/PressStart2P.ttf", 24)

	pixel_font.render_to(screen, (10, SCREENHEIGHT - 30), "Loading..." + (" (SOUND MUTED)" if SOUNDMUTED else ""), (255, 255, 255))
	display.flip()


clock = time.Clock()


text_cache = {}
mission_completed = False

aim_thresholds = [0.125, 0.25, 0.375, 0.5]
num_aim_thresholds = len(aim_thresholds)
def aim_threshold_round(aim):
	for i, thresh in enumerate(aim_thresholds):
		if aim < thresh:
			return i + 1

def get_aim():
	return atan2(mouse.get_pos()[1]/3 - (main.tanks[0].y+1)*SCALE, mouse.get_pos()[0]/4 - (main.tanks[0].x+0.5)*SCALE) / pi / 2

def aim_to_action(aim):
	"""
	This defines action 1 to 4 to be each eight of the bottom half 
	going clock wise and 5 to 8 for the top half going
	counterclockwise 
	"""
	thresh = aim_threshold_round(abs(aim))
	#print(f"{thresh=}")
	if aim > 0:
		shoot_action = thresh
	else:
		shoot_action = num_aim_thresholds + thresh
	return shoot_action

def keys_to_movement(d, s, a, w):
	"""
	 Creates the following mapping
		 right: 7; top: 5; left: 1; bottom: 3; 
		 top-right: 8; top-left: 2; bottom-left: 0; bottom-right: 6;
	"""
	#print(f"{d=}, {s=}, {a=}, {w=}")
	# Start with no movement
	action = 4
	if w: action += 1
	if s: action -= 1
	if d: action += 3
	if a: action -= 3
	return action

def goto_stage(stage="1"):
	main.map = [[None for y in range(FIELDHEIGHT)] for x in range(FIELDWIDTH)]
	main.tanks = [Tank("white", 0, 0, no_ai=True, stop=True)]
	main.effects = []
	main.mines = []
	main.bullets = []
	main.stage = stage
	main.map, main.tanks = load_stage(main.stage, main.map, main.tanks)
	
	render_bg_layer()
	render_shadow_layer()
	
	global round_counter, intro_counter, win_counter, lose_counter, over_counter
	round_counter = 0
	intro_counter = 0
	win_counter = -1
	lose_counter = -1
	over_counter = -1
	
	text_cache.clear()
	
	##sound.play_mission_intro()

def render_bg_layer():
	if main.TRAINING:
		return
	global bg_layer
	bg_layer = Surface((SCREENWIDTH, SCREENHEIGHT))
	
	for x in range(0, SCREENWIDTH, SCALEX):
		for y in range(0, SCREENHEIGHT, SCALEY):
			bg_layer.blit(main.assets["floor"], (x, y))

def render_shadow_layer():
	if main.TRAINING:
		return
	global shadow_layer
	shadow_layer = Surface((SCREENWIDTH, SCREENHEIGHT), SRCALPHA)
	
	for x in range(FIELDWIDTH):
		for y in range(FIELDHEIGHT):
			if not main.map[x][y] == None:
				shadow_layer.blit(main.assets["wall_shadow"], (int((x-0.5) * SCALEX), int((y+0.5) * SCALEY)))
	
	shadow_layer.fill((255, 255, 255, 120), special_flags=BLEND_RGBA_MULT)

def draw_to_bg(img, x, y):
	if main.TRAINING:
		return
	global bg_layer
	bg_layer.blit(img, (x * SCALEX - img.get_width()//2, y * SCALEY - img.get_height()//2))

def draw_text(font, text, x, y, centered=False, border=2, color=(255, 255, 255), rotation=0):
	if main.TRAINING:
		return
	global screen
	global text_cache
	rect = font.get_rect(text, rotation=rotation)
	
	if text.strip() == "":
		return
	
	if (text, border, color, rotation) not in text_cache:
		surf = Surface((rect[2]+border*2, rect[3]+border*2), pygame.SRCALPHA)
		for i in range(-border, border+1):
			for j in range(-border, border+1):
				font.render_to(surf, (i + border, j + border), None, (0, 0, 0), rotation=rotation)
		font.render_to(surf, (border, border), None, color, rotation=rotation)
		text_cache[(text, border, color, rotation)] = surf
	screen.blit(text_cache[(text, border, color, rotation)], (x - (rect[2]/2 if centered else 0), y - (rect[3]/2 if centered else 0)))



main.tanks = []

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
    
    def __init__(self, render_mode=None, width=FIELDWIDTH, height=FIELDHEIGHT, level=1):
        self.width = width
        self.height = height
        self.level = level
        # TODO: map better
        # Assuming number of targets == number of levels
        self.num_tanks = level + 1

        self.observation_space = spaces.Dict(
        	{
        		"agent": spaces.Box(low=np.array([0, 0]), high=np.array([width, height]), shape=(2,), dtype=int),
        		"targets": spaces.Sequence(spaces.Box(low=np.array([0, 0]), high=np.array([width, height]), shape=(2,), dtype=int), seed=seed),
        		"bullets": spaces.Sequence(spaces.Box(low=np.array([0, 0, -0.5, 0.1]), high=np.array([width, height, 0.5, 0.2001]), shape=(4,), dtype=float), seed=seed),
        		"walls": spaces.Sequence(spaces.Box(low=np.array([0, 0]), high=np.array([width, height]), shape=(2,), dtype=int), seed=seed),
        		"map": spaces.Box(low=np.array([0, 0]), high=np.array([width, height]), shape=(2,), dtype=int),
        	},
        	seed=seed,
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
        shoot_action = 9
        self.action_space =  spaces.MultiDiscrete(
        	[movement_actions, shoot_action]
        )


        # TODO: something similar to their action to distribution thing

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def _get_obs(self):
        return {
				"agent": self._agent_location,
                "targets": self._targets_locations,
                "bullets": self._bullets_locations,
				"walls": self._walls_locations,
				"map": [[1 if tile is not None else 0 for tile in lst] for lst in main.map],
        }

    # For calculations we want to access often
#    def _get_info(self):
#        return {
#            "distance": np.linalg.norm(
#                self._agent_location - self._target_location, ord=1
#            )
#        }

    def _get_tanks_locations(self):
        return (
			np.array((main.tanks[0].x, main.tanks[0].y)),
			[np.array([tank.x, tank.y]) for tank in main.tanks[1:]],
		)

    def _get_num_alive_tanks(self):
        # num = [1 for tank in main.tanks if not tank.dead]
        # print(f"{num=}, {sum(num)-1=}")
        return sum([1 for tank in main.tanks if not tank.dead]) - 1

    def _get_bullets_locations(self):
        return [
			(
				np.array([bullet.x, bullet.y, bullet.dir, bullet.speed]),
				bullet.shot_by_agent
			) for bullet in main.bullets]

    def _get_walls_locations(self):
        walls_locations = []
        for j, col in enumerate(main.map):
            for i, tile in enumerate(col):
                if tile is not None and ("wall0" in tile or "dwall" in tile):
                    walls_locations.append((i, j))
        return walls_locations

#    def _get_dwalls_locations(self):
#        dwalls_locations = []
#        for j, col in enumerate(main.map):
#            for i, tile in enumerate(col):
#                if tile is not None and "dwall1" in tile:
#                    dwalls_locations.append((i, j))
#        return dwalls_locations
	 
    def reset(self, seed=None, options=None, level=1):
        super().reset()
        if not main.TRAINING:
            main.assets = {}
            main.assets.update(load_tiles())
            main.assets.update(load_effects())
            main.assets.update(load_tanks())
            main.assets.update(load_extra())

            main.redraw_bg = render_bg_layer
            main.redraw_shadow = render_shadow_layer
            main.draw_to_bg = draw_to_bg

            transition_surf = Surface((SCREENWIDTH, SCREENHEIGHT))
            transition_surf.fill((0, 0, 0))
		
        training_map.make_map(level, min_size=10, max_size=10, should_save_map=True)


        main.lives = PLAYER_LIVES
        transition_counter = -1
        goto_stage(stage=f"{level}")

        self._agent_location, self._targets_locations = self._get_tanks_locations()
        self._bullets_locations = self._get_bullets_locations()
        self._walls_locations = self._get_walls_locations()
        #self._dwalls_locations = self._get_dwalls_locations()
        self._num_targets_last_step = self._get_num_alive_tanks()


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
#        # TODO: delete if not necessary
#        text_cache.clear()

        observation = self._get_obs()
        #info = self._get_info()
        info = None

        return observation, info

    def _action_to_mouse_position(self, shoot_action):
        """
        From shoot action get the appropriate value for mouse position. 
        Returns positive value and needs to be adjusted.
        """
        return max(0.125 * (shoot_action % 4), 0.5)

    def _action_to_movement(self, move_action):
        if move_action % 2 == 1:
            if move_action > 4:
                speed = -1 * main.tanks[0].speed
            if move_action < 4:
                speed = main.tanks[0].speed

            if move_action == MOVE_TOP or move_action == MOVE_BOTTOM:
                main.tanks[MAIN_TANK].move_y(speed)
            if move_action == MOVE_RIGHT or move_action == MOVE_LEFT:
                main.tanks[MAIN_TANK].move_x(-1 * speed)

        elif move_action % 2 == 0:
            speed = main.tanks[MAIN_TANK].speed / math.sqrt(2)
            if move_action == MOVE_TOP_RIGHT:
                main.tanks[MAIN_TANK].move_x(speed)
                main.tanks[MAIN_TANK].move_y(-1 * speed)
            if move_action == MOVE_BOTTOM_RIGHT:
                main.tanks[MAIN_TANK].move_x(speed)
                main.tanks[MAIN_TANK].move_y(speed)
            if move_action == MOVE_TOP_LEFT:
                main.tanks[MAIN_TANK].move_x(-1 * speed)
                main.tanks[MAIN_TANK].move_y(-1 * speed)
            if move_action == MOVE_BOTTOM_LEFT:
                main.tanks[MAIN_TANK].move_x(-1 * speed)
                main.tanks[MAIN_TANK].move_y(speed)

    def _get_reward(self):
        # TODO: Implement the following rewards
        #	- Health Points
        #   - Damage to enemy
        #	- Distance from enemy bullets
        #		- Maybe simplify to distance to closes bullet

        reward = 0
        death_weight = 1
        kill_weight = 1
        distance_weight = 0.1
		# Health Points
		# 	This actually simplifies to just knowing if the tank is dead 
		# 	which is main.tanks[0].dead
        if main.tanks[MAIN_TANK].dead:
            reward -= death_weight

		# Damage to Enemy
		# 	Similarly to Health points, enemies are either dead or alive so 
		# 	this simplifies to if the number of enemies decreased
		# 	I'll introduce a variable for the initial number of enemies
        num_targets_curr = self._get_num_alive_tanks()
        reward += (num_targets_curr - self._num_targets_last_step) * kill_weight
		# Distance from closes bullet
        agent_x, agent_y = self._agent_location
        distances = [
			# Adding 0.5 to have the distance measured from the middle of the 
			# block
			(agent_x - bullet_x + 0.5)**2 + (agent_y - bullet_y + 0.5)**2
            for (bullet_x, bullet_y, _, _), shot_by_agent
			in self._bullets_locations
			if not shot_by_agent
		]
        reward -= (
			(2 - min(distances)) * distance_weight
			if len(distances) > 0  and min(distances) < 2
			else 0
		)
        return reward

    def step(self, action):
        # TODO: make the action take effect by editing the agent's location
        # TODO: use clip to make sure we don't leave the map
        # TODO: add terminated based on whether the enemy is dead
        # TODO: calculated reward

        move_action, shoot_action = action

        # Handle movement
        if move_action != 4:
            #print(f"inside move action if")
            self._action_to_movement(move_action)

        # Handle shooting
        if shoot_action != 0:
            negate_aim = False if shoot_action < 4 else True
            new_aim = (
                (-1 if negate_aim else 1)
                * self._action_to_mouse_position(shoot_action)
            )
            main.tanks[MAIN_TANK].aim_target = new_aim
            main.tanks[MAIN_TANK].shoot(shot_by_agent=True)
		
        main.tanks[MAIN_TANK].tick(round_counter if intro_counter <= -1 else (-2 if intro_counter == INTROLENGTH-1 else -1))

        # TODO: write a function to get the location of tanks and call it here
        # and other places
        # < 0: top; > 0: bottom; right: 0; left: +- 0.5; top: -0.25; bottim: 0.25
        #for i, tank in enumerate(self._tanks_locations):
        #        self._tanks_locations[i] = np.array([main.tanks[i].x, main.tanks[i].y])
        self._agent_location, self._targets_locations = self._get_tanks_locations()
        self._bullets_locations = self._get_bullets_locations()
        self._walls_locations = self._get_walls_locations()
        #self._dwalls_locations = self._get_dwalls_locations()

        run_step()
        observation = self._get_obs()
        print(f"{observation=} \n{len(observation['map'][0])=}")
        reward = self._get_reward()
        print(f"{self._get_num_alive_tanks()=}\n{self._num_targets_last_step=}")
        print(f"{reward=}")
        self._num_targets_last_step = self._get_num_alive_tanks()

        terminated = main.tanks[MAIN_TANK].dead and mission_completed
        #info = self._get_info()
        info = None

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






if not main.TRAINING:
	transition_surf = Surface((SCREENWIDTH, SCREENHEIGHT))
	transition_surf.fill((0, 0, 0))

transition_counter = -1

main.env = WiiTanks()

def run_step():
	global round_counter, intro_counter, win_counter, lose_counter, over_counter
	global transition_counter
	move_action = 4
	shoot_action = 0
	for e in event.get():
		if e.type == QUIT:
			stop = True
		if e.type == MOUSEBUTTONDOWN:
			# New code
			aim = get_aim()
			aim_to_action
			shoot_action = aim_to_action(aim)
			#print(f"{aim=}, {shoot_action=}")
			# TODO: Add shooting
			# Old code
			#main.tanks[0].shoot()
		if e.type == KEYDOWN:
			if e.key == K_SPACE:
				main.tanks[MAIN_TANK].place_mine()
		if e.type in sound.end_event:
			pass
			#sound.end_event[e.type]()
	
	if not main.TRAINING:
		screen.blit(bg_layer, (0, 0))
		screen.blit(shadow_layer, (0, 0))
	
	win_test = True
	for t in main.tanks[1:]:
		if not t.dead:
			win_test = False
			break
	if win_counter <= -1 and lose_counter <= -1 and intro_counter <= -1 and over_counter <= -1:
		round_counter += 1
		if (main.tanks[MAIN_TANK].dead and main.lives <= 0) or (win_test and STAGES.index(main.stage) == len(STAGES)-1):
			for t in main.tanks:
				t.stop = True
			#sound.stop_mission_loop()
			#sound.play_results()
			over_counter = 0
		elif main.tanks[MAIN_TANK].dead:
			#sound.play_mission_lose()
			lose_counter = LOSELENGTH
		elif win_test:
			main.tanks[MAIN_TANK].stop = True
			##sound.play_mission_win()
			win_counter = WINLENGTH
	if lose_counter > 0:
		lose_counter -= 1
	if win_counter > 0:
		win_counter -= 1
	if over_counter >= 0:
		over_counter += 1
	if (lose_counter == 0 or win_counter == 0 or over_counter >= 0) and transition_counter == -1:
		transition_counter = 0
	if transition_counter >= 0 and transition_counter < TRANSITIONLENGTH*2:
		transition_counter += 1
		if transition_counter == TRANSITIONLENGTH:
			if win_counter == 0:
				mission_completed = True
				stop = True
				#goto_stage(STAGES[STAGES.index(main.stage)+1])
			else:
				mission_completed = True
				main.lives -= 1
				stop = True
				#goto_stage(main.stage)
	if transition_counter >= TRANSITIONLENGTH*2:
		transition_counter = -1
	if over_counter >= 0 and transition_counter / TRANSITIONLENGTH >= 0.5:
		transition_counter = int(TRANSITIONLENGTH * 0.5)
	
	if intro_counter > 0:
		intro_counter -= 1
	elif intro_counter == 0:
		intro_counter = -1
		round_counter = 0
		##sound.build_mission_loop()
		##sound.play_mission_loop()
		for t in main.tanks:
			t.stop = False
		text_cache.clear()
	
	new_aim = get_aim()
	if new_aim != main.tanks[MAIN_TANK].aim_target:
		main.tanks[MAIN_TANK].aim_target = new_aim
	aim_to_action
	
	pressed = key.get_pressed()
	
	# New code
	move_action = keys_to_movement(
		pressed[K_d], pressed[K_s], pressed[K_a], pressed[K_w]
	)
	#print(f"{move_action=}")

	# Old code
	#movex = 0
	#if pressed[K_a]:
	#	movex = -main.tanks[0].speed
	#if pressed[K_d]:
	#	movex = main.tanks[0].speed
	#
	#movey = 0
	#if pressed[K_w]:
	#	movey = -main.tanks[0].speed
	#if pressed[K_s]:
	#	movey = main.tanks[0].speed
	#
	#if movex != 0 and movey != 0:
	#	main.tanks[0].move_x(movex / sqrt(2))
	#	main.tanks[0].move_y(movey / sqrt(2))
	#elif movex != 0:
	#	main.tanks[0].move_x(movex)
	#elif movey != 0:
	#	main.tanks[0].move_y(movey)
	
	if DEBUGMODE:
		ai_math.screen = screen
		ai_math.display = display
		ai_math.draw = draw
		ai_math.event = event
	
	ai.map_out_tanks()
	
	for t in main.tanks:
		t.tick(round_counter if intro_counter <= -1 else (-2 if intro_counter == INTROLENGTH-1 else -1))
		if t.track_counter >= 1:
			t.track_counter -= 1
			if not main.TRAINING:
				bg_layer.blit(main.assets["tracks"], ((t.x+0.5)*SCALEX - (TRACKTILEX//2), (t.y+1.5)*SCALEY - (TRACKTILEY//2)), Rect(t.get_body_sheet()[0]*TRACKTILEX, t.get_body_sheet()[1]*TRACKTILEY, TRACKTILEX, TRACKTILEY))
	
	for m in main.mines[:]:
		m.tick()
		if m.fuse < 0:
			screen.blit(main.assets["mine_unarmed"], ((m.x) * SCALEX, (m.y) * SCALEY))
		else:
			if DEBUGMODE:
				draw.ellipse(screen, (255, 0, 0) if m.detect else (0, 0, 255), ((m.x+0.5-MINERANGE) * SCALEX, (m.y+0.5-MINERANGE) * SCALEY, MINERANGE*2 * SCALEX, MINERANGE*2 * SCALEY), 10)
			screen.blit(main.assets["mine" if int(log(FUSELENGTH-m.fuse+10) * 10) % 2 == 1 else "mine_lit"], ((m.x) * SCALEX, (m.y) * SCALEY))
	
	for b in main.bullets:
		b.tick()
	
	for y in range(FIELDHEIGHT):
		if main.TRAINING:
			break
		for t in main.tanks:
			if int(t.y+0.5) == y and not t.dead:
				screen.blit(main.assets["tank_body_outline"], ((t.x+0.5)*SCALEX - (BODYTILEX//2), (t.y+1.5)*SCALEY - (BODYTILEY//2)), Rect(t.get_body_sheet()[0]*BODYTILEX, t.get_body_sheet()[1]*BODYTILEY, BODYTILEX, BODYTILEY))
				screen.blit(main.assets["tank_head_outline"], ((t.x+0.5)*SCALEX - (HEADTILEX//2), (t.y+1.5)*SCALEY - (HEADTILEY//2)), Rect(t.get_aim_sheet()[0]*HEADTILEX, t.get_aim_sheet()[1]*HEADTILEY, HEADTILEX, HEADTILEY))
		for t in main.tanks:
			if int(t.y+0.5) == y and not t.dead:
				screen.blit(main.assets["tank_" + t.type + "_body"], ((t.x+0.5)*SCALEX - (BODYTILEX//2), (t.y+1.5)*SCALEY - (BODYTILEY//2)), Rect(t.get_body_sheet()[0]*BODYTILEX, t.get_body_sheet()[1]*BODYTILEY, BODYTILEX, BODYTILEY))
		for t in main.tanks:
			if int(t.y+0.5) == y and not t.dead:
				screen.blit(main.assets["tank_" + t.type + "_head"], ((t.x+0.5)*SCALEX - (HEADTILEX//2), (t.y+1.5)*SCALEY - (HEADTILEY//2)), Rect(t.get_aim_sheet()[0]*HEADTILEX, t.get_aim_sheet()[1]*HEADTILEY, HEADTILEX, HEADTILEY))
				
		
		for b in main.bullets:
			if int(b.y+0.5) == y:
				screen.blit(main.assets["rocket"] if b.rocket else main.assets["bullet"], ((b.x+0.5)*SCALEX - (BULLETTILEX//2), (b.y+1.5)*SCALEY - (BULLETTILEY//2)), Rect(b.get_dir_sheet()[0]*BULLETTILEX, b.get_dir_sheet()[1]*BULLETTILEY, BULLETTILEX, BULLETTILEY))
		
		for e in main.effects:
			if int(e.y) == y and not e.top and e.frame >= 0:
				screen.blit(main.assets[e.name], (e.x * SCALEX - (EFFECTSCALE//2), e.y * SCALEY - (EFFECTSCALE//2)), Rect(int(e.frame) * EFFECTSCALE, 0, EFFECTSCALE, EFFECTSCALE))
		
		for x in range(FIELDWIDTH):
			if not main.map[x][y] == None:
				screen.blit(main.assets[main.map[x][y]], (x * SCALEX, (y+2) * SCALEY - main.assets[main.map[x][y]].get_height()))
	
	for e in main.effects[:]:
		if main.TRAINING:
			break
		if e.top and e.frame >= 0:
			screen.blit(main.assets[e.name], (e.x * SCALEX - (EFFECTSCALE//2), e.y * SCALEY - (EFFECTSCALE//2)), Rect(int(e.frame) * EFFECTSCALE, 0, EFFECTSCALE, EFFECTSCALE))
		if int(e.frame) * EFFECTSCALE >= main.assets[e.name].get_width():
			main.effects.remove(e)
		e.frame += EFFECTSPEED
	
	if DEBUGMODE:
		for t in main.tanks:
			draw.line(screen, (0, 255, 0), ((t.x + 0.5) * SCALEX, (t.y + 1) * SCALEY), ((t.x + 0.5 + cos(t.aim_target*2*pi)) * SCALEX , (t.y + 1 + sin(t.aim_target*2*pi)) * SCALEY), 4)
			if t.ai_has_target:
				draw.line(screen, (255, 0, 0), ((t.x + 0.5) * SCALEX, (t.y + 1) * SCALEY), ((t.x + 0.5 + cos(t.aim_target*2*pi)) * SCALEX , (t.y + 1 + sin(t.aim_target*2*pi)) * SCALEY), 4)
			tar_list = [(t.x, t.y)] + t.ai_move_targets
			for i, tar in enumerate(tar_list):
				if i > 0:
					draw.line(screen, (0, 100, 255), ((tar[0]+0.5) * SCALEX, (tar[1]+1.5) * SCALEY), ((tar_list[i-1][0]+0.5) * SCALEX, (tar_list[i-1][1]+1.5) * SCALEY), 2)
					draw.rect(screen, (0, 255, 255), ((tar[0]+0.5) * SCALEX-4, (tar[1]+1.5) * SCALEY-4, 8, 8))
	
	if not main.TRAINING:
		draw_text(type_font, f"MISSION {main.stage} "[round_counter:min((-intro_counter + 150)//5, -1)], SCREENWIDTH/2, SCREENHEIGHT/2-90, True)
		draw_text(type_font, f"ENEMY TANKS: {len(main.tanks)-1} "[round_counter:min((-intro_counter + 50)//5, -1)], SCREENWIDTH/2, SCREENHEIGHT/2, True)
	
	if lose_counter > -1:
		if not main.TRAINING:
			draw_text(type_font, f"MISSION {main.stage} "[:min((-lose_counter + 60)//5, -1)], SCREENWIDTH/2, SCREENHEIGHT/2-90, True)
			draw_text(secret_font, "FAILED", SCREENWIDTH/2+140, SCREENHEIGHT/2-20, True, color=(255, 0, 0), rotation=20)
	if win_counter > -1:
		if not main.TRAINING:
			draw_text(type_font, f"MISSION {main.stage} "[:min((-win_counter + 60)//5, -1)], SCREENWIDTH/2, SCREENHEIGHT/2-90, True)
			draw_text(secret_font, "PASSED", SCREENWIDTH/2+140, SCREENHEIGHT/2-20, True, color=(255, 0, 0), rotation=20)
	
	if not main.TRAINING:
		draw_text(pixel_font, f"Lives: {main.lives}", 10, 10)
	
	if transition_counter >= 0 and not main.TRAINING:
		transition_surf.set_alpha(int((1-abs(transition_counter / TRANSITIONLENGTH - 1)) * 255), RLEACCEL)
		if not main.TRAINING:
			screen.blit(transition_surf, (0, 0))
	
	if over_counter > -1:
		if not main.TRAINING:
			draw_text(type_font, "GAME OVER"[:max(over_counter - 60, 0)//8], SCREENWIDTH/2, SCREENHEIGHT/2-150, True)
			draw_text(type_font, "THANKS FOR PLAYING!"[:max(over_counter - 140, 0)//5], SCREENWIDTH/2, SCREENHEIGHT/2+200, True)
	
	if not main.TRAINING:
		display.flip()
	clock.tick(60)

	return move_action, shoot_action


def run():
	global round_counter, intro_counter, win_counter, lose_counter, over_counter
	global transition_counter
	main.env.reset(level=1)

	# print(f"\n\n\n\n\nrunning while loop\n\n\n\n\n")
	#goto_stage(STAGES[0])

	stop = False
	# Defaults to not moving/shooting
	shoot_action = 0
	while not stop:
		move_action, shoot_action = run_step()
		main.env.step((move_action, shoot_action))
