
import sys



PLAYBACK_SPEED = 2 # How fast to play the game; this effectively multiplies onto all velocities



COLORKEY = (255, 0, 255)

# SOUNDMUTED = "mute" in sys.argv
SOUNDMUTED = True
COLORKEY = (255, 0, 255)

TANK_MOVE_EVERY_X_FRAME = 10

SOUNDMUTED = "mute" in sys.argv
DEBUGMODE = "debug" in sys.argv

PLAYER_LIVES = 15

SHEETWIDTH = 16
SHEETHEIGHT = 16

WALLTILE = 16
BODYTILE = 32
TRACKTILE = 32
HEADTILE = 64
BULLETTILE = 32
EFFECTTILE = 64

BODYTILEX = BODYTILE*4
BODYTILEY = BODYTILE*3
TRACKTILEX = TRACKTILE*4
TRACKTILEY = TRACKTILE*3
HEADTILEX = HEADTILE*4
HEADTILEY = HEADTILE*3
BULLETTILEX = BULLETTILE*4
BULLETTILEY = BULLETTILE*3

EFFECTSCALE = EFFECTTILE*4

SCALE = WALLTILE
SCALEX = 4 * SCALE
SCALEY = 3 * SCALE

FIELDWIDTH = 22
FIELDHEIGHT = 16

# Screen dimensions cannot currently be changed, a lot of refactoring and fixing would be needed to fix this
SCREENWIDTH = FIELDWIDTH * SCALEX
SCREENHEIGHT = FIELDHEIGHT * SCALEY

TANKBODYSPEED = 0.02 * PLAYBACK_SPEED
TANKAIMSPEED = 0.2 * PLAYBACK_SPEED

TANKCOLLISION = 0.45
TANKTANKCOLLISION = 0.55

TRACKSPEED = 4 * PLAYBACK_SPEED
EFFECTSPEED = 0.4 * PLAYBACK_SPEED
MOVESOUNDSPEED = 4 * PLAYBACK_SPEED

FUSELENGTH = 360
ARMLENGTH = 80

MINERANGE = 2.2
MINEWALLRANGE = 2

BULLETSPEED = 0.1 * PLAYBACK_SPEED
ROCKETSPEED = 0.2 * PLAYBACK_SPEED

DETTANKRANGE = 0.7
DETMINERANGE = 0.8
DETBULLETRANGE = 0.2

TANKSHOTCOOLDOWN = 10

INTROLENGTH = 240
WINLENGTH = 120
LOSELENGTH = 120
TRANSITIONLENGTH = 30

FADEOUTLENGTH = 100

# bullet_limit, bullet_bounce, mine_limit, rocket, speed
TANKPROPERTIES = {
	"white" : (5, 1, 2, False, 0.06 * PLAYBACK_SPEED),
	
	"grey" : (1, 1, 0, False, 0 * PLAYBACK_SPEED),
	"yellow" : (2, 0, 0, False, 0.02 * PLAYBACK_SPEED),
	"blue" : (5, 2, 0, False, 0.04 * PLAYBACK_SPEED),
	"green" : (0, 0, 5, False, 0.08 * PLAYBACK_SPEED),
	"red" : (5, 3, 0, True, 0 * PLAYBACK_SPEED),
	"black" : (2, 1, 2, True, 0.05 * PLAYBACK_SPEED),
}

CHANNELS = {
	"flute" : 0,
	"synth" : 1,
	"snare" : 2,
	"timpani1" : 3,
	"timpani2" : 4,
	"symbals1" : 5,
	"symbals2" : 6,
	"hihat1" : 7,
	"hihat2" : 8,
	"triangle" : 9,
	"tuba" : 10,
	"bells" : 11,
}

THEME_VERSIONS = {
	"base" : (["flute", "snare", "triangle", "hihat2"], []),
	
	"grey" : (["hihat1"], ["hihat2"]),
	"yellow" : (["symbals1"], []),
	"blue" : (["timpani1"], []),
	"green" : (["tuba"], []),
	"red" : (["bells"], []),
	"black" : (["synth"], ["flute"]),
}

# VOLUMES = {
# 	"tank_move" : 0.10,
# 	"mission_win" : 0.60,
# 	"mission_lose" : 0.60,
# 	"mission_intro" : 0.80,
# }

VOLUMES = {
	"tank_move" : 0.0,
	"mission_win" : 0.0,
	"mission_lose" : 0.0,
	"mission_intro" : 0.0,
}

STAGES = [
	"1", "2", "3", "4", "5", "6", "7", "8", "ALPHA"
]
