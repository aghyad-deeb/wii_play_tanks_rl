import numpy as np

ASCII_SPACE = ord(' ')
ASCII_ZERO = ord('0')
ASCII_PLAYER = ord('.')
ASCII_ENEMY_GRAY = ord('A')
ASCII_ENEMY_YELLOW = ord('B')
ASCII_ENEMY_BLUE = ord('C')
ASCII_ENEMY_GREEN = ord('D')
ASCII_ENEMY_RED = ord('E')
ASCII_ENEMY_BLACK = ord('F')
ASCII_WALL = ord('1')


immobile_enemy_asciis = [
    ASCII_ENEMY_GRAY,
    ASCII_ENEMY_RED,
]

mobile_enemy_asciis = [
    ASCII_ENEMY_YELLOW,
    ASCII_ENEMY_BLUE,
    ASCII_ENEMY_GREEN,
    ASCII_ENEMY_BLACK,
]

def main():
    map = make_map(1, min_size=10, max_size=10, should_save_map=True)
    map = make_map(2, min_size=10, max_size=10, should_save_map=True)
    map = make_map(3, min_size=10, max_size=10, should_save_map=True)
    map = make_map(4, min_size=10, max_size=10, should_save_map=True)
    map = make_map(5, min_size=10, max_size=10, should_save_map=True)


def save_map(map, num):
    chars = ''
    for row in map:
        for ascii in row:
            chars += chr(ascii)
        chars += '\n'
    chars = chars[:-1]


    with open(f'training_stages\\{num}.txt', 'w') as file:
        file.write(chars)
    
    return



def make_template_map(min_size:int=6, max_size:int=20)->np.ndarray:
    '''
    Returns a numpy array filled with 32's (spaces) with 48's (zeros) in the border.
    '''
    h, w = np.random.randint(min_size, max_size+1, 2)
    map = ASCII_SPACE * np.ones((h, w), dtype=int)
    map[:, 0] = ASCII_ZERO
    map[0, :] = ASCII_ZERO
    map[:, -1] = ASCII_ZERO
    map[-1, :] = ASCII_ZERO
    return map



def make_map(num:int, **kwargs)->np.ndarray:
    '''
    Makes training map number `num`. Passes *params to the corresponding map
    creation function.

    Returns:
        map (np.ndarray): integer of ASCII values representing game state.
    '''
    match num:
        case 1:
            return make_map_1(**kwargs)
        case 2:
            return make_map_2(**kwargs)
        case 3:
            return make_map_3(**kwargs)
        case 4:
            return make_map_4(**kwargs)
        case 5:
            return make_map_5(**kwargs)
        case _:
            raise ValueError('Invalid num provided. Must be an integer between 1 and 5.')


def attempt_placement(map, obj, max_attempts=20)->np.ndarray:
    '''
    Attempts to place an object on the map in a free spot. Throws an exception
    if placement fails.
    '''
    attempts = 0
    while attempts < max_attempts:
        h = np.random.randint(1, map.shape[0] - 1)
        w = np.random.randint(1, map.shape[1] - 1)
        
        if map[h, w] == ASCII_SPACE:
            map[h, w] = obj
            return map
        
        attempts += 1

    raise Exception(f"Could not locate a valid place for the object in {max_attempts} attempts.")


def attempt_wall_placement(map, sz, orientation, max_attempts=40)->np.ndarray:
    '''
    Attempts to place a wall of size `sz` on the map in a free set of spots, oriented either
    0 for 'vertical' or 1 for 'horizontal'.
    
    Does nothing to the map if a valid position cannot be found, and returns 0.

    Returns a map and a boolean representing success. 1 is success, 0 is failure.
    '''
    if (orientation != 0 and orientation != 1):
        raise ValueError('Invalid orientation.')
    if (
        (orientation == 0 and sz > map.shape[0] - 3) or 
        (orientation == 1 and sz > map.shape[1] - 3)
    ):
        raise ValueError('Invalid size with given orientation.')
    
    attempts = 0
    while attempts < max_attempts:
        if orientation == 0:
            h = np.random.randint(1, map.shape[0] - sz - 1)
            w = np.random.randint(1, map.shape[1] - 1)
        
            for h_idx in range(h, h+sz):
                map[h_idx, w] = ASCII_WALL
        
            return map, 1
        
        elif orientation == 1:
            h = np.random.randint(1, map.shape[0] - 1)
            w = np.random.randint(1, map.shape[1] - sz)
        
            for w_idx in range(w, w+sz):
                map[h, w_idx] = ASCII_WALL
        
            return map, 1
        
        attempts += 1
        
    print('Unable to place wall.')

    return map, 0


def make_map_1(min_size:int=10, max_size:int=20, should_save_map:bool=False)->np.ndarray:
    '''
    Training map 1. Basic immobile enemy without wall.
    '''
    if (min_size < 5):
        raise ValueError('Invalid min_size. Must be at least 5.')
    
    map = make_template_map(min_size, max_size)
    map = attempt_placement(map, ASCII_PLAYER)
    map = attempt_placement(map, ASCII_ENEMY_GRAY)

    if should_save_map:
        save_map(map, 1)

    return map
    

def make_map_2(min_size:int=6, max_size:int=20, should_save_map:bool=False, min_wall_sz:int=2)->np.ndarray:
    '''
    Training map 2. Basic immobile enemy with wall.
    '''
    if (min_size < 5):
        raise ValueError('Invalid min_size. Must be at least 5.')
    if (min_wall_sz > min_size - 3):
        raise ValueError('Invalid min_wall_size. Must be at least min_size - 3.')
    
    map = make_template_map(min_size, max_size)

    #better to place walls before players in order to reduce the chance of failure.
    orientation = np.random.randint(0, 2)
    wall_sz = np.random.randint(min_wall_sz, map.shape[orientation]-2)
    map, _ = attempt_wall_placement(map, wall_sz, orientation)
    
    map = attempt_placement(map, ASCII_PLAYER)
    map = attempt_placement(map, ASCII_ENEMY_GRAY)
    
    if should_save_map:
        save_map(map, 2)
    return map


def make_map_3(min_size:int=6, max_size:int=20, should_save_map:bool=False)->np.ndarray:
    '''
    Training map 3. Random mobile enemy without a wall.
    '''
    if (min_size < 5):
        raise ValueError('Invalid min_size. Must be at least 5.')
    
    enemy_ascii = np.random.choice(mobile_enemy_asciis)
    map = make_template_map(min_size, max_size)
    map = attempt_placement(map, ASCII_PLAYER)
    map = attempt_placement(map, enemy_ascii)
    
    if should_save_map:
        save_map(map, 3)


def make_map_4(min_size:int=6, max_size:int=20, should_save_map:bool=False, min_wall_sz:int=2)->np.ndarray:
    '''
    Training map 4. Random mobile enemy with a wall.
    '''
    if (min_size < 5):
        raise ValueError('Invalid min_size. Must be at least 5.')
    
    map = make_template_map(min_size, max_size)
    
    #better to place walls before players in order to reduce the chance of failure.
    orientation = np.random.randint(0, 2)
    wall_sz = np.random.randint(min_wall_sz, map.shape[orientation]-2)
    map, _ = attempt_wall_placement(map, wall_sz, orientation)
    
    enemy_ascii = np.random.choice(mobile_enemy_asciis)
    map = attempt_placement(map, ASCII_PLAYER)
    map = attempt_placement(map, enemy_ascii)
    
    if should_save_map:
        save_map(map, 4)


def make_map_5(min_size:int=6, max_size:int=20, should_save_map:bool=False)->np.ndarray:
    '''
    Training map 5. Two random enemies, each can be mobile or immobile, no walls.
    '''
    if (min_size < 5):
        raise ValueError('Invalid min_size. Must be at least 5.')
    
    enemy_ascii_1, enemy_ascii_2 = np.random.choice(mobile_enemy_asciis + immobile_enemy_asciis, 2, replace=True)
    map = make_template_map(min_size, max_size)
    map = attempt_placement(map, ASCII_PLAYER)
    map = attempt_placement(map, enemy_ascii_1)
    map = attempt_placement(map, enemy_ascii_2)
    
    if should_save_map:
        save_map(map, 5)



if __name__ == "__main__":
    main()