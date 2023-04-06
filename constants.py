import numpy as np

WHITE = 0
YELLOW = 1
BLUE = 2
GREEN = 3
ORANGE = 4
RED = 5
VOID = 6

COLOR_RGB = {
    WHITE: (255, 255, 255),
    YELLOW: (255, 255, 0),
    BLUE: (0, 0, 255),
    GREEN: (0, 128, 0),
    ORANGE: (255, 165, 0),
    RED: (255, 0, 0),
    VOID: (0,0,0),
}


INIT_ARRAY = np.array([
    WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE,
    YELLOW, YELLOW, YELLOW, YELLOW, YELLOW, YELLOW, YELLOW, YELLOW, YELLOW,
    ORANGE, ORANGE, ORANGE, ORANGE, ORANGE, ORANGE, ORANGE, ORANGE, ORANGE,
    RED, RED, RED, RED, RED, RED, RED, RED, RED,
    GREEN, GREEN, GREEN, GREEN, GREEN, GREEN, GREEN, GREEN, GREEN,
    BLUE, BLUE, BLUE, BLUE, BLUE, BLUE, BLUE, BLUE, BLUE], dtype=np.int32)


ACTION_LIST = ["U", "D", "R", "L", "F", "B", "U'", "D'", "R'", "L'", "F'", "B'", "U*", "D*", "R*", "L*", "F*", "B*"]


ACTION_PERM = {
    "U": [6,3,0,7,4,1,8,5,2,9,10,11,12,13,14,15,16,17,36,37,38,21,22,23,24,25,26,45,46,47,30,31,32,33,34,35,29,28,27,39,40,41,42,43,44,20,19,18,48,49,50,51,52,53,],
    "D": [0,1,2,3,4,5,6,7,8,15,12,9,16,13,10,17,14,11,18,19,20,21,22,23,42,43,44,27,28,29,30,31,32,51,52,53,36,37,38,39,40,41,35,34,33,45,46,47,48,49,50,26,25,24,],
    "R": [0,1,38,3,4,41,6,7,44,9,10,47,12,13,50,15,16,53,18,19,20,21,22,23,24,25,26,29,32,35,28,31,34,27,30,33,36,37,17,39,40,14,42,43,11,45,46,8,48,49,5,51,52,2,],
    "L": [51,1,2,48,4,5,45,7,8,42,10,11,39,13,14,36,16,17,24,21,18,25,22,19,26,23,20,27,28,29,30,31,32,33,34,35,0,37,38,3,40,41,6,43,44,9,46,47,12,49,50,15,52,53,],
    "F": [0,1,2,3,4,5,26,23,20,9,10,11,12,13,14,35,32,29,18,19,15,21,22,16,24,25,17,27,28,6,30,31,7,33,34,8,42,39,36,43,40,37,44,41,38,45,46,47,48,49,50,51,52,53,],
    "B": [24,21,18,3,4,5,6,7,8,33,30,27,12,13,14,15,16,17,9,19,20,10,22,23,11,25,26,0,28,29,1,31,32,2,34,35,36,37,38,39,40,41,42,43,44,51,48,45,52,49,46,53,50,47,],
    "U'": [2,5,8,1,4,7,0,3,6,9,10,11,12,13,14,15,16,17,47,46,45,21,22,23,24,25,26,38,37,36,30,31,32,33,34,35,18,19,20,39,40,41,42,43,44,27,28,29,48,49,50,51,52,53,],
    "D'": [0,1,2,3,4,5,6,7,8,11,14,17,10,13,16,9,12,15,18,19,20,21,22,23,53,52,51,27,28,29,30,31,32,44,43,42,36,37,38,39,40,41,24,25,26,45,46,47,48,49,50,33,34,35,],
    "R'": [0,1,53,3,4,50,6,7,47,9,10,44,12,13,41,15,16,38,18,19,20,21,22,23,24,25,26,33,30,27,34,31,28,35,32,29,36,37,2,39,40,5,42,43,8,45,46,11,48,49,14,51,52,17,],
    "L'": [36,1,2,39,4,5,42,7,8,45,10,11,48,13,14,51,16,17,20,23,26,19,22,25,18,21,24,27,28,29,30,31,32,33,34,35,15,37,38,12,40,41,9,43,44,6,46,47,3,49,50,0,52,53,],
    "F'": [0,1,2,3,4,5,29,32,35,9,10,11,12,13,14,20,23,26,18,19,8,21,22,7,24,25,6,27,28,17,30,31,16,33,34,15,38,41,44,37,40,43,36,39,42,45,46,47,48,49,50,51,52,53,],
    "B'": [27,30,33,3,4,5,6,7,8,18,21,24,12,13,14,15,16,17,2,19,20,1,22,23,0,25,26,11,28,29,10,31,32,9,34,35,36,37,38,39,40,41,42,43,44,47,50,53,46,49,52,45,48,51,],
    'U*': [8,7,6,5,4,3,2,1,0,9,10,11,12,13,14,15,16,17,29,28,27,21,22,23,24,25,26,20,19,18,30,31,32,33,34,35,47,46,45,39,40,41,42,43,44,38,37,36,48,49,50,51,52,53],
    'D*': [0,1,2,3,4,5,6,7,8,17,16,15,14,13,12,11,10,9,18,19,20,21,22,23,35,34,33,27,28,29,30,31,32,26,25,24,36,37,38,39,40,41,53,52,51,45,46,47,48,49,50,44,43,42],
    'R*': [0,1,17,3,4,14,6,7,11,9,10,8,12,13,5,15,16,2,18,19,20,21,22,23,24,25,26,35,34,33,32,31,30,29,28,27,36,37,53,39,40,50,42,43,47,45,46,44,48,49,41,51,52,38],
    'L*': [15,1,2,12,4,5,9,7,8,6,10,11,3,13,14,0,16,17,26,25,24,23,22,21,20,19,18,27,28,29,30,31,32,33,34,35,51,37,38,48,40,41,45,43,44,42,46,47,39,49,50,36,52,53],
    'F*': [0,1,2,3,4,5,17,16,15,9,10,11,12,13,14,8,7,6,18,19,35,21,22,32,24,25,29,27,28,26,30,31,23,33,34,20,44,43,42,41,40,39,38,37,36,45,46,47,48,49,50,51,52,53],
    'B*': [11,10,9,3,4,5,6,7,8,2,1,0,12,13,14,15,16,17,33,19,20,30,22,23,27,25,26,24,28,29,21,31,32,18,34,35,36,37,38,39,40,41,42,43,44,53,52,51,50,49,48,47,46,45]
}

ACTION_ID = {'U': 0,
            'D' : 1,
            'R' : 2,
            'L' : 3,
            'F' : 4,
            'B' : 5,
            "U'": 6,
            "D'": 7,
            "R'": 8,
            "L'": 9,
            "F'": 10,
            "B'": 11,
            'U*': 12,
            'D*': 13,
            'R*': 14,
            'L*': 15,
            'F*': 16,
            'B*': 17
        }


OPPOSITE_FACES = {
    'U':'D',
    'R':'L',
    'F':'B',
    'D':'U',
    'L':'R',
    'B':'F',
}