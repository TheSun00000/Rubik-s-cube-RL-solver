import numpy as np

from constants import INIT_ARRAY


def instadeep2nazim(conf):
    
    # From orinigal format to my format

    conf = conf.replace('0', 'w')
    conf = conf.replace('1', 'g')
    conf = conf.replace('2', 'r')
    conf = conf.replace('3', 'b')
    conf = conf.replace('4', 'o')
    conf = conf.replace('5', 'y')

    conf = conf.replace('w', '0')
    conf = conf.replace('g', '3')
    conf = conf.replace('r', '5')
    conf = conf.replace('b', '2')
    conf = conf.replace('o', '4')
    conf = conf.replace('y', '1')

    cube = np.zeros(54, dtype=np.int64)

    # UP
    for i in range(3):
        for j in range(3):cube[i*3 + j] = int(conf[i*3 + j])
    # FRONT
    for i in range(3):
        for j in range(3):cube[36 + i*3 + j] = int(conf[9+i*3 + j])
    # RIGHT
    for i in range(3):
        for j in range(3):cube[27 + i*3 + 2-j] = int(conf[18 + i*3 + j])
    # BEHIND
    for i in range(3):
        for j in range(3):cube[45 + 2 + i*3 - j] = int(conf[27 + i*3 + j])
    # LEFT
    for i in range(3):
        for j in range(3):cube[18 + i*3 + j] = int(conf[36 + i*3 + j])
    # DOWN
    for i in range(3):
        for j in range(3):cube[9+6 - i*3 + j] = int(conf[45 + i*3 + j])

    return cube



def is_solved(cube):

    U = cube[0 :9 ]
    D = cube[9 :18]
    L = cube[18:27]
    R = cube[27:36]
    F = cube[36:45]
    B = cube[45:54]

    if len(np.unique(U))==1 and len(np.unique(D))==1 and len(np.unique(F))==1 and len(np.unique(B))==1 \
        and len(np.unique(R))==1 and len(np.unique(L))==1:
        return True

    return False

def is_goal_state(cube):
    return all(cube == INIT_ARRAY)



