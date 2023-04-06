import random
from jax import numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import torch

from constants import *



class CubeClass:
    def __init__(self) -> None:
        self.cube = self.init_cube()
        self.action_list = ACTION_LIST
        self.action_id = ACTION_ID


    def init_cube(self):
        return INIT_ARRAY.copy()

    def scramble(self, N):
        
        cube = INIT_ARRAY.copy()
        
        for _ in range(N):
            a = random.choice(self.action_list)
            cube = np.take(cube, ACTION_PERM[a])

        return cube

    
    def scramble_sars(self, N, p):

        cube = INIT_ARRAY.copy()

        states = [cube]
        actions = ['x']

        for i in range(N):
            a = random.choice(self.action_list)
            a_ = actions[-1]
            
            
            if a[-1] != '*' and a_[-1] != '*':
                while a == a_:
                    a = random.choice(self.action_list)

                cube = np.take(cube, ACTION_PERM[a])
                states.append( cube )
                
                if a[-1] == '*':
                    actions.append(a)
                else:
                    actions.append( a+"'" if len(a) == 1 else a[0] )
                
                
            elif a[-1] != '*' and a_[-1] == '*':
                cube = np.take(cube, ACTION_PERM[a])
                states.append( cube )
                actions.append( a+"'" if len(a) == 1 else a[0] )

                            
            elif a[-1] == '*' and a_[-1] != '*':
                cube = np.take(cube, ACTION_PERM[a])
                states.append( cube )
                actions.append( a )
            
            
            elif a[-1] == '*' and a_[-1] == '*':
                while a == a_:
                    a = random.choice(self.action_list)
                    
                cube = np.take(cube, ACTION_PERM[a])
                states.append( cube )
                
                if a[-1] == '*':
                    actions.append(a)
                else:
                    actions.append( a+"'" if len(a) == 1 else a[0] )             
                
                
                

        next_states = states[:-1]
        states = states[1:]
        rewards = [1/(1+0.75*i) for i in range(N)]
        actions = actions[1:]


        for i in range(N):
            if random.random() < p:
                a = random.choice(self.action_list)
                while a == actions[i]:
                    a = random.choice(self.action_list)

                actions[i] = a
                next_states[i] = np.take(states[i], ACTION_PERM[a])
                rewards[i] = 0
        
        
        states_ = []
        next_states_ = []
        actions_ = []
        rewards_ = []
        for i in range(N):
            if random.random() < (1-np.exp(-0.5*(i+1))):
                states_.append( states[i] )
                next_states_.append( next_states[i] )
                actions_.append( actions[i] )
                rewards_.append( rewards[i] )
        
        
        states  = torch.tensor(np.array(states_), dtype=torch.int64)
        next_states  = torch.tensor(np.array(next_states_), dtype=torch.int64)
        actions = torch.tensor([self.action_id[a] for a in actions_])
        rewards = torch.tensor(rewards_)

        
        return states, actions, rewards, next_states



    def render(self, cube=None):

        if cube is None:
            cube = self.cube

        U = cube[0 :9 ].reshape(3,3)
        D = cube[9 :18].reshape(3,3)
        L = cube[18:27].reshape(3,3)
        R = cube[27:36].reshape(3,3)
        F = cube[36:45].reshape(3,3)
        B = cube[45:54].reshape(3,3)

        # Flipping/Rotating the faces before rendering:
        B = np.rot90(B, 1)
        B = np.flipud(B)
        R = np.fliplr(R)
        F = np.rot90(F, 3)
        L = np.rot90(L, 2)
        U = np.rot90(U, 3)
        D = np.rot90(D, 3)
        D = np.fliplr(D)

        # 2d plane:
        plane = np.zeros((9, 12))
        plane[:, :] = VOID
        plane[3:6, 0:3]  = F 
        plane[3:6, 3:6]  = U
        plane[3:6, 6:9]  = B 
        plane[3:6, 9:12] = D 
        plane[0:3, 3:6]  = L
        plane[6:9, 3:6]  = R

        # Colored plane:
        colored_plane = np.zeros((9, 12, 3), dtype=np.int64)
        for i in range(len(plane)):
            for j in range(len(plane[0])):
                colored_plane[i, j, :] = COLOR_RGB[int(plane[i,j])]

        plt.imshow(colored_plane)


    def action(self, move, cube):
        
        moves = move.strip().split(' ')
        if cube is None:
            for a in moves:
                self.cube = np.take(self.cube, ACTION_PERM[a])
        else:
            for a in moves:
                cube = np.take(cube, ACTION_PERM[a])
            return cube

    
    def step(self, move):

        self.action(move)

        reward = 0
        done = 0
        if self.is_solved():
            reward = 5
            done = 1

        return reward, done


    def is_solved(self, cube=None):
        
        if cube is None:
            cube = self.cube

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
        

    def state(self):
        cube = np.asarray(self.cube)
        x = torch.tensor(cube, dtype=torch.long)
        return x