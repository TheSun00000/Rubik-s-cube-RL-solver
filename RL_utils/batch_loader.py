import torch
from RL_utils.rubiks_cube_env import CubeClass


def get_batch(n_env, n_scramble, p):

    envs = [CubeClass() for _ in range(n_env)]


    while True:

        states = []
        actions = []
        rewards = []
        next_states = []
        
        for env in envs:
            env.init_cube()
            state, action, reward, next_state = env.scramble_sars(n_scramble, p)
            states.append( state )
            actions.append( action )
            rewards.append( reward )
            next_states.append( next_state )
            
            

        states = torch.concat(states)
        actions = torch.concat(actions)
        rewards = torch.concat(rewards)
        next_states = torch.concat(next_states)

        yield states, actions, rewards, next_states