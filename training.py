import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from IPython.display import clear_output
import matplotlib.pyplot as plt
from tqdm import tqdm


from RL_utils.rubiks_cube_env import CubeClass
from RL_utils.batch_loader import get_batch
from RL_utils.replay_buffer import ReplayBuffer, Transition
from ML_utils.model import Model
from ML_utils.utils import eval_model, optimize_model



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('[INFO] Imports complete ...')



BATCH_SIZE = 128
GAMMA = 0.999
TARGET_UPDATE = 2000
N_ENV = 200
N_SCRAMBLE = 12

batch_loader = get_batch(N_ENV, N_SCRAMBLE, 0.75)



policy_net = Model().to(device)
target_net = Model().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# optimizer = optim.RMSprop(policy_net.parameters())
optimizer = optim.Adam(policy_net.parameters())

env = CubeClass()
memory = ReplayBuffer(20000)


steps_done = 0
episode_durations = []
epslions = []

print('[INFO] Models built ...')


num_episodes = 100000

print(f'[INFO] Training start: {num_episodes} epochs ...')

for i_episode in tqdm(range(num_episodes)):
    # Initialize the environment and state
    
    state, action, reward, next_state = next(batch_loader)
    
    for i in range(len(state)):
        memory.push(
            state[i], 
            action[i].unsqueeze(0).unsqueeze(0),
            next_state[i],
            reward[i].unsqueeze(0)
        )

    optimize_model(
        policy_net, 
        target_net, 
        optimizer, 
        memory, 
        BATCH_SIZE, 
        device, 
        GAMMA
    )
    
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
        
    if i_episode % 1000 == 0:
        
        torch.save(policy_net.state_dict(), 'models/policy_net.pt')
        torch.save(optimizer.state_dict(), 'models/policy_net_adam.pt')
        
        episode_durations += eval_model(N_SCRAMBLE, policy_net, env, device)
        
        # clear_output(True)
        # plt.figure(2)
        # plt.yticks(np.arange(0, N_SCRAMBLE+((N_SCRAMBLE*2)//2), 1))
        # plt.grid(True)
        # durations_t = torch.tensor(episode_durations, dtype=torch.float)
        # f = 200
        # if len(durations_t) >= f:
        #     means = durations_t.unfold(0, f, 1).mean(1).view(-1)
        #     means = torch.cat((torch.zeros(f-1), means))
        #     plt.plot(means.numpy())
        # plt.show()

    
print('Complete')