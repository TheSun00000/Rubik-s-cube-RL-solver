import torch
import torch.nn as nn
import numpy as np

from constants import ACTION_PERM
from RL_utils.replay_buffer import Transition



def eval_model(s, model, env, device):
    
    num_episodes = 10    
    episode_durations = []

    for _ in range(num_episodes):
        

        env.init_cube()
        state = env.scramble(s) 
        state  = torch.tensor(state, dtype=torch.int64, device=device)

        actions = []

        for t in range(100):
            model.eval()
            with torch.no_grad():
                # state  = torch.tensor(state, dtype=torch.int64, device=device)
                state  = state.clone().detach()
                action = model(state).cpu()
                action = action.max(1)[1].view(1, 1)

            action_ = env.action_list[action.item()]
            actions.append(action_)
            next_state = np.take(state.cpu(), ACTION_PERM[action_])

            done = env.is_solved(next_state)
                    
            state = next_state
            
            if done or t == 20:
                
                episode_durations.append(t + 1)
                break
        

    
    return episode_durations




def optimize_model(policy_net, target_net, optimizer, memory, batch_size, device, GAMMA):
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)

    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    
    batch_not_none = [s for s in batch.next_state if s is not None]
    
    n = 5
    while len(batch_not_none) == 0 and n > 0:
        transitions = memory.sample(batch_size)
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=device, dtype=torch.bool)
        batch_not_none = [s for s in batch.next_state if s is not None]
        n -= 1    
    if n == 0:
        return
        
    non_final_next_states = torch.cat(batch_not_none).to(device)
    
    
    state_batch = torch.cat(batch.state).to(device)
    action_batch = torch.cat(batch.action).to(device)
    reward_batch = torch.cat(batch.reward).to(device)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(batch_size, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        
    a, b = state_action_values.cpu().detach().numpy().reshape(-1), expected_state_action_values.cpu().detach().numpy().reshape(-1)
    
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    
    
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
        
    optimizer.step()

