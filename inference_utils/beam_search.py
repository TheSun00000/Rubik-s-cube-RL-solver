import torch
import numpy as np

from constants import ACTION_PERM, ACTION_LIST




def beam_search(model, start_node, beam_width, max_iterations, is_goal_state, device):
    """
    Conducts beam search from the start node to find the goal state.

    Args:
    - start_node: the starting node
    - beam_width: the number of nodes to consider at each level of the search
    - max_iterations: the maximum number of iterations to conduct
    - is_goal_state: a function that takes a node and returns True if it is a goal state, False otherwise

    Returns:
    - the goal node if found, None otherwise
    """
    iteration = 0
    beam = [(start_node, [])]  # (cost, node) heap

    while iteration < max_iterations and beam:

        # Pop the beam_width nodes with the lowest cost

#         print(len(beam))

        candidates = []
        for _ in range(min(len(beam), beam_width)):
            node, seq = beam.pop(0)
            if is_goal_state(node):
                return seq
            candidates.append((node, seq))

        candidates.append((node, seq))
        candidates.append((node, seq))

        X = [node for (node, seq) in candidates]
        X = torch.tensor( np.array(X) ).to(device)
        Q = model(X)

#         print(X.shape, Q.shape)
        
        top_k = Q.flatten().argsort(descending=True)[:beam_width].cpu().numpy()

        node_child = [ (i//18, i%18) for i in top_k ]

        beam = []

        # print(candidates)

        for (n, a) in node_child:
            # candidates
            beam.append(
                (
                    np.take(candidates[n][0], ACTION_PERM[ACTION_LIST[a]]) ,
                    candidates[n][1]+[ACTION_LIST[a]] 
                )
            )
        
        # Sort the expanded nodes by cost and add the beam_width lowest cost nodes to the beam

        iteration += 1
    return None




import heapq

def beam_search_noise(model, revert_iteration, start_node, beam_width, max_iterations, is_goal_state, device):
    """
    Conducts beam search from the start node to find the goal state.

    Args:
    - start_node: the starting node
    - beam_width: the number of nodes to consider at each level of the search
    - max_iterations: the maximum number of iterations to conduct
    - is_goal_state: a function that takes a node and returns True if it is a goal state, False otherwise

    Returns:
    - the goal node if found, None otherwise
    """
    iteration = 0
    beam = [(start_node, [])]  # (cost, node) heap

    while iteration < max_iterations and beam:

        # Pop the beam_width nodes with the lowest cost

#         print(len(beam))

        candidates = []
        for _ in range(min(len(beam), beam_width)):
            node, seq = beam.pop(0)
            if is_goal_state(node):
                return seq
            candidates.append((node, seq))

        candidates.append((node, seq))
        candidates.append((node, seq))

        X = [node for (node, seq) in candidates]
        X = torch.tensor( np.array(X) ).to(device)
        Q = model(X)

#         print(X.shape, Q.shape)
        
        if iteration == revert_iteration:
            top_k = Q.flatten().argsort(descending=False)[:beam_width].cpu().numpy()
        else:
            top_k = Q.flatten().argsort(descending=True)[:beam_width].cpu().numpy()
        
#         print(iteration, len(Q.flatten().sort(descending=True).values))

        node_child = [ (i//18, i%18) for i in top_k ]

        beam = []

        # print(candidates)

        for (n, a) in node_child:
            # candidates
            beam.append(
                (
                    np.take(candidates[n][0], ACTION_PERM[ACTION_LIST[a]]) ,
                    candidates[n][1]+[ACTION_LIST[a]] 
                )
            )
        
        # Sort the expanded nodes by cost and add the beam_width lowest cost nodes to the beam

        iteration += 1
    return None