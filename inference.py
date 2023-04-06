import torch

from tqdm import tqdm

from ML_utils.model import Model
from inference_utils.utils import instadeep2nazim, is_goal_state
from inference_utils.beam_search import beam_search


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



print(f'[INFO] Loading evaluation tests ...')

easy_conf = []
with open('inference_utils/evaluation samples/easy_samples.txt') as file:
    easy_conf = [s.strip() for s in file.readlines()]

medium_conf = []
with open('inference_utils/evaluation samples/medium_samples.txt') as file:
    medium_conf = [s.strip() for s in file.readlines()]

hard_conf = []
with open('inference_utils/evaluation samples/hard_samples.txt') as file:
    hard_conf = [s.strip() for s in file.readlines()]




print('[INFO] Load model ...')

model = Model().to(device)
model.load_state_dict( torch.load('models/nazim_policy_net_8_16a_12000s.pt', map_location=device) )
model = model.eval()




print('[INFO] Evaluation started ...')

res = []
widths = [100, 500, 1000, 5000, 10000]

for diff, conf_list in zip(['easy', 'medium', 'hard'], [easy_conf, medium_conf, hard_conf]):

    seqs = []
    
    for conf in tqdm(conf_list, desc=f'{diff}'):
        state = instadeep2nazim(conf)

        for width in widths:
            seq = beam_search(model, state, width, 20, is_goal_state, device)
            if seq is not None:
                break
                    
        seqs.append(seq)
    
    res.append( seqs.copy() )



print('[INFO] Results ...')

diff_len = {
    'easy':1,
    'medium':4,
    'hard':8
}

for seqs in res:
    print(list([len(seq) for seq in seqs]))

for diff, seqs in zip(['easy', 'medium', 'hard'], res):
    n_none = sum([ seq is None for seq in seqs ])

    print(f'Difficulty:\t{diff}')
    print(f'    Solve accuracy: \t{ (len(seqs) - n_none)/len(seqs)* 100 }%')
