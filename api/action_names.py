from src.mdm.data_loaders.a2m.humanact12 import humanact12_coarse_action_enumerator as humanact12_actions


with open('./dataset/uestc_actions.txt', 'r') as f:
    uestc_actions = [line.rstrip() for line in f]
    uestc_actions = {i: a for i, a in enumerate(uestc_actions)}


all_actions = [
    f'{dataset}/{action}' \
    for dataset, actions in [('HumanAct12Poses', humanact12_actions), ('UESTC_RGBD', uestc_actions)]
    for action in actions.values()
]

all_actions = sorted(all_actions)
all_actions = tuple(all_actions)
        

