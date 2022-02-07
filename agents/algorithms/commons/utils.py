import numpy as np

def state_reduction(state, keep_features, condition=True):

    state = state[0][0:keep_features]

    if condition == True:
        state = np.reshape(state, [1, keep_features])

    return state

def action_construction(d_action, random_c_action):

    a_lst = list(random_c_action)
    a_lst[0] = d_action
    action = tuple(a_lst)

    return action

def to_gym_action(action_c, action_d, flat_actions=True):
    # assuming both are torch tensors

    if flat_actions:
        ac = action_c.tolist()
    else:
        ac = action_c.unsqueeze(-1).tolist()[0]

    ad = action_d.squeeze().item()

    return [ad, ac]

def gym_to_buffer(action, flat_actions=True):

    ad = action[0]
    if flat_actions:
        ac = np.hstack(action[1:])
    else:
        ac = action[1]
    return [ad] + np.array(ac).flatten().tolist()

def to_torch_action(actions):

    ad = actions[:, 0]
    ac = actions[:, 1:]
    return ac, ad
