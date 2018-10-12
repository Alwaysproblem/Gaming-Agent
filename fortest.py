
import numpy as np

STATE_DIM, ACTION_DIM, REWARD_DIM, DONE_DIM = 4, 2, 1, 1


def Store_State(ReplayMemory, ReplayMemory_size, s, a, r, s_, done):
    elements = np.expand_dims(np.hstack((s, a, r, s_, done)), axis = 0)
    ReplayMemory = np.concatenate((ReplayMemory, elements), 0)
    if len(ReplayMemory) > ReplayMemory_size:
        ReplayMemory = np.delete(ReplayMemory, 0, axis=0)
    full = any(ReplayMemory[0, :])
    return ReplayMemory, full

def Sample_State(ReplayMemory, sample_num, replace = False):
    ind = np.random.choice(range(len(ReplayMemory)), sample_num, replace = replace)
    Sample_batch = ReplayMemory[ind, :]
    s = Sample_batch[:, : STATE_DIM]
    a = Sample_batch[:, STATE_DIM: STATE_DIM + ACTION_DIM]
    r = Sample_batch[:, STATE_DIM + ACTION_DIM : STATE_DIM + ACTION_DIM + REWARD_DIM]
    s_ = Sample_batch[:, STATE_DIM + ACTION_DIM + REWARD_DIM : STATE_DIM + ACTION_DIM + REWARD_DIM + STATE_DIM]
    done = Sample_batch[:, -1:]

    return s, a, r, s_, done

r = 1

Done = 0

S = [1, 2, 3, 4]
A = [0, 1]
S_ = [2, 3, 4, 5]

S1 = [1, 2, 2, 4]
A1 = [0, 1]
S_1 = [2, 3, 7, 5]

S2 = [1, 2, 3, 6]
A2 = [1, 0]
S_2 = [2, 3, 4, 3]

S3 = [1, 7, 3, 4]
A3 = [1, 0]
S_3 = [2, 8, 4, 1]

R = np.zeros((3, STATE_DIM+ACTION_DIM+REWARD_DIM+DONE_DIM+STATE_DIM))

R, full = Store_State(R, 3, S, A, r, S_, Done)
print(R, full)
R, full= Store_State(R, 3, S1, A1, r, S_1, Done)
print(R, full)
R, full= Store_State(R, 3, S2, A2, r, S_2, Done)
print(R, full)


s, a, re, s_, done = Sample_State(R, 2)

print(f"s: \n{s}")
print(f"a: \n{a}")
print(f"re: \n{re}")
print(f"s_: \n{s_}")
print(f"done: \n{done}")