import subprocess
import random
import time
from keras.models import load_model
import numpy as np
np.random.seed()
map_settings = [32, 40, 48, 56, 64]

#halite.exe --no-logs --no-timeout --width 32 --height 32 "python DQN_reborn.py 0.01" "python DQN_reborn2.py 0.01"

def create_programs(game_number, epsilon):
    programs = []
    for i in range(NB_WORKERS):
        seed = np.random.randint(10000, 16000000)
        if game_number % 30 == 0 and game_number != 0:
            programs.append('halite.exe -s {} --no-logs --no-timeout --width {} --height {} "python DQN_reborn.py {}" "python DQN_reborn.py {}"'.format(seed, 32, 32, epsilon, (epsilon + epsilon*epsilon_decay)/2))
        else:
            programs.append('halite.exe -s {} --no-logs --no-timeout --no-replay --width {} --height {} "python DQN_reborn.py {}" "python DQN_reborn.py {}"'.format(seed, 32, 32, epsilon, (epsilon + epsilon*epsilon_decay)/2))
    return programs


NB_WORKERS = 1
epsilon_decay = 0.996
epsilon_min = 0.05
start = 0
epsilon = max(epsilon_min, epsilon_decay**start)


for i in range(start, 15000):
    print("\n=========== GAME NUMBER : " + str(i*NB_WORKERS) + '\n')

    start_time = time.time()
    programs = create_programs(i, epsilon)
    processes = [subprocess.Popen(program) for program in programs]
    # wait
    for process in processes:
        process.wait()


    epsilon *= epsilon_decay
    epsilon = max(epsilon_min, epsilon)

    # if NB_WORKERS == 1:
    #     time.sleep(2)
    # time.sleep(5)

    print("Episode time: " + str(time.time() - start_time)  + '\n')
