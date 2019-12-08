#!/usr/bin/env python3
# Python 3.6
import os
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
import sys
stdout = sys.stderr
sys.stderr = open(os.devnull, 'w')
import numpy as np
import keras
from keras import backend as k
from keras.models import Model, load_model
from keras.layers import Input, Dense, Lambda, Conv2D, Flatten
from keras import optimizers
from keras.layers.merge import Add
import tensorflow as tf
from math import *
import random
import pickle
import logging
import hlt
from hlt import constants
from hlt.positionals import Direction, Position
from hlt.entity import Dropoff
import time
sys.stderr = stdout

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)


###################################
# TensorFlow wizardry
config = tf.ConfigProto()
# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True
# Only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 0.05
# Create a session with the above options specified.
k.tensorflow_backend.set_session(tf.Session(config=config))
###################################

def compute_state(global_state, ship):
    state = global_state[ship.position.y+1:ship.position.y+32, ship.position.x+1:ship.position.x+32, :]
    return state


def compute_global_state(game_map, me):
    """BUT: retourner un vecteur qui représente bien l'état de la partie"""
    sight_window = 16
    game_state = np.zeros((3, 32, 32))

    for i in range(sight_window * 2):
        for j in range(sight_window * 2):
            #halite
            cell = game_map[Position(i, j)]
            game_state[0, i, j] = cell.halite_amount
            #dropoff
            if cell.structure:
                if cell.structure.owner == me.id:
                    game_state[1, i, j] = 1
            #ships
            if cell.ship:
                if me.has_ship(cell.ship.id):
                    game_state[2, i, j] = cell.ship.halite_amount + 1000
                else:
                    game_state[2, i, j] = -2000

    game_state[0] = np.around(game_state[0] / 1000, 3)
    game_state[2] = game_state[2] / 2000
    game_state = np.transpose(game_state, (2, 1, 0))
    game_state = np.pad(game_state, ((16, 16), (16, 16), (0,0)), mode='wrap')
    return game_state

def valid_move(ship, game_map):
    #Déplacement sur la map et calcul du nouveau halite amount et modification de la position du ship
    cell = game_map[ship]
    ship_halite_amount = ship.halite_amount - ceil(0.1*cell.halite_amount)
    if ship_halite_amount < 0: #Can't move
        return False
    return True

def valid_stay_still(ship, game_map):
    if game_map[ship].structure or ship.halite_amount > 850 or ship.position in futur_position:
        return False
    return True

def valid_stay_still_true(ship, game_map):
    if game_map[ship].structure or ship.position in futur_ally_position:
        return False
    return True

def valid_turn_into_dropoff(ship, game_map, my_halite_amount):
    if (my_halite_amount + ship.halite_amount + game_map[ship].halite_amount) < 4000 or game_map[ship].structure != None:
        return False #Trop cher
    sight_window = 11
    surronding_halite = 0
    for i in range(-(sight_window//2), sight_window//2 + 1):
        for j in range(-(sight_window//2) + abs(i), sight_window//2 - abs(i) + 1):
            surronding_halite += game_map[Position(i + ship.position.x, j + ship.position.y)].halite_amount
    if surronding_halite < 9000:
        return False #Pas assez de halite à coté
    return True

def compute_valid_action(ship, futur_position, game_map, me):
    valid_movement = valid_move(ship, game_map)
    if valid_movement:
        valid_action = [not get_new_position(0, ship) in futur_position, not get_new_position(1, ship) in futur_position, not get_new_position(2, ship) in futur_position,
                        not get_new_position(3, ship) in futur_position, valid_stay_still(ship, game_map)]

        if not any(valid_action):
            valid_action = [not get_new_position(0, ship) in futur_ally_position, not get_new_position(1, ship) in futur_ally_position, not get_new_position(2, ship) in futur_ally_position,
                            not get_new_position(3, ship) in futur_ally_position, valid_stay_still_true(ship, game_map)]
            if not any(valid_action):
                        valid_action = [True, True, True, True, True]
    else:
        valid_action = [False, False, False, False, True]

    return valid_action

def get_new_position(action, ship):
    return game_map.normalize(Position(ship.position.x + directions[action][0], ship.position.y + directions[action][1]))

def get_next_pos_halite_amount(ship, game_map):
    next_pos_halite_amount = np.zeros(5)
    for index, direction in enumerate(directions):
        next_pos_halite_amount[index] = game_map[Position(ship.position.x + direction[0], ship.position.y + direction[1])].halite_amount
    return next_pos_halite_amount

def huber_loss(y_true, y_pred):
    return tf.losses.huber_loss(y_true,y_pred)

def predict_futur_enemy_pos(my_dropoffs, nb_enemy_ships, nb_enemy, nb_my_ships):
    for enemy in enemies:
        for ship in enemy.get_ships():
            if ship.halite_amount > 800 and (nb_enemy_ships / nb_enemy) <= nb_my_ships: #Bateau très rempli et on a bcp de ships, on s'en fou de la collision
                continue
            if ship.halite_amount < ceil(0.1*game_map[ship.position].halite_amount): # le bateau ne peut pas bouger
                futur_position.add(ship.position)
            elif ship.position in my_dropoffs:
                continue
            else:
                next_pos_halite_amount = get_next_pos_halite_amount(ship, game_map)
                if max(next_pos_halite_amount[:4]) < next_pos_halite_amount[4] / 2 and next_pos_halite_amount[4] > 300: #Je suppose qu'il va rester sur place
                    futur_position.add(ship.position)
                else: # je sais pas je mets tout
                    for direction in directions:
                        futur_position.add(game_map.normalize(Position(ship.position.x + direction[0], ship.position.y + direction[1]))) # Je suppose qu'il bouge à une meilleur case

    for my_dropoffs_pos in my_dropoffs:
        futur_position.discard(my_dropoffs_pos)

def compute_reward(previous_state, ship):
    previous_halite_amount = previous_state[15, 15, 2] * 2000 - 1000
    gain_halite = ship.halite_amount - previous_halite_amount
    if ship.position in my_dropoffs:#objectif principal
        reward = - gain_halite / 1000 # dans [0,1]
    else:
        if gain_halite < 0: #cout de déplacement
            reward = gain_halite / 1000 # dans [-1, 0[
        else:
            reward = 0

    return gain_halite, reward


gamma = 0.99
eps = float(sys.argv[1])
directions = [Direction.North, Direction.South, Direction.East, Direction.West, Direction.Still]
actions = np.array([0, 1, 2, 3, 4])

go2 = True
while go2:
    try:
        model = load_model('my_model.h5', custom_objects={'huber_loss': huber_loss, 'tf': tf})
        model.load_weights("my_model_weights")
        go2 = False
    except:
        continue


secondDropoff = False
total_halite = 0
total_reward = 0
counter = 0
nb_of_ships = 0
hash_dim_turn = {32:400, 40:425, 48:450, 56:475, 64:500}
np.random.seed()
hash_previous_state = dict()
experiences = []
previous_my_ships = []

game = hlt.Game()
total_initial_halite = sum([game.game_map[Position(i,j)].halite_amount for i in range(32) for j in range(32)])
total_current_halite = total_initial_halite
last_modified_model_time =  time.ctime(os.path.getmtime("my_model_weights"))
# This game object contains the initial game state.
game.ready("DQN-REBORN")
""" <<<Game Loop>>> """
while True:
    game.update_frame()
    #update model weights from trainer
    curr_modified_model_time = time.ctime(os.path.getmtime("my_model_weights"))
    if curr_modified_model_time != last_modified_model_time:
        go = True
        while go:
            try:
                logging.info("loading new weights")
                model.load_weights("my_model_weights")
                last_modified_model_time = curr_modified_model_time
                go = False
            except:
                continue
    me = game.me
    enemies = [game.players[id] for id in game.players.keys() if id != me.id]
    game_map = game.game_map
    my_halite_amount = me.halite_amount
    command_queue = []
    futur_position = set()
    futur_ally_position = set()
    new_state = np.array([])
    state = np.array([])
    my_ships = me.get_ships()
    nb_my_ships = len(my_ships)
    nb_enemy = len(enemies)
    nb_enemy_ships = sum([len(player.get_ships()) for player in enemies])
    my_dropoffs = me.get_dropoffs() + [me.shipyard]
    my_dropoffs = [drop.position for drop in my_dropoffs]
    max_turn_number = hash_dim_turn[game_map.width]
    ally_collision_pos = set()
    previous_ally_collision_pos = set()
    state = []
    global_state = compute_global_state(game_map, me)
    total_current_halite = round(np.sum(global_state[:32, :32, 0]) * 1000, 0)

    #On ajoute aux futurs positions, tous les ships qui ne vont pas bouger
    for ship in my_ships:
        if ship.halite_amount < ceil(0.1*game_map[ship.position].halite_amount):
            futur_position.add(ship.position)
            futur_ally_position.add(ship.position)
        else:
            ship.distance_dropoff = min(np.array([game_map.calculate_distance(ship.position, dropoff) for dropoff in my_dropoffs]))

    #On ajoute les futurs position des adversaire pour éviter les collisions
    predict_futur_enemy_pos(my_dropoffs, nb_enemy_ships, nb_enemy, nb_my_ships)


    #On itère sur les agents
    for ship in sorted(my_ships, key=lambda x: (x.distance_dropoff, -x.halite_amount)):
                ######################   SELECTION OF ACTION #####################
        valid_action_bool = compute_valid_action(ship, futur_position, game_map, me)
        valid_action = actions[valid_action_bool]
        state = compute_state(global_state, ship)
        state = np.expand_dims(state, axis=0)
        target_vec = model.predict(state)[0]
        random = np.random.random()
        if random < eps:
            #Exploration
            index = np.random.randint(0, len(valid_action))
            a = valid_action[index]

        else:
            #Exploitation
            target_vec_valid = target_vec[valid_action_bool]
            index = np.argmax(target_vec_valid)
            a = valid_action[index]

        #On l'ajoute l'action à la command queue
        command_queue.append(ship.move(directions[a]))
        new_pos = get_new_position(a, ship)
        futur_position.add(new_pos)
        futur_ally_position.add(new_pos)

        if ship.id in hash_previous_state:
            previous_state, previous_target_vec, previous_action = hash_previous_state[ship.id]
            gain_halite, reward = compute_reward(previous_state[0], ship)
            total_reward += reward
            total_halite += gain_halite
            counter += 1
            target = reward + gamma * np.max(target_vec[valid_action_bool])
            td_error = abs(previous_target_vec[previous_action] - target)
            experiences.append([td_error, (previous_state, previous_action, reward, state, valid_action_bool)])

            if len(experiences) == 2000:
                # put max priority to the experiences because they are new
                pickle.dump(experiences, open("experiences/experiences_{}.pickle".format(random), "wb"))
                experiences = []

        hash_previous_state[ship.id] = (state, target_vec, a)


    if (total_current_halite/(len(my_ships) + nb_enemy_ships + 1)) > 1500 and game.turn_number < int(max_turn_number*0.85) and my_halite_amount >= constants.SHIP_COST and not game_map[me.shipyard].position in futur_ally_position:
        command_queue.append(game.me.shipyard.spawn())
        nb_of_ships += 1

    if game.turn_number == max_turn_number:
        # put max priority to the experiences because they are new
        pickle.dump(experiences, open("experiences/experiences_{}.pickle".format(np.random.random()), "wb"))
        with open('progress.txt', "a") as myfile:
            myfile.write("\n-------------\nHalite won : " + str(me.halite_amount) + "\n")
            myfile.write("Total reward : " + str(total_reward) +"\n")
            myfile.write("Halite ratio : " + str(float(me.halite_amount)*100/total_initial_halite) +"\n")#halite won normalized
            myfile.write("Reward per iteration : " + str(total_reward/float(counter)) +"\n")
            myfile.write("Reward per iteration normalized: " + str(total_reward/(float(counter*total_initial_halite))) +"\n")
            myfile.write("Espilon: " + str(eps) +"\n")
    game.end_turn(command_queue)
