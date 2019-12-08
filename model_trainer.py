import subprocess
import random
import time
import pickle
from keras.models import load_model
import numpy as np
import tensorflow as tf
import keras
from keras import backend as k
from keras.models import Model, load_model, clone_model
from keras.layers import Input, Dense, Lambda, Conv2D, Flatten
from keras import optimizers
from keras.layers.merge import Add
from PrioritizedExperienceReplayBuffer import *
import os
import os.path
import glob
import torch
import pprofile
from multiprocessing import Pool


def huber_loss(y_true, y_pred):
    return tf.losses.huber_loss(y_true,y_pred)

def create_model(input_shape, action_size, lr):
    input_layer = Input(shape=input_shape)
    x = Conv2D(filters=32, kernel_size=[4,4], strides=(2, 2), activation='relu')(input_layer)
    x = Conv2D(filters=64, kernel_size=[3,3], activation='relu')(x)
    x = Conv2D(filters=64, kernel_size=[2,2], activation='relu')(x)
    flatten = Flatten()(x)

    fc1 = Dense(512, activation='relu')(flatten)
    advantage = Dense(action_size, activation='linear')(fc1)

    fc2 = Dense(512, activation='relu')(flatten)
    value = Dense(1, activation='linear')(fc2)

    advantage = Lambda(lambda advantage: advantage - tf.reduce_mean(advantage, axis=-1, keepdims=True))(advantage)
    value = Lambda(lambda value: tf.tile(value, [1, action_size]))(value)
    q_value = Add()([value, advantage])

    model = Model(inputs=[input_layer], outputs=[q_value])
    opt = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=0.00015)
    model.compile(loss=huber_loss, optimizer=opt, metrics=['mae'])
    return model


###################################
# TensorFlow wizardry
config = tf.ConfigProto()
# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True
# Only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 0.3
# Create a session with the above options specified.
k.tensorflow_backend.set_session(tf.Session(config=config))
######################################################################


loadmodel = True
loadmemory = False
memory_size = 400000
min_memory_size = 100000
lr = 0.000125
input_shape = (31, 31, 3)
action_size = 5
nb_update_by_file = 250
batch_size = 32
discount_factor = 0.99
total_nb_update = 0
update_target_frequency = 5000
saving_model_frequency = 1500
indexes = np.arange(batch_size)
count_file = 0

#Getting model
if loadmodel:
    print("------------Loading model from file-----------")
    model = load_model('my_model.h5', custom_objects={'huber_loss': huber_loss, 'tf': tf})
    model.load_weights("my_model_weights")
    target_model = create_model(input_shape, action_size, lr)
    for i in range(len(model.layers)):
        target_model.layers[i].set_weights(model.layers[i].get_weights())

else:
    print("------------Building model-----------")
    model = create_model(input_shape, action_size, lr)
    target_model = create_model(input_shape, action_size, lr)
    for i in range(len(model.layers)):
        target_model.layers[i].set_weights(model.layers[i].get_weights())
    model.save('my_model.h5')
    model.save_weights("my_model_weights")

# Getting Memory
if loadmemory:
    memory = pickle.load(open("memory.data", "rb"))
else:
    memory = Memory(memory_size)
    pickle.dump(memory, open("memory.data", 'wb'))

print(model.summary())
print("Training...")
profiler = pprofile.Profile()


while True:
    print("Loading experience file")
    try:
        files = glob.glob("experiences/*")
        try:
            sorted_files = sorted([(time.ctime(os.path.getctime(file)), file) for file in files], key=lambda x:x[0])
            if len(sorted_files) > 200:
                for i in range(len(sorted_files)- 200):
                    date, old_file = sorted_files[i] #le plus vieux
                    #os.remove(old_file)
        except FileNotFoundError:
            time.sleep(0.01)
            continue
        if len(sorted_files) != 0:
            date, file = sorted_files[-1] #le plus récent
            print(file)
            game_data = pickle.load(open(file, "rb"))
        else:
            time.sleep(0.01)
            continue
    except EOFError:
        time.sleep(0.01)
        continue
    data_size = len(game_data)
    print("File of size {}".format(data_size))

    for experience in game_data:
        memory.add(experience[0], experience[1])
    #os.remove(file)

    if memory.tree.n_entries >= min_memory_size:
        count_file += 1
      #  with profiler:
        nb_update = int(nb_update_by_file * data_size / 1000)
        print("Training for {} steps".format(nb_update))
        start = time.time()
        for j in range(nb_update):
            mini_batch, idxs, is_weight = memory.sample(batch_size)
            mini_batch = np.array(mini_batch).transpose()

            states = np.vstack(mini_batch[0])
            actions = list(mini_batch[1])
            rewards = list(mini_batch[2])
            next_states = np.vstack(mini_batch[3])
            valid_actions = np.vstack(mini_batch[4])

            # Q function of current state
            target_vec = model.predict(states)

            # Selection only the Q vvalue of the executed action
            pred = target_vec[indexes, actions]

            # Q function of next state
            next_pred = target_model.predict(next_states)
            next_pred[~valid_actions] = -np.inf #ne prend pas en compte les non valids dans le max

            # Q Learning: get maximum Q value at s' from target model
            target = rewards + discount_factor * next_pred.max(1)
            errors = abs(pred - target)
            target_vec[indexes, actions] = target

            

            model.fit(states, target_vec, sample_weight=is_weight, shuffle=False, epochs=1, verbose=0, batch_size=batch_size)

            # update priority
            memory.update(idxs, errors)

            if (total_nb_update + j) % update_target_frequency == 0 and total_nb_update != 0:
                print("Synchronizing model and target model...")
                for i in range(len(model.layers)):
                    target_model.layers[i].set_weights(model.layers[i].get_weights())

            if (total_nb_update + j) % saving_model_frequency == 0 and total_nb_update != 0:
                print("Saving model weigths...")
                model.save_weights("my_model_weights")


        print("{} time by step".format((time.time() - start)/(nb_update+1)))
        total_nb_update += nb_update
        if count_file % 300 == 0:
            model.save('my_model_{}-t4.h5'.format(count_file))
        # profiler.print_stats()
        # # Or to a file:
       # profiler.dump_stats("profiler_stats.txt")