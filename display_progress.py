
import matplotlib.pyplot as plt
import scipy
import scipy.signal
import numpy as np
import math
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots


smoothing_factor = 15
# Design the Gaussian filter
def gaussian_filter_1d(sigma):
    # The filter radius is 3.5 times sigma
    rad = int(math.ceil(3.5 * sigma))
    sz = 2 * rad + 1
    h = np.zeros((sz,))
    for i in range(-rad, rad + 1):
        h[ i + rad] = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-(i * i)/(2 * sigma * sigma))
    h = h / np.sum(h)
    return [h]

file = open("progress.txt", "r")
file = file.readlines()

smoothing = True
filter = gaussian_filter_1d(smoothing_factor)

values = []
for line in file:
    line = line.split(' ')
    if len(line) > 1:
        values.append(float(line[-1]))



total_halite = [elt for index, elt in enumerate(values) if index%6==0]
total_reward = [elt for index, elt in enumerate(values) if index%6==1]
total_ships = [elt for index, elt in enumerate(values) if index%6==2]
total_reward_per_iteration = [elt for index, elt in enumerate(values) if index%6==3]
total_reward_per_iteration_normalized = [elt for index, elt in enumerate(values) if index%6==4]
eps = [elt for index, elt in enumerate(values) if index%6==5]
x = [i for i in range(len(total_halite))]
print(len(x))

if smoothing:
    total_halite = scipy.signal.convolve2d([total_halite], filter, mode='same')[0]
    total_reward = scipy.signal.convolve2d([total_reward], filter, mode='same')[0]
    total_ships = scipy.signal.convolve2d([total_ships], filter, mode='same')[0]
    total_reward_per_iteration = scipy.signal.convolve2d([total_reward_per_iteration], filter, mode='same')[0]
    total_reward_per_iteration_normalized = scipy.signal.convolve2d([total_reward_per_iteration_normalized], filter, mode='same')[0]
    x = [i for i in range(len(total_halite))]


fig = make_subplots(rows=6, cols=1)
fig.append_trace(go.Scatter(x=x, y=total_halite, mode='lines', name='final score'), row=1, col=1)
fig.append_trace(go.Scatter(x=x, y=total_reward, mode='lines', name='total reward'), row=2, col=1)
fig.append_trace(go.Scatter(x=x, y=total_ships, mode='lines', name='total halite normalized'), row=3, col=1)
fig.append_trace(go.Scatter(x=x, y=total_reward_per_iteration, mode='lines', name='reward per iteration'), row=4, col=1)
fig.append_trace(go.Scatter(x=x, y=total_reward_per_iteration_normalized, mode='lines', name='reward per iteration normalized'), row=5, col=1)
fig.append_trace(go.Scatter(x=x, y=eps, mode='lines', name='eps'), row=6, col=1)
fig.update_layout(height=1200, width=1500, title_text="DQN IS 2")
plotly.offline.plot(fig, filename = 'progress.html', auto_open=False)
fig.show()
