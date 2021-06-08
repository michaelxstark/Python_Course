import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import sounddevice as sd

# reading in data
df = pd.read_csv('CovidFaelle_Timeline.csv', sep=';')

# displaying first ten, checking columns
df.head(10)
lof = list(df.columns)
lof

# cleaning data

# changing to proper floats
df['SiebenTageInzidenzFaelle'] = df['SiebenTageInzidenzFaelle'].str.replace(',', '.')
df['SiebenTageInzidenzFaelle'].head(20)

# getting rid of every tenth value (whole of austria)
df = df.loc[df['BundeslandID'] % 10 != 0, :].reset_index()
df.tail(10)


# scaling values for volume, size and color


BIDS = list(set(df['BundeslandID']))
AmpSize = [np.array(df.loc[(df['BundeslandID'] == bid), 'AnzahlFaelle'])
           for bid in BIDS]


Color = [np.array(df.loc[(df['BundeslandID'] == bid),
                         'SiebenTageInzidenzFaelle']) for bid in BIDS]


Pitch = [np.array(df.loc[(df['BundeslandID'] == bid),
                         'SiebenTageInzidenzFaelle']) for bid in BIDS]

# Converter function to floats


def strtofloat(a):
    for i in range(len(a)):
        a[i] = float(a[i])
    return a


Color2 = [np.apply_along_axis(strtofloat, 0, Color[j])
          for j in range(len(Color))]

# scale funtion for 'colorchanges' and pitchchanges
Pitch2 = [np.apply_along_axis(strtofloat, 0, Pitch[j])
          for j in range(len(Pitch))]
Pitch2


def pv(ar):
    for i in range(len(ar)):
        if ar[i] < 50:
            ar[i] = 1
        elif 50 < ar[i] < 150:
            ar[i] = 1.5
        elif 100 < ar[i] < 300:
            ar[i] = 2
        elif ar[i] > 300:
            ar[i] = 3
    return ar


Pitches = [np.apply_along_axis(pv, 0, Pitch2[j])
           for j in range(len(Pitch2))]


def cc(ar):
    for i in range(len(ar)):
        if ar[i] < 50:
            ar[i] = 'green'
        elif 50 < ar[i] < 150:
            ar[i] = 'yellow'
        elif 100 < ar[i] < 300:
            ar[i] = 'orange'
        elif ar[i] > 300:
            ar[i] = 'red'
    return ar


Color3 = [np.apply_along_axis(cc, 0, Color2[j])
          for j in range(len(Color2))]


# last preparations

AmpSize1 = []
for i in range(len(AmpSize[0])):
    for j in range(len(AmpSize)):
        AmpSize1.append(AmpSize[j][i])
AmpSize1

Color4 = []
for x in range(len(Color3[0])):
    for y in range(len(Color3)):
        Color4.append(Color3[y][x])


# AudioEngine

sd.query_devices()
sd.default.device = 'BlackHole 16ch, Core Audio'


def puresine(freq, dur, phase):
    sr = 44100
    phase1 = phase * np.pi
    t = np.arange(dur * sr) / sr
    sine = 1 * np.sin(2 * np.pi * freq * t + phase1)
    return sine


# simple panning - algorithm
def panner(x, angle):
    # pan a mono audio source into stereo
    # x is a numpy array, angle is the angle in radiants
    left = np.sqrt(2)/2.0 * (np.cos(angle) - np.sin(angle)) * x
    right = np.sqrt(2)/2.0 * (np.cos(angle) + np.sin(angle)) * x
    return np.dstack((left, right))[0]


# Scaling to values between 0 and 1
Amps = [np.array(df.loc[(df['BundeslandID'] == bid), 'AnzahlFaelle'])
        for bid in BIDS]
Amps2 = [Amps[i] / Amps[i].max() for i in range(len(Amps))]


sr = 44100
splits = len(Amps2[0])
dur = splits / 5
dur
global line
line = int(round((sr * dur) / splits, 0))
Amps2N = [np.append(Amps2[u], [0]) for u in range(len(Amps2))]
Pitches2 = [np.append(Pitches[u], [Pitches[u][-1]])
            for u in range(len(Pitches))]
basefreqs = [110, 110 * 1.5, 220, 440 * (15/8),
             550, 440 * (3/4), 880 * (9/8), 990,
             880]
Pitches3 = [Pitches2[i] * basefreqs[i] for i in range(len(Pitches2))]

pitch = [np.concatenate([np.linspace(Pitches3[j][i], Pitches3[j][i + 1], line)
                         for i in range(len(Pitches3[j]) - 1)])
         for j in range(len(Pitches3))]


env = [np.concatenate([np.linspace(Amps2N[j][i], Amps2N[j][i + 1], line)
                       for i in range(len(Amps2N[j]) - 1)])
       for j in range(len(Amps2N))]


def summation(callback, freqs):
    # Cumulative Sum
    phaseY = np.cumsum(freqs)
    # sin (cumulative sum (f) )
    x = np.sin((phaseY) * np.pi * 2 / sr)
    return x


longsines = [summation(np.sin, pitch[i]) * env[i] for i in range(len(pitch))]


# plot

plt.style.available
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(20, 8))
ax.set_xticks(df['BundeslandID'][:9])
ax.set_xticklabels(list(df['Bundesland'][:9]))
ax.set_ylim(-2, 12)
ax.set_frame_on(False)
ax.axes.get_yaxis().set_visible(True)
ax.axes.get_xaxis().set_visible(True)
ax.set_yticklabels([])
ax.grid(False, axis='both')
ax.set_title('Covid19_Cases_in_Austria')

x = np.array(list(set(df['BundeslandID'])))
y = [4, 3, 2, 5, 4, 6, 7, 3, 6]


lines = ax.scatter(x, y,
                   marker='o',
                   s=50,
                   c='green', alpha=0.8)

plt.close()


def animate(i):
    lines.set_sizes(np.array(AmpSize1[i:i+9]) * 5)
    lines.set_color(Color4[i:i+9])
    ax.set_ylabel(df['Time'][i][:10])
    if i == len(df) - 9:
        sd.play((panner(longsines[0], np.radians(-50)) +
                 panner(longsines[1], np.radians(0)) +
                 panner(longsines[2], np.radians(50)) +
                 panner(longsines[3], np.radians(10)) +
                 panner(longsines[4], np.radians(20)) +
                 panner(longsines[5], np.radians(-20)) +
                 panner(longsines[6], np.radians(-30)) +
                 panner(longsines[7], np.radians(5)) +
                 panner(longsines[8], np.radians(-5))) * 0.25, sr)
    return lines,


animation = FuncAnimation(fig, func=animate,
                          frames=np.arange(27, len(df), 9),
                          interval=(dur / splits) * 1000,
                          blit=False, repeat=False)


HTML(animation.to_html5_video())
