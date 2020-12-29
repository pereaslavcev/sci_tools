import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.signal import argrelextrema as ext
from scipy.stats import linregress as reg
from skimage import io

img=io.imread('B601---1.jpg',as_gray=True)
img=img//np.mean(img)
row,col=img.shape

c=[]
dc=[]
for i in range(row):
    s=img[i,:]
    ds=np.abs(np.diff(s))
    p=ext(ds,np.greater)[0]
    c.append(p)
    dp=np.diff(p)
    dc.append(dp)

c=pd.DataFrame(c)
c=c.fillna(c.median())
dc=pd.DataFrame(dc)
dc=dc.fillna(dc.median())

# PSD calculation
power, freqs = plt.psd(dc[10],len(dc[10]),detrend='mean')

# Linear fit
x = np.log10(freqs[1:])
y = np.log10(power[1:])

slope, inter, r2, p, stderr = reg(x, y)

print(slope, inter)

# Plot
line = (inter + slope * (10 * np.log10(freqs)))
plt.semilogx(freqs, line)