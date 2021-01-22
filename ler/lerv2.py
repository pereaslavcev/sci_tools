import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

from scipy.signal import argrelextrema, welch
from scipy.optimize import curve_fit
from scipy.stats import zscore

from skimage import io

'''
загружаем данные
'''
img = io.imread('B601---1.jpg', as_gray=True)
img = img // np.mean(img)
row, col = img.shape

c = []
dc = []
for i in range(row):
    s = img[i, :] # строка
    ds = abs(np.diff(s)) # модуль производной строки
    p = argrelextrema(ds, np.greater)[0] # позиция края
    c.append(p) # пре ler
    dp = np.diff(p) # разница позиции края
    dc.append(dp) # пре lwr

c = pd.DataFrame(c)
c = c.fillna(c.median())
dc = pd.DataFrame(dc)
dc = dc.fillna(dc.median())

'''
убираем выбросы из пре ler
'''
z_scores = zscore(c)
abs_z_scores = abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)
c = c[filtered_entries]

'''
убираем выбросы из пре lwr
'''
z_scores = zscore(dc)
abs_z_scores = abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)
dc = dc[filtered_entries]

#'''
#рисуем статистику линейных размеров
#'''
#plt.subplot(311)
#plt.boxplot(dc[1])
#
#plt.subplot(312)
#i = 4
#while i <= dc.shape[1]:
#    plt.boxplot(dc[i], positions=[i])
#    i += 4
#plt.ylabel('размер в пикселях')
#
#plt.subplot(313)
#j = 6
#while j <= dc.shape[1]:
#    plt.boxplot(dc[j], positions=[j])
#    j += 4
#plt.xlabel('номер зазора')
#
#plt.tight_layout()

'''
считаем psd функцию
'''
f, psd = welch(c[0], window='blackmanharris', nperseg=512, 
               detrend='linear', average='median')

def g(f, s, T):
    return (s**2 * T / 2 * np.pi**0.5) * np.exp(-f**2 * T**2 / 4)

def e(f, s, T):
    return (s**2 * T / np.pi) * (1 / (1 + f**2 * T**2))

popt, pcov = curve_fit(e, f, psd)
fe = np.linspace(min(f), max(f), len(f) * 2)
psde = e(fe, *popt)

plt.loglog(f, psd)
plt.loglog(fe, psde)
plt.title('ler = %.1f px, corr = %.1f px' % (popt[0]*3, popt[1]))
plt.xlabel('f')
plt.ylabel('psd')

plt.tight_layout()

#from lmfit.models import LinearModel, ExponentialModel
#from lmfit.models import GaussianModel, LorentzianModel, VoigtModel
#
#line = LinearModel(prefix='line_')
#exp = ExponentialModel(prefix='exp_')
#gauss = GaussianModel(prefix='gauss_')
#lorentz = LorentzianModel(prefix='lorentz_')
#voigth = VoigtModel(prefix='voigth_')
#
#pars = exp.make_params()
#pars += voigth.guess(psd, x=f)
#
#mod = voigth + exp
#out = mod.fit(psd, pars, x=f)
#comps = out.eval_components()
#
#print(out.fit_report(show_correl=False))
#
#plt.loglog(f, psd)
#plt.loglog(f, out.best_fit)
#plt.loglog(f, comps['exp_'])
#
#plt.tight_layout()