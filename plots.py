#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 14:23:01 2018

@author: veronikasamborska
"""
import numpy as np
import pylab as plt
import matplotlib.pyplot as plot
import seaborn as sns
import pandas as pd
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.tools import FigureFactory as FF
import pandas as pd
import scipy
from scipy import signal
from scipy.signal import butter, lfilter, freqz, filtfilt
from scipy.fftpack import fft, ifft
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def session_plot(session): 
    
# Set frequency band 
    low_pass = False 
    high_pass = True 
    sub_sample_1Hz = False
    gcamp = session.get('gcamp') # Values for gcamp
    rfp = session.get('rfp')# Values for rfp
    #demedian_gcamp = gcamp - np.median(gcamp) 
    #demedian_rfp = rfp - np.median(rfp)
    fs = session.get('fs') # Sampling frequency
    order = 6
    wn_10 = 20/(fs/2) #10 Hz filter 
    wn_2 = 2/(fs/2) # 2 Hz filter
    b, a = butter(order, wn_10, 'low', analog = False) # Low - pass filter(attenuates signals above certain threshold 
    output_signal_gcamp = filtfilt(b, a, gcamp) # Passes frequencies below 10 in gcamp
    sub_sampl_gcamp = output_signal_gcamp[::2] # Sampling at 1 Hz
    output_signal_rfp = filtfilt(b, a, rfp) # Passes frequencies below 10 in rfp
    sub_sampl_rfp = output_signal_rfp[::2] # Sampling at 1 Hz 
    y, x = butter(order, wn_2, 'low', analog = False)  # High - pass filter(attenuates signals below a certain threshold)
    output_signal_gcamp_2Hz = filtfilt(y, x, gcamp)
    output_signal_rfp_2Hz = filtfilt(y, x, rfp)
    
    # Plot 2D histogram (heatmap)
    if  low_pass == True and sub_sample_1Hz == True: 
        plot.hist2d(sub_sampl_gcamp, sub_sampl_rfp, bins=(100, 100), cmap=plt.cm.jet)
    elif low_pass == True and sub_sample_1Hz == False:
        plot.hist2d(output_signal_gcamp, output_signal_rfp, bins=(100, 100), cmap=plt.cm.jet)
    elif high_pass == True: 
        plot.hist2d(output_signal_gcamp_2Hz, output_signal_rfp_2Hz, bins=(100, 100), cmap=plt.cm.jet)

    # Frequency band to be used for linear regression 
    if low_pass == True and sub_sample_1Hz == True: 
        gcamp = sub_sampl_gcamp
        rfp = sub_sampl_rfp
    elif low_pass == True and sub_sample_1Hz == False:
        gcamp = output_signal_gcamp
        rfp = output_signal_rfp
    elif high_pass == True:
        gcamp = output_signal_gcamp_2Hz
        rfp = output_signal_rfp_2Hz
        
    # Linear Regression 
    lm = LinearRegression()  
    rfp_2d = np.array(rfp)[:,np.newaxis]
    ones_rfp2 = np.ones(len(rfp)).reshape(len(rfp),1)
    rfp_2d = np.hstack((rfp_2d, ones_rfp2))
    gcamp_2d = np.array(gcamp)[:,np.newaxis]
    ones_gcamp2 = np.ones(len(gcamp)).reshape(len(gcamp),1)
    gcamp_2d = np.hstack((gcamp_2d, ones_gcamp2))        
    lm.fit(rfp_2d, gcamp_2d) # Fitting gcamp from rfp signal 
    gcamp_predict = lm.predict(rfp_2d) # Predicting gcamp from rfp signal
    gcamp_resid = (gcamp_2d - gcamp_predict)
    plt.figure()
    plt.plot(gcamp_resid[:,0], 'blue')
    plt.plot(gcamp,'yellow')
    plt.plot(gcamp_predict[:,0],'green')
    print('Variance score: %.2f' % r2_score(gcamp_2d[:,0], gcamp_predict[:,0]))
    #print("Mean squared error: %.5f" % mean_squared_error(gcamp_2d[:,0], gcamp_predict[:,0]))
