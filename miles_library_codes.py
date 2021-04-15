#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: mdebs
Base code author: gregz
"""

from astropy.convolution import Gaussian1DKernel, convolve
from scipy.interpolate import interp1d
import numpy as np
from astropy.io import fits
import os.path as op
import glob
from sklearn.decomposition import PCA

def get_pca_miles(n_components=50, 
                  lp='C:\\Users\\mdebs\\OneDrive\\Desktop\\VIRUS'):
    basedir = lp
    file1 = open(op.join(basedir, 'C:\\Users\\mdebs\\OneDrive\\Desktop\\VIRUS\\paramsMILES_v9.1.txt'), 'r')
    Lines = file1.readlines()
    info = []
    for line in Lines:
        l = line.strip()
        n = l[23:27]
        t, g, m = l[51:72].split()
        if m == '--':
            m = 0.0
        if t == '--':
            continue
       
        t = float(t)
        g = float(g)
        m = float(m)
        info.append([n, t, g, m])
    
    # define wavelength
    G = Gaussian1DKernel(1.0)
    wave = np.linspace(3470, 5540, 1036)
    models = np.zeros((len(info), len(wave)))
    i_array = np.zeros((len(info), 3))
    for i in np.arange(len(info)):
        g = fits.open(op.join(basedir, 's%s.fits' % info[i][0]))
        waveo = (g[0].header['CRVAL1'] + np.linspace(0, len(g[0].data[0])-1,
                                                 len(g[0].data[0]))
                 * g[0].header['CDELT1'])
        I = interp1d(waveo, g[0].data[0], kind='quadratic', bounds_error=False,
                     fill_value=np.nan)
        models[i] = convolve(I(wave), G, preserve_nan=True)
        i_array[i] = info[i][1:]
    
    new_models = models * 1.    
    new_models[np.isnan(new_models)] = 0.0
    M = new_models.mean(axis=1)
    S = new_models.std(axis=1)
    z = (new_models - M[:, np.newaxis]) / S[:, np.newaxis]
    pca = PCA(n_components=n_components).fit(z)
    H = pca.components_
    return H

#def get_pca_fsps1(n_components=50, zbins=np.linspace(0.01, 0.5, 50), 
                 #wave=np.linspace(3470, 5540, 1036), 
                 #orig_wave=np.linspace(2300, 5540, 1621),
                 #lp='/Users/gregz/cure/Remedy/ssp.fits'):
    #f = fits.open(lp)
    #models = f[0].data
    #new_models = models * 1.    
    #new_models[np.isnan(new_models)] = 0.0
    #M = new_models.mean(axis=1)
    #S = new_models.std(axis=1)
    #z = (new_models - M[:, np.newaxis]) / S[:, np.newaxis]
    #pca = PCA(n_components=n_components).fit(z)
    #H = pca.components_
    #M = np.zeros((len(zbins), H.shape[0], len(wave)))
    #for j in np.arange(H.shape[0]):
        #I = interp1d(orig_wave, H[j], kind='quadratic', bounds_error=False, 
                    # fill_value=0.0)
        #for i, z in enumerate(zbins):
            #M[i, j] = I(wave/(1.+z))
    #return M

def get_pca_dr1qso(n_components=50, zbins=np.linspace(0.01, 0.5, 50), 
                 wave=np.linspace(3470, 5540, 1036), 
                 lp='C:\\Users\\mdebs\\OneDrive\\Desktop\\VIRUS\\redrock\\py\\redrock\\templates\\rrtemplate-qso.fits'):
    f = fits.open(lp)
    models = f[0].data
    orig_wave = 10**(f[0].header['CRVAL1']+np.arange(models.shape[1])*
                     f[0].header['CDELT1'])
    M = np.zeros((len(zbins), models.shape[0], len(wave)))
    for j in np.arange(models.shape[0]):
        I = interp1d(orig_wave, models[j], kind='quadratic', bounds_error=False, 
                     fill_value=0.0)
        for i, z in enumerate(zbins):
            M[i, j] = I(wave/(1.+z))
    M = (M - np.mean(M, axis=(0, 2))[np.newaxis, :, np.newaxis]) / np.std(M, axis=(0, 2))[np.newaxis, :, np.newaxis]
    return M

def get_pca_fsps(n_components=50, zbins=np.linspace(0.01, 0.5, 50), 
                 wave=np.linspace(3470, 5540, 1036), 
                 lp='C:\\Users\\mdebs\\OneDrive\\Desktop\\VIRUS\\redrock\\py\\redrock\\templates\\rrtemplate-galaxy.fits'):
    f = fits.open(lp)
    models = f[0].data
    orig_wave = (f[0].header['CRVAL1']+np.arange(models.shape[1])*
                 f[0].header['CDELT1'])
    M = np.zeros((len(zbins), models.shape[0], len(wave)))
    for j in np.arange(models.shape[0]):
        I = interp1d(orig_wave, models[j], kind='quadratic', bounds_error=False, 
                     fill_value=0.0)
        for i, z in enumerate(zbins):
            M[i, j] = I(wave/(1.+z))
    M = (M - np.mean(M, axis=(0, 2))[np.newaxis, :, np.newaxis]) / np.std(M, axis=(0, 2))[np.newaxis, :, np.newaxis]
    return M
