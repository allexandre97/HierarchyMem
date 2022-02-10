#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 13:38:44 2022

@author: xandre
"""

import HierarchyMem as hm
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable as mal
from matplotlib.ticker import FormatStrFormatter

def bins(L):

    q25, q75 = np.percentile(L, [25, 75])
    iqr = q75 - q25
    binwidth = (2*iqr)/(len(L)**(1/3))

    binns = np.arange(min(L), max(L) + binwidth, binwidth)

    return binns

def graphTHI(THI):
    
    fig, ax = plt.subplots(1,2,figsize = (14,12))
    
    ax[0].set_title('Thickness Heatmap',fontsize = 16, fontweight = 'bold')
    im1 = ax[0].imshow(np.mean(THI[0:40],axis = 0).T, origin = 'lower', cmap = 'coolwarm', interpolation = 'gaussian', extent = [gridX[0], gridX[-1], gridX[0], gridX[-1]])
    divider1 = mal(ax[0])
    cax1 = divider1.append_axes('right',
                                size = '5%',
                                pad = 0.05)
    cbar1 = fig.colorbar(im1,
                         cax=cax1,
                         format = '%.2f')
    cbar1.set_label(r'$\mathbf{Thick. (\AA)}$',
                    fontsize = 11,
                    fontweight = 'bold',
                    rotation = 0,
                    labelpad = -15)
    cbar1.set_ticks(np.linspace(np.min(np.mean(THI[0:40],axis = 0)), np.max(np.mean(THI[0:40],axis = 0)),4),2)
    cbar1.ax.tick_params(labelsize=14)
    
    thi = THI.reshape(THI.shape[0]*THI.shape[1]*THI.shape[2])
    
    m = np.mean(thi)
    s = np.std(thi)
    
    binns = bins(thi)
    
    ax[1].set_title('Thickness Distribution',fontsize = 16, fontweight = 'bold')
    N, bi, p  = ax[1].hist(thi,bins=binns,density=True,color='royalblue',edgecolor='k',rasterized=True)
    ax[1].set_xlabel(r'$\mathbf{\Delta Z (nm)}$', fontsize=18)
    ax[1].set_ylabel(r'$\mathbf{P(\Delta Z)}$', fontsize = 18)
    ax[1].tick_params(axis='both', labelsize = 16)
    ax[1].set_xticks(np.linspace(m-5*s,m+5*s,5))
    ax[1].set_xlim(m-5*s,m+5*s)
    ax[1].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    ax[0].set_xlabel(r'$\mathbf{X(\AA)}$', fontsize = 18)
    ax[0].set_ylabel(r'$\mathbf{Y(\AA)}$', fontsize = 18)
    a = (binns[-1]-binns[0])/(np.max(N))
    plt.tight_layout()
    ax[1].set_aspect(a)
    

#LRS Graphs
def graphLRS(POS, LRS):
    fig, ax = plt.subplots(2,3,figsize=(14,10))
    
    im1 = ax[0,0].imshow(np.mean(POS[0:40,0,:,:,2],axis = 0).T, origin = 'lower', cmap = 'terrain', interpolation = 'gaussian', extent = [gridX[0], gridX[-1], gridX[0], gridX[-1]])
    im2 = ax[0,1].imshow(np.mean(POS[0:40,0,:,:,2],axis = 0).T, origin = 'lower', cmap = 'terrain', interpolation = 'gaussian', extent = [gridX[0], gridX[-1], gridX[0], gridX[-1]])
    im3 = ax[0,2].imshow(np.mean(POS[0:40,0,:,:,2],axis = 0).T, origin = 'lower', cmap = 'terrain', interpolation = 'gaussian', extent = [gridX[0], gridX[-1], gridX[0], gridX[-1]])
    
    divider1 = mal(ax[0,0])
    cax1 = divider1.append_axes('right',
                                size = '5%',
                                pad = 0.05)
    cbar1 = fig.colorbar(im1,
                         cax=cax1,
                         format = '%.2f')
    cbar1.set_label(r'$\mathbf{\Delta Z^{\prime} (\AA)}$',
                    fontsize = 11,
                    fontweight = 'bold',
                    rotation = 0,
                    labelpad = -15)
    cbar1.set_ticks(np.linspace(np.min(np.mean(POS[0:40,0,:,:,2],axis = 0)), np.max(np.mean(POS[0:40,0,:,:,2],axis = 0)),4),2)
    cbar1.ax.tick_params(labelsize=14)
    
    divider1 = mal(ax[0,1])
    cax1 = divider1.append_axes('right',
                                size = '5%',
                                pad = 0.05)
    cbar1 = fig.colorbar(im1,
                         cax=cax1,
                         format = '%.2f')
    cbar1.set_label(r'$\mathbf{\Delta Z^{\prime} (\AA)}$',
                    fontsize = 11,
                    fontweight = 'bold',
                    rotation = 0,
                    labelpad = -15)
    cbar1.set_ticks(np.linspace(np.min(np.mean(POS[0:40,0,:,:,2],axis = 0)), np.max(np.mean(POS[0:40,0,:,:,2],axis = 0)),4),2)
    cbar1.ax.tick_params(labelsize=14)
    
    divider1 = mal(ax[0,2])
    cax1 = divider1.append_axes('right',
                                size = '5%',
                                pad = 0.05)
    cbar1 = fig.colorbar(im1,
                         cax=cax1,
                         format = '%.2f')
    cbar1.set_label(r'$\mathbf{\Delta Z^{\prime} (\AA)}$',
                    fontsize = 11,
                    fontweight = 'bold',
                    rotation = 0,
                    labelpad = -15)
    cbar1.set_ticks(np.linspace(np.min(np.mean(POS[0:40,0,:,:,2],axis = 0)), np.max(np.mean(POS[0:40,0,:,:,2],axis = 0)),4),2)
    cbar1.ax.tick_params(labelsize=14)
    
    
    ax[0,0].streamplot(XX, YY, np.mean(LRS[0:40,0,:,:,2,0],axis=0).T, np.mean(LRS[0:40,0,:,:,2,1],axis=0).T, color = 'k')
    ax[0,1].streamplot(XX, YY, np.mean(LRS[0:40,0,:,:,0,0],axis=0).T, np.mean(LRS[0:40,0,:,:,0,1],axis=0).T, color = 'k')
    ax[0,2].streamplot(XX, YY, np.mean(LRS[0:40,0,:,:,1,0],axis=0).T, np.mean(LRS[0:40,0,:,:,1,1],axis=0).T, color = 'k')
    
    ax[0,0].set_title('Upper Leaflet Curvature', fontsize = 16, fontweight = 'bold')
    ax[0,1].set_title('Upper Leaflet Isocurvature', fontsize = 16, fontweight = 'bold')
    ax[0,2].set_title('Upper Leaflet Anisocurvature', fontsize = 16, fontweight = 'bold')
    
    ax[0,0].set_xlabel(r'$\mathbf{X(\AA)}$', fontsize = 18)
    ax[0,0].set_ylabel(r'$\mathbf{Y(\AA)}$', fontsize = 18)
    ax[0,1].set_xlabel(r'$\mathbf{X(\AA)}$', fontsize = 18)
    ax[0,1].set_ylabel(r'$\mathbf{Y(\AA)}$', fontsize = 18)
    ax[0,2].set_xlabel(r'$\mathbf{X(\AA)}$', fontsize = 18)
    ax[0,2].set_ylabel(r'$\mathbf{Y(\AA)}$', fontsize = 18)
    
    ax[0,0].tick_params(labelsize=16)
    ax[0,1].tick_params(labelsize=16)
    ax[0,2].tick_params(labelsize=16)
    
    im1 = ax[1,0].imshow(np.mean(POS[0:40,1,:,:,2],axis = 0).T, origin = 'lower', cmap = 'terrain_r', interpolation = 'gaussian', extent = [gridX[0], gridX[-1], gridX[0], gridX[-1]])
    im2 = ax[1,1].imshow(np.mean(POS[0:40,1,:,:,2],axis = 0).T, origin = 'lower', cmap = 'terrain_r', interpolation = 'gaussian', extent = [gridX[0], gridX[-1], gridX[0], gridX[-1]])
    im3 = ax[1,2].imshow(np.mean(POS[0:40,1,:,:,2],axis = 0).T, origin = 'lower', cmap = 'terrain_r', interpolation = 'gaussian', extent = [gridX[0], gridX[-1], gridX[0], gridX[-1]])
    
    
    divider1 = mal(ax[1,0])
    cax1 = divider1.append_axes('right',
                                size = '5%',
                                pad = 0.05)
    cbar1 = fig.colorbar(im1,
                         cax=cax1,
                         format = '%.2f')
    cbar1.set_label(r'$\mathbf{\Delta Z^{\prime} (\AA)}$',
                    fontsize = 11,
                    fontweight = 'bold',
                    rotation = 0,
                    labelpad = -15)
    cbar1.set_ticks(np.linspace(np.min(np.mean(POS[0:40,1,:,:,2],axis = 0)), np.max(np.mean(POS[0:40,1,:,:,2],axis = 0)),4),2)
    cbar1.ax.tick_params(labelsize=14)
    
    divider1 = mal(ax[1,1])
    cax1 = divider1.append_axes('right',
                                size = '5%',
                                pad = 0.05)
    cbar1 = fig.colorbar(im1,
                         cax=cax1,
                         format = '%.2f')
    cbar1.set_label(r'$\mathbf{\Delta Z^{\prime} (\AA)}$',
                    fontsize = 11,
                    fontweight = 'bold',
                    rotation = 0,
                    labelpad = -15)
    cbar1.set_ticks(np.linspace(np.min(np.mean(POS[0:40,1,:,:,2],axis = 0)), np.max(np.mean(POS[0:40,1,:,:,2],axis = 0)),4),2)
    cbar1.ax.tick_params(labelsize=14)
    
    divider1 = mal(ax[1,2])
    cax1 = divider1.append_axes('right',
                                size = '5%',
                                pad = 0.05)
    cbar1 = fig.colorbar(im1,
                         cax=cax1,
                         format = '%.2f')
    cbar1.set_label(r'$\mathbf{\Delta Z^{\prime} (\AA)}$',
                    fontsize = 11,
                    fontweight = 'bold',
                    rotation = 0,
                    labelpad = -15)
    cbar1.set_ticks(np.linspace(np.min(np.mean(POS[0:40,1,:,:,2],axis = 0)), np.max(np.mean(POS[0:40,1,:,:,2],axis = 0)),4),2)
    cbar1.ax.tick_params(labelsize=14)
    
    ax[1,0].streamplot(XX, YY, np.mean(LRS[0:40,1,:,:,2,0],axis=0).T, np.mean(LRS[0:40,1,:,:,2,1],axis=0).T, color = 'k')
    ax[1,1].streamplot(XX, YY, np.mean(LRS[0:40,1,:,:,0,0],axis=0).T, np.mean(LRS[0:40,1,:,:,0,1],axis=0).T, color = 'k')
    ax[1,2].streamplot(XX, YY, np.mean(LRS[0:40,1,:,:,1,0],axis=0).T, np.mean(LRS[0:40,1,:,:,1,1],axis=0).T, color = 'k')
    
    ax[1,0].set_title('Lower Leaflet Curvature', fontsize = 16, fontweight = 'bold')
    ax[1,1].set_title('Lower Leaflet Isocurvature', fontsize = 16, fontweight = 'bold')
    ax[1,2].set_title('Lower Leaflet Anisocurvature', fontsize = 16, fontweight = 'bold')
    
    ax[1,0].set_xlabel(r'$\mathbf{X(\AA)}$', fontsize = 18)
    ax[1,0].set_ylabel(r'$\mathbf{Y(\AA)}$', fontsize = 18)
    ax[1,1].set_xlabel(r'$\mathbf{X(\AA)}$', fontsize = 18)
    ax[1,1].set_ylabel(r'$\mathbf{Y(\AA)}$', fontsize = 18)
    ax[1,2].set_xlabel(r'$\mathbf{X(\AA)}$', fontsize = 18)
    ax[1,2].set_ylabel(r'$\mathbf{Y(\AA)}$', fontsize = 18)
    
    ax[1,0].tick_params(labelsize=16)
    ax[1,1].tick_params(labelsize=16)
    ax[1,2].tick_params(labelsize=16)
    
    plt.tight_layout()

#Tails graphs
def graphTails(POS, SN1, SN2):
    fig, ax = plt.subplots(2,2,figsize=(14,10))
    
    im1 = ax[0,0].imshow(np.mean(POS[0:40,0,:,:,2],axis = 0).T, origin = 'lower', cmap = 'terrain', interpolation = 'gaussian', extent = [gridX[0], gridX[-1], gridX[0], gridX[-1]])
    im2 = ax[0,1].imshow(np.mean(POS[0:40,0,:,:,2],axis = 0).T, origin = 'lower', cmap = 'terrain', interpolation = 'gaussian', extent = [gridX[0], gridX[-1], gridX[0], gridX[-1]])
    
    divider1 = mal(ax[0,0])
    cax1 = divider1.append_axes('right',
                                size = '5%',
                                pad = 0.05)
    cbar1 = fig.colorbar(im1,
                         cax=cax1,
                         format = '%.2f')
    cbar1.set_label(r'$\mathbf{\Delta Z^{\prime} (\AA)}$',
                    fontsize = 11,
                    fontweight = 'bold',
                    rotation = 0,
                    labelpad = -15)
    cbar1.set_ticks(np.linspace(np.min(np.mean(POS[0:40,0,:,:,2],axis = 0)), np.max(np.mean(POS[0:40,0,:,:,2],axis = 0)),4),2)
    cbar1.ax.tick_params(labelsize=14)
    
    divider1 = mal(ax[0,1])
    cax1 = divider1.append_axes('right',
                                size = '5%',
                                pad = 0.05)
    cbar1 = fig.colorbar(im1,
                         cax=cax1,
                         format = '%.2f')
    cbar1.set_label(r'$\mathbf{\Delta Z^{\prime} (\AA)}$',
                    fontsize = 11,
                    fontweight = 'bold',
                    rotation = 0,
                    labelpad = -15)
    cbar1.set_ticks(np.linspace(np.min(np.mean(POS[0:40,0,:,:,2],axis = 0)), np.max(np.mean(POS[0:40,0,:,:,2],axis = 0)),4),2)
    cbar1.ax.tick_params(labelsize=14)
    
    
    ax[0,0].streamplot(XX, YY, np.mean(SN1[0:40,0,:,:,0],axis=0).T, np.mean(SN1[0:40,0,:,:,1],axis=0).T, color = 'k')
    ax[0,1].streamplot(XX, YY, np.mean(SN2[0:40,0,:,:,0],axis=0).T, np.mean(SN2[0:40,0,:,:,1],axis=0).T, color = 'k')
    
    ax[0,0].set_title('Upper Leaflet SN1 Tail', fontsize = 16, fontweight = 'bold')
    ax[0,1].set_title('Upper Leaflet SN2 Tail', fontsize = 16, fontweight = 'bold')
    
    ax[0,0].set_xlabel(r'$\mathbf{X(\AA)}$', fontsize = 18)
    ax[0,0].set_ylabel(r'$\mathbf{Y(\AA)}$', fontsize = 18)
    ax[0,1].set_xlabel(r'$\mathbf{X(\AA)}$', fontsize = 18)
    ax[0,1].set_ylabel(r'$\mathbf{Y(\AA)}$', fontsize = 18)
    
    ax[0,0].tick_params(labelsize=16)
    ax[0,1].tick_params(labelsize=16)
    
    im1 = ax[1,0].imshow(np.mean(POS[0:40,1,:,:,2],axis = 0).T, origin = 'lower', cmap = 'terrain_r', interpolation = 'gaussian', extent = [gridX[0], gridX[-1], gridX[0], gridX[-1]])
    im2 = ax[1,1].imshow(np.mean(POS[0:40,1,:,:,2],axis = 0).T, origin = 'lower', cmap = 'terrain_r', interpolation = 'gaussian', extent = [gridX[0], gridX[-1], gridX[0], gridX[-1]])
    
    divider1 = mal(ax[1,0])
    cax1 = divider1.append_axes('right',
                                size = '5%',
                                pad = 0.05)
    cbar1 = fig.colorbar(im1,
                         cax=cax1,
                         format = '%.2f')
    cbar1.set_label(r'$\mathbf{\Delta Z^{\prime} (\AA)}$',
                    fontsize = 11,
                    fontweight = 'bold',
                    rotation = 0,
                    labelpad = -15)
    cbar1.set_ticks(np.linspace(np.min(np.mean(POS[0:40,1,:,:,2],axis = 0)), np.max(np.mean(POS[0:40,1,:,:,2],axis = 0)),4),2)
    cbar1.ax.tick_params(labelsize=14)
    
    divider1 = mal(ax[1,1])
    cax1 = divider1.append_axes('right',
                                size = '5%',
                                pad = 0.05)
    cbar1 = fig.colorbar(im1,
                         cax=cax1,
                         format = '%.2f')
    cbar1.set_label(r'$\mathbf{\Delta Z^{\prime} (\AA)}$',
                    fontsize = 11,
                    fontweight = 'bold',
                    rotation = 0,
                    labelpad = -15)
    cbar1.set_ticks(np.linspace(np.min(np.mean(POS[0:40,1,:,:,2],axis = 0)), np.max(np.mean(POS[0:40,1,:,:,2],axis = 0)),4),2)
    cbar1.ax.tick_params(labelsize=14)
    
    ax[1,0].streamplot(XX, YY, np.mean(SN1[0:40,1,:,:,0],axis=0).T, np.mean(SN1[0:40,1,:,:,1],axis=0).T, color = 'k')
    ax[1,1].streamplot(XX, YY, np.mean(SN2[0:40,1,:,:,0],axis=0).T, np.mean(SN2[0:40,1,:,:,1],axis=0).T, color = 'k')
    
    ax[1,0].set_title('Lower Leaflet SN1 Tail', fontsize = 16, fontweight = 'bold')
    ax[1,1].set_title('Lower Leaflet SN2 Tail', fontsize = 16, fontweight = 'bold')
    
    ax[1,0].set_xlabel(r'$\mathbf{X(\AA)}$', fontsize = 18)
    ax[1,0].set_ylabel(r'$\mathbf{Y(\AA)}$', fontsize = 18)
    ax[1,1].set_xlabel(r'$\mathbf{X(\AA)}$', fontsize = 18)
    ax[1,1].set_ylabel(r'$\mathbf{Y(\AA)}$', fontsize = 18)
    
    ax[1,0].tick_params(labelsize=16)
    ax[1,1].tick_params(labelsize=16)
    
    plt.tight_layout()


def graphSN(SN1A, SN2A):
    
    fig, ax = plt.subplots(2,2,figsize=(12,12))
    
    up1 = SN1A[:,0].reshape(SN1A.shape[0]*SN1A.shape[2]*SN1A.shape[3])
    dn1 = np.pi - SN1A[:,1].reshape(SN1A.shape[0]*SN1A.shape[2]*SN1A.shape[3])
    up2 = SN2A[:,0].reshape(SN2A.shape[0]*SN1A.shape[2]*SN1A.shape[3])
    dn2 = np.pi - SN2A[:,1].reshape(SN2A.shape[0]*SN1A.shape[2]*SN1A.shape[3])
    
    ax[0,0].set_title('Upper Leaflet SN1', fontsize=22, fontweight='bold')
    ax[0,0].hist(up1,bins=bins(up1),density=True,color='royalblue',edgecolor='k',rasterized=True)
    ax[0,0].set_xlabel(r'$\mathbf{\Theta(rad)}$', fontsize=20)
    ax[0,0].set_ylabel(r'$\mathbf{P(\Theta)}$', fontsize = 20)
    ax[0,0].tick_params(axis='both', labelsize = 18)
    ax[0,0].set_xlim(2.9,np.pi)
    
    ax[0,1].set_title('Lower Leaflet SN1', fontsize=22, fontweight='bold')
    ax[0,1].hist(dn1,bins=bins(dn1),density=True,color='royalblue',edgecolor='k',rasterized=True)
    ax[0,1].set_xlabel(r'$\mathbf{\Theta(rad)}$', fontsize=20)
    ax[0,1].set_ylabel(r'$\mathbf{P(\Theta)}$', fontsize = 20)
    ax[0,1].tick_params(axis='both', labelsize = 18)
    ax[0,1].set_xlim(2.9,np.pi)
    
    ax[1,0].set_title('Upper Leaflet SN2', fontsize=22, fontweight='bold')
    ax[1,0].hist(up2,bins=bins(up2),density=True,color='royalblue',edgecolor='k',rasterized=True)
    ax[1,0].set_xlabel(r'$\mathbf{\Theta(rad)}$', fontsize=20)
    ax[1,0].set_ylabel(r'$\mathbf{P(\Theta)}$', fontsize = 20)
    ax[1,0].tick_params(axis='both', labelsize = 18)
    ax[1,0].set_xlim(2.9,np.pi)
    
    ax[1,1].set_title('Lower Leaflet SN2', fontsize=22, fontweight='bold')
    ax[1,1].hist(dn2,bins=bins(dn2),density=True,color='royalblue',edgecolor='k',rasterized=True)
    ax[1,1].set_xlabel(r'$\mathbf{\Theta(rad)}$', fontsize=20)
    ax[1,1].set_ylabel(r'$\mathbf{P(\Theta)}$', fontsize = 20)
    ax[1,1].tick_params(axis='both', labelsize = 18)
    ax[1,1].set_xlim(2.9,np.pi)
    plt.tight_layout()


#Execute the Search algorithm
SMem = hm.SearchMem()
SMem.DoSearch()

gridX = SMem.gridX #Grids define the limints of the cells, so a transformation is needed
X = []
for x in range(gridX.shape[0]-1):
    X.append((gridX[x]+gridX[x+1])/2)
X = np.array(X)
XX, YY = np.meshgrid(X,X) #X and Y grids are equal

#Execute the averaging algorithm
prom_pos = hm.Promed('pos') #Between brackets goes property to promediate (pos, sn1 or sn2)
prom_pos.GetPromed()
prom_sn1 = hm.Promed('sn1')
prom_sn1.GetPromed()
prom_sn2 = hm.Promed('sn2')
prom_sn2.GetPromed()

POS = prom_pos.Result
SN1 = prom_sn1.Result #Calls to the result matrix
SN2 = prom_sn2.Result

#Execute the normal determination algorithm
Norms = hm.Normals()
Norms.ComputeNormals()
NOR = Norms.Result #Calls to the result Matrix

#Execute the LRS determination algorithm
NewSR = hm.LRS()
NewSR.ComputeLRS()
LRS = NewSR.Result

#Compute the normal angle determination algorithm
NormAng = hm.NormalAngles()
NormAng.ComputeNormang()
ANGN = NormAng.Result


#Executes the Thickness determinationalgorithm
Thick = hm.Thickness()
Thick.ComputeThickness()
THI = Thick.Result

graphLRS(POS, LRS)

graphTails(POS, SN1, SN2)

graphTHI(THI)

#Execute the tail angle determination algorithm
SN1Ang = hm.TailAngles('sn1') #Between brackets which tail to analyze
SN1Ang.ComputeAngle()
SN2Ang = hm.TailAngles('sn2')
SN2Ang.ComputeAngle()

SN1A = SN1Ang.Result
SN2A = SN2Ang.Result


graphSN(SN1A, SN2A)


#Execute the translation to LRS algorithm
SN1_LRS = hm.TailLRS('sn1') #Between brackets which tail to analyze
SN1_LRS.ComputeTailLRS()
SN2_LRS = hm.TailLRS('sn2')
SN2_LRS.ComputeTailLRS()

#Execute SVD
SVD = hm.SVD('sn1up') #The type (buildtype) of svd you want to do
SVD.BuildData() #Build data matrix based on buildtype
SVD.ComputeSVD() #Compute the SVD
SVD.MakePlots() #Plot all the results in one graph


SVD = hm.SVD('sn1dn') #The type (buildtype) of svd you want to do
SVD.BuildData() #Build data matrix based on buildtype
SVD.ComputeSVD() #Compute the SVD
SVD.MakePlots() #Plot all the results in one graph

SVD = hm.SVD('sn2up') #The type (buildtype) of svd you want to do
SVD.BuildData() #Build data matrix based on buildtype
SVD.ComputeSVD() #Compute the SVD
SVD.MakePlots() #Plot all the results in one graph

SVD = hm.SVD('sn2dn') #The type (buildtype) of svd you want to do
SVD.BuildData() #Build data matrix based on buildtype
SVD.ComputeSVD() #Compute the SVD
SVD.MakePlots() #Plot all the results in one graph

SVD = hm.SVD('redux') #The type (buildtype) of svd you want to do
SVD.BuildData() #Build data matrix based on buildtype
SVD.ComputeSVD() #Compute the SVD
SVD.MakePlots() #Plot all the results in one graph






