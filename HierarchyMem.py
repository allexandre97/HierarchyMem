#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 13:07:29 2022

@author: xandre
"""

import MDAnalysis as mda
from MDAnalysis.analysis.leaflet import LeafletFinder as lf
import numpy as np
import math
from numba import cuda
import pickle
import os
import unicodedata as ud

class SearchMem: 
    def __init__(self, select = False):    
        try:
            with open('./Pickles/pos_up.pickle', 'rb') as f:
                self.POSu = pickle.load(f)
            f.close()
            with open('./Pickles/pos_dn.pickle', 'rb') as f:
                self.POSd = pickle.load(f)
            f.close()
            with open('./Pickles/sn1_up.pickle', 'rb') as f:
                self.SN1u = pickle.load(f)
            f.close()
            with open('./Pickles/sn1_dn.pickle', 'rb') as f:
                self.SN1d = pickle.load(f)
            f.close()
            with open('./Pickles/sn2_up.pickle', 'rb') as f:
                self.SN2u = pickle.load(f)
            f.close()
            with open('./Pickles/sn2_dn.pickle', 'rb') as f:
                self.SN2d = pickle.load(f)
            f.close()
            with open('./Pickles/gridX.pickle', 'rb') as f:
                self.gridX = pickle.load(f)
            f.close()
            with open('./Pickles/gridY.pickle', 'rb') as f:
                self.gridY = pickle.load(f)
            f.close()
            with open('./Pickles/rnm_up.pickle', 'rb') as f:
                self.RNMu = pickle.load(f)
            f.close()
            with open('./Pickles/rnm_dn.pickle', 'rb') as f:
                self.RNMd = pickle.load(f)
            f.close()
            with open('./Pickles/rid_up.pickle', 'rb') as f:
                self.RIDu = pickle.load(f)
            f.close()
            with open('./Pickles/rid_dn.pickle', 'rb') as f:
                self.RIDd = pickle.load(f)
            f.close()
            
            print('Initial pickle files found!\n')
            self.pick = True
        
        except:
            if select != False:
                print('Initial pickle files not found...\n')
                pdbfile = input('Input pdb or gro file:\n')
                xtcfile = input('Input xtc file:\n')
                self.nlipids = input('Input desired number of lipids per grid cell. Default is 10:\n')
                self.b       = input('Input desired extension to grid cells. Default is 0:\n')
                print('Loading files...\n')
                self.u = mda.Universe(pdbfile,xtcfile,in_memory=True)
                print('Files loaded!')
                self.names = {}
                res = self.u.residues
                n = 0
                for r in res:
                    if not r.resname in self.names:
                        self.names[r.resname] = n+1
                        n+=1
                        self.pick = False       
            else:
                print('Initial pickle files not found...\n')
                pdbfile = input('Input pdb or gro file:\n')
                xtcfile = input('Input xtc file:\n')
                self.nlipids = 11
                self.b       = 0
                print('Loading files...\n')
                self.u = mda.Universe(pdbfile,xtcfile,in_memory=True)
                print('Files loaded!')
                self.names = {}
                res = self.u.residues
                n = 0
                for r in res:
                    if not r.resname in self.names:
                        self.names[r.resname] = n+1
                        n+=1
                        self.pick = False        
        
    #Gets residue names and translates them to a float value as identifier
    def transname(self, resnames, names):
        trs = []
        for r in resnames:
            
            trs.append(names[r])
            
        return np.array(trs)
    
    #compute average grid dimensions based on number of lipids per grid cell
    def grid(self, nl, dims, d, Lipid_total):
        
        N = int(Lipid_total/2)
        nl = int(nl)
        
        width = math.sqrt(nl)*dims[d]/math.sqrt(N) #Step size of the grid based on how much lipids should fit in a grid cell
        N = int(dims[d]/width) #Number of grid points depending on the number of lipids per grid cell
        X = np.linspace(0, dims[d], N)
        return X
    
    #Count number of lipids per grid cell
    @cuda.jit
    def estimate(X, Y, P, M):
        
        width = X.shape[0]
        height = Y.shape[0]
        
        startX, startY = cuda.grid(2);
        gridX = cuda.gridDim.x * cuda.blockDim.x;
        gridY = cuda.gridDim.y * cuda.blockDim.y;
        
        for i in range(startX, width, gridX):
            for j in range(startY, height, gridY):
                for pos in P:
                    if X[i] <= pos[0] < X[i+1] and Y[j] <= pos[1] < Y[j+1]:
                        M[i, j] += 1
    
    #Estimate average number of lipids in grid Cell
    def estimation(self, dims, POS, b, nl, Lipid_total):
        
        gridX, gridY = self.grid(nl, dims, 0, Lipid_total), self.grid(nl, dims, 1, Lipid_total)
        b = int(b)
    
        blck = (8, 8)
        g = int(np.ceil(gridX.shape[0]/8))
        grd = (g, g)
    
        X0 = np.ascontiguousarray(gridX)
        Y0 = np.ascontiguousarray(gridY)
        P0 = np.ascontiguousarray(POS)
    
        M0 = np.zeros((X0.shape[0]-1, Y0.shape[0]-1))
    
        M_d = cuda.to_device(M0)
    
        self.estimate[grd, blck](X0, Y0, P0, M_d)
    
        M0 = M_d.copy_to_host()
        
        nl = (2*b+1)*np.mean(M0)
        
        return int(np.ceil(nl))
    
    #Calculate PO4-->Tail vector
    def tailvector(self, P, T, dims):    
    
        V = np.zeros(P.shape)
        
        for lipid in range(V.shape[0]):
            
            v = T[lipid] - P[lipid]
            
            for x in range(3):
                
                if abs(v[x]) >= dims[x]/3:
                    
                    if v[x] > 0:
                        
                        v[x] = v[x] - dims[x]
                        
                    else:
                        
                        v[x] = v[x] + dims[x]
                        
            V[lipid] += v
                
        return V
    
    def getdims(self, u):#This loops through the whole trajectory to ensure the grid is as big as the biggest frame
        dms = np.array([0,0,0])
        for n, frame in enumerate(u.trajectory):
            print('Analyzing frame dimensions... %i %%' % (100*((n+1)/len(u.trajectory))), end = '\r')
            dims = frame.dimensions[0:3]
            for d in range(3):
                if dims[d] > dms[d]:
                    dms[d] = dims[d]
        return dms
    
    #Function that associates different properties to positions in XY grid
    @cuda.jit
    def searchmat(X, Y, P, V1, V2, IDS, NMS, M, b):
        
        width = M.shape[0]
        height = M.shape[1]
        
        startX, startY = cuda.grid(2);
        gridX = cuda.gridDim.x * cuda.blockDim.x;
        gridY = cuda.gridDim.y * cuda.blockDim.y;
        
        for i in range(startX, width, gridX):
            
            if i < b:
                px1 = M.shape[0] - b + i
                px2 = i + b + 1
                outx = 1
            elif i >= M.shape[0] - b:
                px1 = i - b
                px2 = i + b - M.shape[0]
                outx = 1
            else:
                px1 = i - b
                px2 = i + b + 1
                outx = 0
                
            for j in range(startY, height, gridY):
                
                if j < b:
                    py1 = M.shape[1] - b + j
                    py2 = j + b + 1
                    outy = 1
                elif j >= M.shape[1] - b:
                    py1 = j - b
                    py2 = j + b - M.shape[1]
                    outy = 1
                else:
                    py1 = j - b
                    py2 = j + b + 1
                    outy = 0
                    
                n2 = 0
                
                for n, pos in enumerate(P):
                    
                    if n2 < M[i,j].shape[0] and n < P.shape[0]:
                    
                        if outx == 0 and outy == 0:
                
                            if X[px1]<= pos[0] < X[px2] and Y[py1] <= pos[1] < Y[py2]:
                    
                                for k in range(3):
                                
                                    M[i,j,n2,k] += pos[k]
                                    
                                for k in range(3,6):
                                    
                                    M[i,j,n2,k] += V1[n][k-3]
     
                                for k in range(6,9):
                                    
                                    M[i,j,n2,k] += V2[n][k-6]
                                    
                                M[i,j,n2,9] += IDS[n]
                                M[i,j,n2,10] += NMS[n]
                                
                                n2 += 1
                        
                        elif outx == 1 and outy == 0:
                        
                            if (pos[0] < X[px2] or X[px1] <= pos[0]) and (Y[py1] <= pos[1] < Y[py2]):
    
                                for k in range(3):
                                
                                    M[i,j,n2,k] += pos[k]
                                    
                                if X[i] > X[px1]:
                                    
                                    if pos[0] < X[px2]:
                                        
                                        M[i,j,n2,0] = X[i] + M[i,j,n2,0]
                                    
                                else:
                                    
                                    if pos[0] >= X[px1]: 
                                    
                                        M[i,j,n2,0] = X[i] - (X[-1] - M[i,j,n2,0])
                                    
                                for k in range(3,6):
                                    
                                    M[i,j,n2,k] += V1[n][k-3]
     
                                for k in range(6,9):
                                    
                                    M[i,j,n2,k] += V2[n][k-6]
                                    
                                M[i,j,n2,9] += IDS[n]
                                M[i,j,n2,10] += NMS[n]
                            
                                n2 += 1
                            
                        elif outx == 0 and outy == 1:
                        
                            if (X[px1] <= pos[0] < X[px2]) and (pos[1] < Y[py2] or Y[py1] <= pos[1]):
                                
                                for k in range(3):
                                
                                    M[i,j,n2,k] += pos[k]
                                    
                                if Y[j] > Y[py1]:
                                    
                                    if pos[1] < Y[py2]:
                                        
                                        M[i,j,n2,1] = Y[j] + M[i,j,n2,1]
                                
                                else:
                                    
                                    if pos[1] >= Y[py1]:
                                    
                                        M[i,j,n2,1] = Y[j] - (Y[-1] - M[i,j,n2,1])
                                    
                                for k in range(3,6):
                                    
                                    M[i,j,n2,k] += V1[n][k-3]
     
                                for k in range(6,9):
                                    
                                    M[i,j,n2,k] += V2[n][k-6]
                                    
                                M[i,j,n2,9] += IDS[n]
                                M[i,j,n2,10] += NMS[n]
                                    
                                n2 += 1
                            
                        elif outx == 1 and outy == 1:
                        
                            if (pos[0] < X[px2] or X[px1] <= pos[0]) and (pos[1] < Y[py2] or Y[py1] <= pos[1]):
                            
                                for k in range(3):
                                
                                    M[i,j,n2,k] += pos[k]
                                    
                                if X[i] > X[px1]:
                                    
                                    if pos[0] < X[px2]:
                                        
                                        M[i,j,n2,0] = X[i] + M[i,j,n2,0]
                                    
                                else:
                                    
                                    if pos[0] >= X[px1]: 
                                    
                                        M[i,j,n2,0] = X[i] - (X[-1] - M[i,j,n2,0])
                                        
                                if Y[j] > Y[py1]:
                                    
                                    if pos[1] < Y[py2]:
                                        
                                        M[i,j,n2,1] = Y[j] + M[i,j,n2,1]
                                
                                else:
                                    
                                    if pos[1] >= Y[py1]:
                                    
                                        M[i,j,n2,1] = Y[j] - (Y[-1] - M[i,j,n2,1]) 
                                    
                                for k in range(3,6):
                                    
                                    M[i,j,n2,k] += V1[n][k-3]
     
                                for k in range(6,9):
                                    
                                    M[i,j,n2,k] += V2[n][k-6]
                                    
                                M[i,j,n2,9] += IDS[n]
                                M[i,j,n2,10] += NMS[n]
                                    
                                n2 += 1      
    
    #Search function that wraps all the analyses and returns the search matrix
    def Search(self, Pup, T1up, T2up, IDS, rnams, b, gridX, gridY, nlips, dims):
    
        PPOS  = Pup.positions
        T1POS = T1up.positions
        T2POS = T2up.positions
        V1, V2 = self.tailvector(PPOS, T1POS, dims), self.tailvector(PPOS, T2POS, dims)
        MSEARCH = np.zeros((gridX.shape[0]-1, gridY.shape[0]-1, nlips, 14))
        blck = (4, 4)
        g = int(np.ceil(MSEARCH.shape[0]/4))
        grd = (g, g)
        M_d = cuda.to_device(MSEARCH)
        self.searchmat[grd, blck](gridX, gridY, PPOS, V1, V2, IDS, rnams, M_d, b)
        MSEARCH = M_d.copy_to_host()
        
        return MSEARCH
    
    def through_traj(self, u, Pup, Pdn, T1up, T1dn, T2up, T2dn, Lup_ids, Ldn_ids, RNamesu, RNamesd, b, gridX, gridY, nlips):
        MSU = []
        MSD = []
        for fr, frame in enumerate(u.trajectory): #Loop through trajectory and perform search    
            print('Analyzing positions and vectors... %i %%' % (int((fr+1)/len(u.trajectory)*100)), end = '\r')
            dims = frame.dimensions
            MSU.append(self.Search(Pup, T1up, T2up, Lup_ids, RNamesu, b, gridX, gridY, nlips, dims))
            MSD.append(self.Search(Pdn, T1dn, T2dn, Ldn_ids, RNamesd, b, gridX, gridY, nlips, dims))
        MSU = np.array(MSU)
        MSD = np.array(MSD)
        
        return MSU, MSD

    #Divides output array in chunks and writes them to .pickle file
    def to_files(self, MS,lf):
        
        lfs = ['_up', '_dn']
        
        POS = MS[:,:,:,:,0:3]
        V1  = MS[:,:,:,:,3:6]
        V2  = MS[:,:,:,:,6:9]
        RID = MS[:,:,:,:,9]
        RNM = MS[:,:,:,:,10]
        
        try:
            os.system('mkdir Pickles')
        except:
            pass
        
        with open('./Pickles/pos'+lfs[lf]+'.pickle', 'wb') as f:
            pickle.dump(POS,f)
        f.close()
        with open('./Pickles/sn1'+lfs[lf]+'.pickle', 'wb') as f:
            pickle.dump(V1,f)
        f.close()
        with open('./Pickles/sn2'+lfs[lf]+'.pickle', 'wb') as f:
            pickle.dump(V2,f)
        f.close()
        with open('./Pickles/rid'+lfs[lf]+'.pickle', 'wb') as f:
            pickle.dump(RID,f)
        f.close()
        with open('./Pickles/rnm'+lfs[lf]+'.pickle', 'wb') as f:
            pickle.dump(RNM,f)
        f.close()
        
    def DoSearch(self,):#, self.nlipids, self.b, u, pick, names):
        if self.pick == False:
            P =  self.u.select_atoms('name PO4', updating = True, periodic = True)#Select PO4 groups
            Lipid_total = len(P)
            Leafs = lf(self.u, P, pbc = True) #Get leaflets
            Lup_ids = Leafs.groups(0).ids #Get resids
            Ldn_ids = Leafs.groups(1).ids
            Lup_res = Leafs.groups(0).residues #Get residues corresponding to leaflets
            Ldn_res = Leafs.groups(1).residues
            Pup  = Leafs.groups(0) #PO4 atom groups
            Pdn  = Leafs.groups(1)
            T1up = mda.AtomGroup([atom for atom in Lup_res.atoms if atom.name == 'C4A']) #Tails atom groups
            T2up = mda.AtomGroup([atom for atom in Lup_res.atoms if atom.name == 'C4B']) 
            T1dn = mda.AtomGroup([atom for atom in Ldn_res.atoms if atom.name == 'C4A'])
            T2dn = mda.AtomGroup([atom for atom in Ldn_res.atoms if atom.name == 'C4B'])
            rnamesu = Pup.resnames #Get residue names and translate them
            rnamesd = Pdn.resnames #Get residue names and translate them
            rnamsu = self.transname(rnamesu, self.names)
            rnamsd = self.transname(rnamesd, self.names)
            dims0 = self.u.trajectory[0].dimensions #get dimensions of first frame
            PPOS0 = Pup.positions #Get positions of phosphates
            nlips = self.estimation(dims0, PPOS0, self.b, self.nlipids, Lipid_total) #Estimate number of lipids per grid cell
            dms = self.getdims(self.u)
            
            self.gridX, self.gridY = self.grid(nlips, dms, 0, Lipid_total), self.grid(nlips, dms, 1, Lipid_total) #We define the grids based on the previous analysis
            
            MUP, MSD = self.through_traj(self.u, Pup, Pdn, T1up, T1dn, T2up, T2dn, Lup_ids, Ldn_ids, rnamsu, rnamsd, int(self.b), self.gridX, self.gridY, nlips)
            
            self.POSu = MUP[:,:,:,:,0:3]
            self.SN1u  = MUP[:,:,:,:,3:6]
            self.SN2u  = MUP[:,:,:,:,6:9]
            self.RIDu = MUP[:,:,:,:,9]
            self.RNMu = MUP[:,:,:,:,10]
            self.POSd = MUP[:,:,:,:,0:3]
            self.SN1d  = MUP[:,:,:,:,3:6]
            self.SN2d  = MUP[:,:,:,:,6:9]
            self.RIDd = MUP[:,:,:,:,9]
            self.RNMd = MUP[:,:,:,:,10]
            
            self.to_files(MUP,0)
            self.to_files(MSD,1)
            with open('./Pickles/gridX.pickle', 'wb') as f:
                pickle.dump(self.gridX,f)
            f.close()
            with open('./Pickles/gridY.pickle', 'wb') as f:
                pickle.dump(self.gridY,f)
            f.close()
        else:
            pass     

class Promed:
    def __init__(self, Prop):
        
        try:
            with open('./Pickles/prom_'+Prop+'.pickle', 'rb') as f:
                self.Result = pickle.load(f)
            f.close()
            print('Already found averaged arrays for %s!\n' % Prop)
            self.found = True
        except:
            print('No average arrays found for this property...\n')
            self.Prop = Prop
            with open('./Pickles/'+Prop+'_up.pickle', 'rb') as f:
                self.PU = pickle.load(f)
            f.close()
            with open('./Pickles/'+Prop+'_dn.pickle', 'rb') as f:
                self.PD = pickle.load(f)
            f.close()
            with open('./Pickles/rnm_up.pickle', 'rb') as f:
                self.RU = pickle.load(f)
            f.close()
            with open('./Pickles/rnm_dn.pickle', 'rb') as f:
                self.RD = pickle.load(f)
            f.close()
            self.found = False
    @cuda.jit
    def prom(M,name,rnm,mout,n):
        width = M.shape[0]
        height = M.shape[1]
        startX, startY = cuda.grid(2);
        gridX = cuda.gridDim.x * cuda.blockDim.x;
        gridY = cuda.gridDim.y * cuda.blockDim.y;
        for i in range(startX, width, gridX):
            for j in range(startY, height, gridY):
                for k in range(M.shape[2]):
                    if M[i,j,k,2] != 0.:# and int(rnm[i,j,k]) == int(name):
                        for l in range(3):
                            mout[i,j,l] += M[i,j,k,l]
                        n += 1
                for l in range(3):    
                    mout[i,j,l] /= n
                n-=n
    def GetPromed(self,):
        if self.found == False:
            blck = (4, 4)
            g = int(np.ceil(self.PU.shape[1]/4))
            grd = (g, g)
            POS = []
            rname = 1
            for f in range(self.PU.shape[0]):
                print('Averaging %s %i %%' % ( self.Prop, 100*(f+1)/self.PU.shape[0]), end = '\r')
                pup = np.zeros((self.PU.shape[1],self.PU.shape[2],3))
                p_d = cuda.to_device(pup)
                self.prom[grd,blck](self.PU[f],rname,self.RU[f],p_d,0)
                pup = p_d.copy_to_host()
                pdn = np.zeros((self.PD.shape[1],self.PD.shape[2],3)) 
                p_d = cuda.to_device(pdn)
                self.prom[grd,blck](self.PD[f],rname,self.RD[f],p_d,0)
                pdn = p_d.copy_to_host()
                POS.append(np.array([pup,pdn]))
            self.Result = np.array(POS)
            with open('./Pickles/prom_'+self.Prop+'.pickle', 'wb') as f:
                pickle.dump(self.Result, f)
            f.close()
        else:
            pass
        
class Normals:
    
    def __init__(self,):
        try:
            with open('./Pickles/normals.pickle', 'rb') as f:
                self.Result = pickle.load(f)
            f.close()
            print('Normals array already found!\n')
            self.found = True
        except:
            print('No Normals array found...')
            with open('./Pickles/prom_pos.pickle', 'rb') as f:
                self.POS = pickle.load(f)
            f.close()
            with open('./Pickles/gridX.pickle', 'rb') as f:
                self.gridX = pickle.load(f)
            f.close()
            with open('./Pickles/gridY.pickle', 'rb') as f:
                self.gridY = pickle.load(f)
            f.close()
            self.found = False
    
    def plane_from_points(self,points):
        # Create this matrix correctly without transposing it later?
        A = np.array([
            points[0,:],
            points[1,:],
            np.ones(points.shape[1])
        ]).T
        b = np.array([points[2, :]]).T
        # fit = (A.T * A).I * A.T * b
        fit = np.dot(np.dot(np.linalg.inv(np.dot(A.T, A)), A.T), b)
        # errors = b - np.dot(A, fit)
        # residual = np.linalg.norm(errors)
        return fit

    def periodic(self,pos,b,gridX,gridY):        
        axis = pos[:,-b:].copy()
        axis[:,:,1]-=gridY[-1]
        pbc = np.append(axis,pos,axis=1)
        axis = pos[:,:b].copy()
        axis[:,:,1]+=gridY[-1]
        pbc = np.append(pbc,axis,axis = 1)
        axis = pbc[-b:,:].copy()
        axis[:,:,0]-=gridX[-1]
        pbc = np.append(axis,pbc,axis = 0)
        axis = pbc[0:b,:].copy()
        axis[:,:,0]+=gridX[-1]
        pbc = np.append(pbc,axis,axis = 0)
        return pbc
    
    def normal(self,pos,b):
        norms = np.zeros((pos.shape[0]-2*b,pos.shape[1]-2*b,3))
        for i in range(norms.shape[0]):
            for j in range(norms.shape[1]):
                Pos = []
                for ii in range(i,i+2*b+1):
                    for jj in range(j,j+2*b+1):
                        Pos.append(pos[ii,jj])
                Pos=np.array(Pos)
                C = np.mean(Pos,axis=0,keepdims=True)
                fit = self.plane_from_points((Pos-C).T)        
                norm = np.array(np.cross([1,0,fit[0,0]],[0,1,fit[1,0]]))
                norm /= np.linalg.norm(norm)
                norms[i,j] += norm
        return norms
    
    def ComputeNormals(self,):
        if self.found == False:
            NORMS = []
            for frame in range(self.POS.shape[0]):
                print('Calculating Normals %i %%' % (100*(frame+1)/self.POS.shape[0]), end = '\r')
                posdn = self.POS[frame,1]#prom(posDN[frame]) 
                posup = self.POS[frame,0]#prom(posUP[frame])
                pbcup = self.periodic(posup.copy(),1,self.gridX,self.gridY)
                pbcdn = self.periodic(posdn.copy(),1,self.gridX,self.gridY)
                nup = self.normal(pbcup,1)
                ndn = self.normal(pbcdn,1)
                NORMS.append(np.array([nup,ndn]))
            self.Result = np.array(NORMS)
            with open('./Pickles/normals.pickle', 'wb') as f:
                pickle.dump(self.Result,f)
            f.close()
        else:
            pass
        
class NormalAngles:
    def __init__(self,):
        try:
            with open('./Pickles/normang.pickle', 'rb') as f:
                self.Result = pickle.load(f)
            f.close()
            print('Normal Angles array already found!\n')
            self.found = True
        except:
            print('Normal Angles array not found...\n')
            with open('./Pickles/normals.pickle', 'rb') as f:
                self.NOR = pickle.load(f)
            f.close()
            self.found = False
        
    def angle(self, v1, v2):
        inn = np.inner(v1,v2)
        norm = np.linalg.norm(v1)*np.linalg.norm(v2)
        cos = inn/norm
        ang = np.arccos(np.clip(cos,-1.,1.))
        return ang

    def getang(self, sr, vec):
        angs = np.zeros((sr.shape[0],sr.shape[1]))
        for i in range(sr.shape[0]):
            for j in range(sr.shape[1]):
                v = sr[i,j]
                angs[i,j] += self.angle(v,vec)
        return angs
    
    def ComputeNormang(self,):
        if self.found == False:
            ANGS = []
            for f in range(self.NOR.shape[0]):
                print('Calculating Normal Angles %i %%' % (100*(f+1)/self.NOR.shape[0]), end = '\r')
                nup = self.NOR[f,0]
                ndn = self.NOR[f,1]    
                angup = self.getang(nup,np.array([0,0,1]))
                angdn = self.getang(ndn,np.array([0,0,-1]))
                ANGS.append(np.array([angup,angdn]))
            self.Result = np.array(ANGS)
            with open('./Pickles/normang.pickle','wb') as f:
                pickle.dump(self.Result,f)
            f.close()
        else:
            pass

class Thickness:
    def __init__(self,):
        try:
            with open('./Pickles/thick.pickle', 'rb') as f:
                self.Result = pickle.load(f)
            f.close()
            print('Thickness array already found!\n')
            self.found = True
        except:
            print('Thickness array not found...\n')
            with open('./Pickles/prom_pos.pickle', 'rb') as f:
                self.POS = pickle.load(f)
            f.close()
            with open('./Pickles/normang.pickle', 'rb') as f:
                self.ANG = pickle.load(f)
            f.close()
            self.found = False
    def getproj(self, pos,angl):
        proj = np.zeros((pos.shape[0],pos.shape[1]))
        for i in range(pos.shape[0]):
            for j in range(pos.shape[1]):        
                angle = angl[i,j]
                z = pos[i,j,2]        
                pr = z/np.cos(angle)        
                proj[i,j] += pr
        return proj
    def ComputeThickness(self,):
        if self.found == False:
            THICK = []
            for f in range(self.POS.shape[0]):
                print('Calculated Thickness %i %%' % (100*(f+1)/self.POS.shape[0]), end = '\r')
                pup = self.POS[f,0]#prom(posup[f])
                pdn = self.POS[f,1]#prom(posdn[f])
                angup = self.ANG[f,0]
                angdn = self.ANG[f,1]
                projup = self.getproj(pup,angup)
                projdn = self.getproj(pdn,angdn)*-1
                thick = projup-projdn
                THICK.append(thick)
            self.Result = np.array(THICK)
            with open('./Pickles/thick.pickle','wb') as f:
                pickle.dump(self.Result,f)
            f.close()
        else:
            pass
    
class LRS:
    
    def __init__(self,):
        try:
            with open('./Pickles/lrs.pickle', 'rb') as f:
                self.Result = pickle.load(f)
            f.close()
            print('LRS array already found!\n')
            self.found = True
        except:
            print('No LRS array found...')
            with open('./Pickles/normals.pickle', 'rb') as f:
                self.NOR = pickle.load(f)
            f.close()
            with open('./Pickles/gridX.pickle', 'rb') as f:
                self.gridX = pickle.load(f)
            f.close()
            with open('./Pickles/gridY.pickle', 'rb') as f:
                self.gridY = pickle.load(f)
            f.close()
            self.found = False
    
    def Jacobian(self, f, dx2):
        
        Jacob = np.zeros((f.shape[0], f.shape[1], 3, 3))
        
        for c in range(3):
        
            Jacob[0,:,c,0] += (f[1,:,c]-f[-1,:,c])/dx2
            Jacob[0,1:-1,c,1] += (f[0,2:,c]-f[0,:-2,c])/dx2
            
            Jacob[-1,:,c,0] += (f[0,:,c]-f[-2,:,c])/dx2
            Jacob[-1,1:-1,c,1] += (f[-1,2:,c]-f[-1,:-2,c])/dx2
            
            Jacob[:,0,c,1] += (f[:,1,c]-f[:,-1,c])/dx2
            Jacob[1:-1,0,c,0] += (f[2:,0,c]-f[:-2,0,c])/dx2
            
            Jacob[:,-1,c,1] += (f[:,0,c]-f[:,-2,c])/dx2
            Jacob[1:-1,-1,c,0] += (f[2:,-1,c]-f[:-2,-1,c])/dx2
            
            Jacob[1:-1,1:-1,c,0] += (f[2:,1:-1,c]-f[:-2,1:-1,c])/dx2
            Jacob[1:-1,1:-1,c,1] += (f[1:-1,2:,c]-f[1:-1,:-2,c])/dx2
            
        return Jacob

    def SR(self, norms, Grad):
        
        SR = np.zeros((norms.shape[0],norms.shape[1],3,3))
        
        for i in range(norms.shape[0]):
            for j in range(norms.shape[1]):
                
                SR[i,j,0] += np.cross(norms[i,j],Grad[i,j])
                SR[i,j,0] /= np.linalg.norm(SR[i,j,0])
                SR[i,j,1] += np.cross(norms[i,j],SR[i,j,0])
                SR[i,j,1] /= np.linalg.norm(SR[i,j,1])
                SR[i,j,2] += norms[i,j]
                SR[i,j,2] /= np.linalg.norm(SR[i,j,2])
                
        return SR
    
    def ComputeLRS(self,):
        
        if self.found == False:
            dx = self.gridX[1]-self.gridX[-1]
            dx2 = 2*dx
            SRS = []
            for frame in range(self.NOR.shape[0]):
                print('Calculating LRS %i %%' % (100*(frame+1)/self.NOR.shape[0]), end = '\r')
                nup = self.NOR[frame,0]
                ndn = self.NOR[frame,1]
                Jup = self.Jacobian(nup,dx2)
                Jdn = self.Jacobian(ndn,dx2)
                Gradup = Jup[:,:,2]
                Graddn = Jdn[:,:,2]
                srup = self.SR(nup, Gradup)
                srdn = self.SR(ndn, Graddn)
                srdn *= -1
                SRS.append(np.array([srup,srdn]))
            self.Result = np.array(SRS)
            with open('./Pickles/lrs.pickle', 'wb') as f:
                pickle.dump(self.Result,f)
            f.close()
        else:
            pass

class TailAngles:
    def __init__(self,Tail):
        self.Tail = Tail
        try:
            with open('./Pickles/ang_%s.pickle' % Tail, 'rb') as f:
                self.Result = pickle.load(f)
            f.close()
            print('Angles array for tail %s already found!\n' % Tail)
            self.found = True
        except:
            print('Angles array for tail %s not found...\n' % Tail)
            with open('./Pickles/prom_%s.pickle' % Tail, 'rb') as f:
                self.TAIL = pickle.load(f)
            f.close()
            with open('./Pickles/normals.pickle', 'rb') as f:
                self.NOR = pickle.load(f)
            f.close()
            self.found = False
            
    def angle(self, v1, v2):
        inn = np.inner(v1,v2)
        norm = np.linalg.norm(v1)*np.linalg.norm(v2)
        cos = inn/norm
        ang = np.arccos(np.clip(cos,-1.,1.))
        return ang

    def getang(self, V, norm):
        mang = np.zeros((V.shape[0], V.shape[1]))
        for i in range(V.shape[0]):
            for j in range(V.shape[1]):
                if V[i,j,0] != 0.:
                    ang1 = V[i,j]
                    ang2 = norm[i,j]
                    a = self.angle(ang1,ang2)
                    if np.isnan(a) == False:
                        mang[i,j] += a
        return mang
    
    def ComputeAngle(self,):
        if self.found == False:
            ANGS = []
            for f in range(self.TAIL.shape[0]):
                print('Calculating %s Tail Angles %i %%' % (self.Tail, 100*(f+1)/self.TAIL.shape[0]), end='\r')
                mv1up = self.TAIL[f,0]
                mv1dn = self.TAIL[f,1]
                mnup = self.NOR[f,0]
                mndn = self.NOR[f,1]
                angup = self.getang(mv1up,mnup)
                angdn = self.getang(mv1dn,mndn)
                ANGS.append(np.array([angup,angdn]))
            self.Result = np.array(ANGS)
            with open('./Pickles/ang_%s.pickle' % self.Tail, 'wb') as f:
                pickle.dump(self.Result, f)
            f.close()
        else:
            pass
        
class TailLRS:
    def __init__(self,Tail):
        self.Tail = Tail
        try:
            with open('./Pickles/%s_lrs.pickle' % Tail, 'rb') as f:
                self.Result = pickle.load(f)
            f.close()
            print('Array for %s Tail in LRS already found!\n' % Tail)
            self.found = True
        except:
            print('No array for %s Tail in LRS found...\n' % Tail)
            with open('./Pickles/prom_%s.pickle' % Tail, 'rb') as f:
                self.TAIL = pickle.load(f)
            f.close()
            with open('./Pickles/lrs.pickle', 'rb') as f:
                self.LRS = pickle.load(f)
            f.close()
            self.found = False
    
    def newvec(self,vec,sr):
        nvec = np.zeros((vec.shape[0],vec.shape[1],3))
        for i in range(vec.shape[0]):
            for j in range(vec.shape[1]):
                basis = sr[i,j]
                v = vec[i,j]
                nvec[i,j] += np.linalg.inv(basis).dot(v)
        return nvec
    
    def ComputeTailLRS(self,):
        if self.found == False:
            VECS = []
            for f in range(self.TAIL.shape[0]):
                print('Calculating %s Tail in LRS %i %%' % (self.Tail, 100*(f+1)/self.TAIL.shape[0]), end = '\r')
                vup = self.TAIL[f,0]
                vdn = self.TAIL[f,1]
                srup = self.LRS[f,0]
                srdn = self.LRS[f,1]
                newup = self.newvec(vup,srup)
                newdn = self.newvec(vdn,srdn)
                VECS.append(np.array([newup,newdn]))
            self.Result = np.array(VECS)
            with open('./Pickles/%s_lrs.pickle' %self.Tail,'wb') as f:
                pickle.dump(self.Result,f)
            f.close()
        else:
            pass

class SVD:
    def __init__(self, Build):
        
        files = [_ for _ in os.listdir('./Pickles/') if _.startswith('svd')]
        
        Builds = []
        
        for file in files:
            
            l = len(file)
            s = ''
            for p in range(l-8,0,-1):
                if file[p]!='_':
                    s+=file[p]
                else:
                    break
            s = s[::-1]
            if not s in Builds:
                Builds.append(s)
        
        if len(Builds) != 0 and Build == None: 
            
            print('\n These are the available already analyzed builds \n')        
            for b in range(len(Builds)):
                print(' %i --> %s' % (b, Builds[b]))
            
            self.Build = input(' Select one of the previous builds by number\n Input None if you do not want any of the builds shown ')
            
        else:
            self.Build = Build
        
        try:
            with open('./Pickles/svd_datamat_%s.pickle' % self.Build, 'rb') as f:
                self.DataMat = pickle.load(f)
            f.close()
            with open('./Pickles/svd_leftmat_%s.pickle' % self.Build, 'rb') as f:
                self.LeftMat = pickle.load(f)
            f.close()
            with open('./Pickles/svd_singval_%s.pickle' % self.Build, 'rb') as f:
                self.SingVal = pickle.load(f)
            f.close()
            with open('./Pickles/svd_rightmat_%s.pickle' % self.Build, 'rb') as f:
                self.RightMat = pickle.load(f)
            f.close()
            with open('./Pickles/svd_relmat_%s.pickle' % self.Build, 'rb') as f:
                self.RelMat = pickle.load(f)
            f.close()
            with open('./Pickles/svd_coupmat_%s.pickle' % self.Build, 'rb') as f:
                self.CoupMat = pickle.load(f)
            f.close()
            print('Results for %s SVD analysis already found!\n' % self.Build)
            self.found = True
        except:
            print('Could not open Build %s for SVD analysis...\n' % self.Build)
            with open('./Pickles/prom_pos.pickle', 'rb') as f:
                POS = pickle.load(f) #Position
            f.close()
            with open('./Pickles/lrs.pickle', 'rb') as f:
                LRS = pickle.load(f) #Internal Reference Systems
            f.close()
            with open('./Pickles/normang.pickle', 'rb') as f:
                ANGn = pickle.load(f) #Curvature Scalar Field
            f.close()
            with open('./Pickles/ang_sn1.pickle', 'rb') as f:
                ANGt = pickle.load(f) #Tail Tilt Scalar Field
            f.close()
            with open('./Pickles/ang_sn2.pickle', 'rb') as f:
                ANGt2 = pickle.load(f) #Tail Tilt Scalar Field
            f.close()
            with open('./Pickles/thick.pickle', 'rb') as f:
                THI = pickle.load(f) #Thickness scalar Field
            f.close()
            with open('./Pickles/sn1_lrs.pickle', 'rb') as f:
                SN1 = pickle.load(f) #SN1 tail orientation
            f.close()
            with open('./Pickles/sn2_lrs.pickle', 'rb') as f:
                SN2 = pickle.load(f) #SN2 tail orientation
            f.close()
            POSz_up = POS[:,0,:,:,2]
            POSz_dn = POS[:,1,:,:,2]
            ISOx_up = LRS[:,0,:,:,0,0]
            ISOx_dn = LRS[:,1,:,:,0,0]
            ISOy_up = LRS[:,0,:,:,0,1]
            ISOy_dn = LRS[:,1,:,:,0,1]
            ISOz_up = LRS[:,0,:,:,0,2]
            ISOz_dn = LRS[:,1,:,:,0,2]
            ANIx_up = LRS[:,0,:,:,1,0]
            ANIx_dn = LRS[:,1,:,:,1,0]
            ANIy_up = LRS[:,0,:,:,1,1]
            ANIy_dn = LRS[:,1,:,:,1,1]
            ANIz_up = LRS[:,0,:,:,1,2]
            ANIz_dn = LRS[:,1,:,:,1,2]
            NORx_up = LRS[:,0,:,:,2,0]
            NORx_dn = LRS[:,1,:,:,2,0]
            NORy_up = LRS[:,0,:,:,2,1]
            NORy_dn = LRS[:,1,:,:,2,1]
            NORz_up = LRS[:,0,:,:,2,2]
            NORz_dn = LRS[:,1,:,:,2,2]
            ANGn_up = ANGn[:,0,:,:]
            ANGn_dn = ANGn[:,1,:,:]
            ANGt_up = ANGt[:,0,:,:]
            ANGt_dn = ANGt[:,1,:,:]
            ANGt2_up = ANGt2[:,0,:,:]
            ANGt2_dn = ANGt2[:,1,:,:]
            SN1x_up = SN1[:,0,:,:,0]
            SN1x_dn = SN1[:,1,:,:,0]
            SN1y_up = SN1[:,0,:,:,1]
            SN1y_dn = SN1[:,1,:,:,1]
            SN1z_up = SN1[:,0,:,:,2]
            SN1z_dn = SN1[:,1,:,:,2]
            SN2x_up = SN2[:,0,:,:,0]
            SN2x_dn = SN2[:,1,:,:,0]
            SN2y_up = SN2[:,0,:,:,1]
            SN2y_dn = SN2[:,1,:,:,1]
            SN2z_up = SN2[:,0,:,:,2]
            SN2z_dn = SN2[:,1,:,:,2]
            self.ALLVARS = np.array([POSz_up, POSz_dn,
                       ISOx_up, ISOx_dn, 
                       ISOy_up, ISOy_dn, 
                       ISOz_up, ISOz_dn, 
                       ANIx_up, ANIx_dn,
                       ANIy_up, ANIy_dn,
                       ANIz_up, ANIz_dn,
                       NORx_up, NORx_dn,
                       NORy_up, NORy_dn,
                       NORz_up, NORz_dn,
                       ANGn_up, ANGn_dn,
                       ANGt_up, ANGt_dn,
                       ANGt2_up, ANGt2_dn,
                       THI,
                       SN1x_up, SN1x_dn,
                       SN1y_up, SN1y_dn,
                       SN1z_up, SN1z_dn,
                       SN2x_up, SN2x_dn,
                       SN2y_up, SN2y_dn,
                       SN2z_up, SN2z_dn])
            self.ALLNAMES = np.array(['Topograhy Upper Leaflet', 'Topography Lower Leaflet',
                       'Isocurvature X Upper Leaflet', 'Isocurvature X Lower Leaflet', 
                       'Isocurvature Y Upper Leaflet', 'Isocurvature Y Lower Leaflet', 
                       'Isocurvature Z Upper Leaflet', 'Isocurvature Z Lower Leaflet', 
                       'Anisocurvature X Upper Leaflet', 'Anisocurvature X Lower Leaflet', 
                       'Anisocurvature Y Upper Leaflet', 'Anisocurvature Y Lower Leaflet', 
                       'Anisocurvature Z Upper Leaflet', 'Anisocurvature Z Lower Leaflet',
                       'Normals X Upper Leaflet', 'Normals X Lower Leaflet', 
                       'Normals Y Upper Leaflet', 'Normals Y Lower Leaflet', 
                       'Normals Z Upper Leaflet', 'Normals Z Lower Leaflet',
                       'Normal Angles Upper Leaflet', 'Normal Angles Lower Leaflet',
                       'SN1 Angles Upper Leaflet', 'SN1 Angles Lower Leaflet',
                       'SN2 Angles Upper Leaflet', 'SN2 Angles Lower Leaflet',
                       'Thickness',
                       "SN1 X' Upper Leaflet", "SN1 X' Lower Leaflet",
                       "SN1 Y' Upper Leaflet", "SN1 Y' Lower Leaflet",
                       "SN1 Z' Upper Leaflet", "SN1 Z' Lower Leaflet",
                       "SN2 X' Upper Leaflet", "SN2 X' Lower Leaflet",
                       "SN2 Y' Upper Leaflet", "SN2 Y' Lower Leaflet",
                       "SN2 Z' Upper Leaflet", "SN2 Z' Lower Leaflet"])
            self.NAMESDICT = {'Topograhy Upper Leaflet':r'$\mathbf{\Delta Z}$', 'Topography Lower Leaflet':r'$\mathbf{\Delta Z}$',
                       'Isocurvature X Upper Leaflet':r'$\mathbf{X^{\prime}_{X}}$', 'Isocurvature X Lower Leaflet':r'$\mathbf{X^{\prime}_{X}}$', 
                       'Isocurvature Y Upper Leaflet':r'$\mathbf{X^{\prime}_{Y}}$', 'Isocurvature Y Lower Leaflet':r'$\mathbf{X^{\prime}_{Y}}$', 
                       'Isocurvature Z Upper Leaflet':r'$\mathbf{X^{\prime}_{Z}}$', 'Isocurvature Z Lower Leaflet':r'$\mathbf{X^{\prime}_{Z}}$', 
                       'Anisocurvature X Upper Leaflet':r'$\mathbf{Y^{\prime}_{X}}$', 'Anisocurvature X Lower Leaflet':r'$\mathbf{Y^{\prime}_{X}}$', 
                       'Anisocurvature Y Upper Leaflet':r'$\mathbf{Y^{\prime}_{Y}}$', 'Anisocurvature Y Lower Leaflet':r'$\mathbf{Y^{\prime}_{Y}}$', 
                       'Anisocurvature Z Upper Leaflet':r'$\mathbf{Y^{\prime}_{Z}}$', 'Anisocurvature Z Lower Leaflet':r'$\mathbf{Y^{\prime}_{Z}}$',
                       'Normals X Upper Leaflet':r'$\mathbf{Z^{\prime}_{X}}$', 'Normals X Lower Leaflet':r'$\mathbf{Z^{\prime}_{X}}$', 
                       'Normals Y Upper Leaflet':r'$\mathbf{Z^{\prime}_{Y}}$', 'Normals Y Lower Leaflet':r'$\mathbf{Z^{\prime}_{Y}}$', 
                       'Normals Z Upper Leaflet':r'$\mathbf{Z^{\prime}_{Z}}$', 'Normals Z Lower Leaflet':r'$\mathbf{Z^{\prime}_{Z}}$',
                       'Normal Angles Upper Leaflet':r'$\mathbf{\angle Z^{\prime}}$', 'Normal Angles Lower Leaflet':r'$\mathbf{\angle Z^{\prime}}$',
                       'SN1 Angles Upper Leaflet':r'$\mathbf{\angle SN1}$', 'SN1 Angles Lower Leaflet':r'$\mathbf{\angle SN1}$',
                       'SN2 Angles Upper Leaflet':r'$\mathbf{\angle SN2}$', 'SN2 Angles Lower Leaflet':r'$\mathbf{\angle SN2}$',
                       'Thickness':r'$\mathbf{THI}$',
                       "SN1 X' Upper Leaflet":r'$\mathbf{SN1_{X^{\prime}}}$', "SN1 X' Lower Leaflet":r'$\mathbf{SN1_{X^{\prime}}}$',
                       "SN1 Y' Upper Leaflet":r'$\mathbf{SN1_{Y^{\prime}}}$', "SN1 Y' Lower Leaflet":r'$\mathbf{SN1_{Y^{\prime}}}$',
                       "SN1 Z' Upper Leaflet":r'$\mathbf{SN1_{Z^{\prime}}}$', "SN1 Z' Lower Leaflet":r'$\mathbf{SN1_{Z^{\prime}}}$',
                       "SN2 X' Upper Leaflet":r'$\mathbf{SN2_{X^{\prime}}}$', "SN2 X' Lower Leaflet":r'$\mathbf{SN2_{X^{\prime}}}$',
                       "SN2 Y' Upper Leaflet":r'$\mathbf{SN2_{Y^{\prime}}}$', "SN2 Y' Lower Leaflet":r'$\mathbf{SN2_{Y^{\prime}}}$',
                       "SN2 Z' Upper Leaflet":r'$\mathbf{SN2_{Z^{\prime}}}$', "SN2 Z' Lower Leaflet":r'$\mathbf{SN2_{Z^{\prime}}}$'}
            self.found = False

    def build(self,Set,Names):
        DATA = Set.reshape(Set.shape[0],Set.shape[3],Set.shape[1]*Set.shape[2])
        DATA = DATA.reshape(DATA.shape[0],DATA.shape[1]*DATA.shape[2])
        DATA = DATA.T
        #Normalize by columns
        for col in range(DATA.shape[1]):
            m = np.mean(DATA[:,col])
            s = np.std(DATA[:,col])
            DATA[:,col] = (DATA[:,col]-m)/s
        return DATA

    def select(self,):
        print('\nThese are all the available Variables:\n')
        sel = []
        for name in self.ALLNAMES:
            s = input(' %s\n Press 1 to ADD VARIABLE, 0 to NOT ADD VARIABLE. Missing an input means NOT add.\n' % name)
            if s == '1':
                s = True
            else:
                s = False
            sel.append(s)
        Set = self.ALLVARS[sel]
        Names = self.ALLNAMES[sel]
        self.plotnames = []
        for n in self.Names:
            self.plotnames.append(self.NAMESDICT[n])
        print('There are the selected variables:\n')
        for name in Names:
            print(name)
        cont = input(' Conintue with selected variables?\n 1-->Yes, 0--> No, select again.\n')
        return Set, Names, int(cont)
        
    def BuildData(self,):
        if self.build == None:
            sel_pars = input(' Would you like to build your own set or properties or use one of the defaults?\n My own --> 0\n Show defaults --> 1\n')
            choose = True
            print('\n')
        else:
            sel_pars = 1
            choose = False
        if int(sel_pars) == 1:
            #print('Here are the default set of properties, as seen in doi.org/10.21203/rs.3.rs-1287323/v1:')
            build1 = 'sn1up'
            set1 = ['Upper Leaflet, SN1 Tail (Build = %s)' % build1,
                    [True, False, 
                    True, False, 
                    True, False, 
                    True, False, 
                    True, False, 
                    True, False, 
                    True, False,
                    True, False,
                    True, False,
                    True, False,
                    True, False,
                    True, False,
                    False, False,
                    True,
                    True, False,
                    True, False,
                    True, False,
                    False, False,
                    False, False,
                    False, False]]
            build2 = 'sn2up'
            set2 = ['Upper Leaflet, SN2 Tail (Build = %s)' % build2, 
                    [True, False, 
                    True, False, 
                    True, False, 
                    True, False, 
                    True, False, 
                    True, False, 
                    True, False,
                    True, False,
                    True, False,
                    True, False,
                    True, False,
                    False, False,
                    True, False,
                    True,
                    False, False,
                    False, False,
                    False, False,
                    True, False,
                    True, False,
                    True, False]]
            build3 = 'sn1dn'
            set3 = ['Lower Leaflet, SN1 Tail (Build = %s)' % build3,
                    [False, True, 
                    False, True, 
                    False, True, 
                    False, True, 
                    False, True, 
                    False, True, 
                    False, True,
                    False, True,
                    False, True,
                    False, True,
                    False, True,
                    False, True,
                    False, False,
                    True,
                    False, True,
                    False, True,
                    False, True,
                    False, False,
                    False, False,
                    False, False]]
            build4 = 'sn2dn'
            set4 = ['Lower Leaflet, SN2 Tail (Build = %s)' % build4,
                    [False, True, 
                    False, True, 
                    False, True, 
                    False, True, 
                    False, True, 
                    False, True, 
                    False, True,
                    False, True,
                    False, True,
                    False, True,
                    False, True,
                    False, False,
                    False, True,
                    True,
                    False, False,
                    False, False,
                    False, False,
                    False, True,
                    False, True,
                    False, True]]
            build5 = 'redux'
            set5 = ['Reduced set of variables (Build = %s)' % build5, 
                    [False, False, 
                    True, False, 
                    False, False, 
                    True, False, 
                    True, False, 
                    False, False, 
                    True, False,
                    True, False,
                    False, False,
                    True, False,
                    False, False,
                    True, False,
                    False, False,
                    True,
                    False, False,
                    False, False,
                    True, False,
                    False, False,
                    False, False,
                    False, False]]
            
            sets = [set1,set2,set3,set4,set5]
            builds = [build1, build2, build3, build4, build5]
            if choose == False:
                selection = builds.index(self.Build)
            else:
                selection = int(input(' %s --> 0\n %s --> 1\n %s --> 2\n %s --> 3\n %s --> 4\n' % (set1[0], set2[0], set3[0], set4[0], set5[0])))
            Set = self.ALLVARS[sets[selection][1]]
            self.Names = self.ALLNAMES[sets[selection][1]]
            self.plotnames = []
            for n in self.Names:
                self.plotnames.append(self.NAMESDICT[n])
            self.names_dict = {}
            for idx, name in enumerate(self.Names):
                self.names_dict[name] = idx
            self.Build = builds[selection]
            self.DatMat = self.build(Set,self.Names)
            with open('./Pickles/svd_datamat_%s.pickle' % self.Build, 'wb') as f:
                pickle.dump(self.DatMat,f)
            f.close()
        else:
            cont = 0            
            while cont == 0:
                Set, self.Names, cont = self.select()
            self.names_dict = {}
            for idx, name in enumerate(self.Names):
                self.names_dict[name] = idx
            self.Build = input(' Enter a Build name for your chosen selection (Ex: user, myset...)\n Please do not enter any underscore (_) in the build name.')
            self.DatMat = self.build(Set,self.Names)
            
            with open('./Pickles/svd_datamat_%s.pickle' % self.Build, 'wb') as f:
                pickle.dump(self.DatMat, f)
            f.close()
            
    def Eigvals(self,SingVal):
        
        SingDiag = np.diag(self.SingVal)
        SigmaT = SingDiag.T                           #Eigenvalues relate to Singular Values such that Sigma.T x Sigma = Lambda  
        EigVals = np.diagonal(SigmaT.dot(SingDiag))
        
        return EigVals
        
    def Vars(self,EigVals):
        CVar = np.zeros(EigVals.shape[0])
        TVar = np.sum(EigVals)
        for idx, sigma in enumerate(EigVals):         #Compute cummulative variance from Eigvals
            if idx == 0:
                CVar[idx] += 100*sigma/TVar
            else:
                CVar[idx] += CVar[idx-1]+100*sigma/TVar
        EVar = np.zeros(EigVals.shape[0])
        for idx, sigma in enumerate(EigVals):
            EVar[idx] += 100*sigma/TVar
        return EVar, CVar
    
    def Relev(self,var, names_dict, Right, EVar):
        var = names_dict[var]
        VAR = Right[:,var]**2
        Rel = np.zeros(VAR.shape)                     #Compute Relevance by Eigvec for a given variable I have defined relevance as Ceff^2*EigVal
        for idx, i in enumerate(VAR):
            Rel[idx] += i*EVar[idx]*0.01
        Rel = 100*Rel/np.sum(Rel)
        
        return Rel
    
    def Relevance(self,EVar):
        Rel = np.zeros((len(self.Names),len(self.Names)))
        for i, var in enumerate(self.Names):
            rel = self.Relev(var, self.names_dict, self.RightMat, EVar) #Compute Rekevance Matrix
            rel /= np.linalg.norm(rel)
            Rel[i] += rel
        return Rel
    
    
    def Couplings(self,):
        varvar = np.zeros((len(self.Names),len(self.Names))) 
        for i in range(self.RelMat.shape[0]):
            r1 = self.RelMat[i]
            for j in range(self.RelMat.shape[0]): #Compute coupling Matrix
                if  i != j:
                    r2 = self.RelMat[j]
                    varvar[i,j] = np.inner(r1,r2)
        return varvar

    
    def ComputeSVD(self,):
    
        self.LeftMat, self.SingVal, self.RightMat = np.linalg.svd(self.DatMat, full_matrices = False)
        
        with open('./Pickles/svd_leftmat_%s.pickle' % self.Build, 'wb') as f:
            pickle.dump(self.LeftMat, f)
        f.close()
        with open('./Pickles/svd_singval_%s.pickle' % self.Build, 'wb') as f:
            pickle.dump(self.SingVal, f)
        f.close()
        with open('./Pickles/svd_rightmat_%s.pickle' % self.Build, 'wb') as f:
            pickle.dump(self.RightMat, f)
        f.close()
        
        self.EigVals = self.Eigvals(self.SingVal)
        self.EVar, self.CVar = self.Vars(self.EigVals)
        
        self.RelMat = self.Relevance(self.EVar)
        self.CoupMat = self.Couplings()
        
        with open('./Pickles/svd_relmat_%s.pickle' % self.Build, 'wb') as f:
            pickle.dump(self.RelMat, f)
        f.close()
        with open('./Pickles/svd_coupmat_%s.pickle' % self.Build, 'wb') as f:
            pickle.dump(self.CoupMat, f)
        f.close()
    
    def MakePlots(self,):
        
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable as mal

        fig, ax = plt.subplots(2,2,figsize = (12,12))
        ax[0,0].scatter(range(1, self.CVar.shape[0]+1), self.CVar, color = 'royalblue')
        ax[0,0].hlines(y=95, xmin = 1, xmax = self.CVar.shape[0], color = 'firebrick')
        ax[0,0].set_xticks(range(1,self.CVar.shape[0]+1,2))
        ax[0,0].set_xlabel('Nº of EigVec.', fontsize = 18, fontweight = 'bold')
        ax[0,0].set_ylabel(r'$\mathbf{C\sigma_{i}^{2}(\%)}$', fontsize = 18, fontweight = 'bold')
        ax[0,0].tick_params(axis = 'both', labelsize = 13)
        ax2 = ax[0,0].twinx()
        ax2.scatter(range(1, self.EVar.shape[0]+1), self.EVar, color = 'firebrick')
        ax2.set_xticks(range(1,self.CVar.shape[0]+1,2))
        ax2.set_xlabel('Nº of EigVec.', fontsize = 18, fontweight = 'bold')
        ax2.set_ylabel(r'$\mathbf{\sigma_{i}^{2}(\%)}$', fontsize = 22, fontweight = 'bold')
        ax2.tick_params(axis = 'both', labelsize = 13)
        im = ax[0,1].imshow(self.RightMat, origin='upper', cmap = 'seismic', 
                       extent = [-0.5, len(self.Names)-0.5, len(self.Names)+0.5, 0.5],
                       vmin = -1., vmax = 1.)
        ax[0,1].set_xticks(range(0, self.RightMat.shape[1]))
        ax[0,1].set_xticklabels(self.plotnames, rotation = 65)
        ax[0,1].set_yticks(range(1, self.RightMat.shape[0]+1))
        ax[0,1].set_xlabel('Vars.', fontsize = 18, fontweight = 'bold')
        ax[0,1].set_ylabel('EigVec.', fontsize = 18, fontweight = 'bold')
        ax[0,1].tick_params(axis  = 'both', labelsize = 13)
        divider = mal(ax[0,1])
        cax = divider.append_axes('right', size = '5%', pad = 0.05)
        cbar = fig.colorbar(im,cax=cax, format = '%.2f')
        cbar.set_ticks(np.linspace(-1, 1, 4))
        cbar.ax.tick_params(labelsize = 13)
        im = ax[1,0].imshow(self.RelMat.T, origin='upper', cmap = 'inferno', 
                       extent = [-0.5, len(self.Names)-0.5, len(self.Names)+0.5, 0.5])
        ax[1,0].set_xticks(range(0, self.RightMat.shape[1]))
        ax[1,0].set_xticklabels(self.plotnames, rotation = 65)
        ax[1,0].set_yticks(range(1, self.RightMat.shape[0]+1))
        ax[1,0].set_xlabel('Vars.', fontsize = 18, fontweight = 'bold')
        ax[1,0].set_ylabel('EigVecs.', fontsize = 18, fontweight = 'bold')
        ax[1,0].tick_params(axis = 'both', labelsize = 13)
        divider = mal(ax[1,0])
        cax = divider.append_axes('right', size = '5%', pad = 0.05)
        cbar = fig.colorbar(im,cax=cax, format='%.2f')
        cbar.set_ticks(np.linspace(np.min(self.RelMat), np.max(self.RelMat), 4))
        cbar.ax.tick_params(labelsize = 13)
        im = ax[1,1].imshow(self.CoupMat, origin='upper', cmap = 'inferno', extent = [-0.5, len(self.Names)-0.5, len(self.Names)+0.5, 0.5])
        ax[1,1].set_xticks(range(0, self.RightMat.shape[1]))
        ax[1,1].set_xticklabels(self.plotnames, rotation = 65)
        ax[1,1].set_yticks(range(1, self.RightMat.shape[0]+1))
        ax[1,1].set_yticklabels(self.plotnames, rotation = 0)
        ax[1,1].tick_params(axis='both', labelsize = 13)
        ax[1,1].set_xlabel('Vars.', fontsize = 18, fontweight = 'bold')
        ax[1,1].set_ylabel('Vars.', fontsize = 18, fontweight = 'bold')
        divider = mal(ax[1,1])
        cax = divider.append_axes('right', size = '5%', pad = 0.05)
        cbar = fig.colorbar(im,cax=cax,format='%.2f')
        cbar.set_ticks(np.linspace(np.min(self.CoupMat), np.max(self.CoupMat), 4))
        cbar.ax.tick_params(labelsize = 13)
        
        plt.tight_layout()
