#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 13:38:44 2022

@author: xandre
"""

import HierarchyMem as hm

#Execute the Search algorithm
SMem = hm.SearchMem()
SMem.DoSearch()

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

#Execute the tail angle determination algorithm
SN1Ang = hm.TailAngles('sn1') #Between brackets which tail to analyze
SN1Ang.ComputeAngle()
SN2Ang = hm.TailAngles('sn2')
SN2Ang.ComputeAngle()

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














