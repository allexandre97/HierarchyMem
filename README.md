# HierarchyMem.

HierarchyMem aims to be an all-in-one script to analyze simulations of flat lipid bilayers and find subjacent higher order structures. Please note that this is still a project under development and might be subject to substantial changes.

## The HierarchyMem module and its dependencies.

HierarchyMem is written in **Python 3** and needs several dependencies installed for it to properly work. 

  * The module should be run on a machine with a CUDA enabled GPU card, and the [numba](https://numba.pydata.org/) library installed. 
  * The interface with the output GROMACS simulation data is handled by the [MDAnalysis](https://www.mdanalysis.org/) library. 
  * The bakcend for figure generation is [Matplotlib](https://matplotlib.org/). 
  * Mathematical operations and data handling are done through [NumPy](https://numpy.org/). 

It is strongly recommended to build a virtual Python 3 environment (see [pip](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) and/or [Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) docs for detailed guides on how to do so) from which then execute the module, as a way to avoid possible conflicts with previously installed packages.

## How to rapidly execute HierarchyMem.

Two .py files are provided in this repository. **HierarchyMem.py** is the library that executes the analyses (described in [here](https://www.researchsquare.com/article/rs-1287323/v1)) on the lipid membrane, and builds the [.pickle files](https://docs.python.org/3/library/pickle.html) used to store the results. **Execute.py** is an example wrapper script that calls for the functions of HierarchyMem to carry out the analyses.

These two files should be put in the same folder. By running ```python3 Execute.py``` the program will be executed with its default values. The module will ask for a route to a structure (.gro, .pdb, ...) and to a trajectory (.trr, .xtc, ...) file and then perform the analyses automatically. A minimal example .pdb and .xtc files can be found [here](https://www.dropbox.com/sh/69piqqumkbvmdvo/AACrYtswccGZYITTPBWrYLdwa?dl=0). 

The script will create a Pickles folder in the directory it is being executed. All the pickle files created will be stored there, which allows for faster re-execution and further manual post-processing if desired. Once the analyses are finished, several figures will be created and shown interactively with equivalent data as the one seen [here](https://www.researchsquare.com/article/rs-1287323/v1).

## A bit more of control on HierarchyMem.

So far, HierarchyMem is meant to be utilized through a python wrapper similar to **Execute.py**. The philosophy of the module's execution lies on the sequential calling of its conforming Classes. Here a bit more in-detail explanation of each one of them, and their return information will be given.

### HierarchyMem.SearchMem()

This is the class used to perform the membrane's discretization in a square grid and the search algorithm that maps several properties to said grid. It can be initialized as
 
 ```python
 import HierarchyMem as hm 
 SMem = hm.SearchMem(select=True)
 ```
 
This will create a SMem dependency, used to interact with the SearchMem class of the module. The argument *select* is of ```Bool``` type and is used to manually define the search grid properties, these are, the *approximate* desired number of lipids that should fit in a grid cell; and an extension window so each grid cell becomes a 2D moving average though the circundant grids.

  * A grid without binning window, only the elements present in a cell will be stored there:

![grid_b0](https://user-images.githubusercontent.com/60816362/153607055-ceebefdc-fec5-478e-b937-dab3ddba992c.png)

  * A grid with 1 cell binning, the elements in circundant cells will be also stored:

![grid_b1](https://user-images.githubusercontent.com/60816362/153607097-aca616a6-1676-423f-a585-3b790a7f5b96.png)

Not setting *select* or setting it to ```False``` will use the default values, 10 lipids per cell and no cell extensions. 

Once the class has been initialized, one can start the analysis by calling the method:
```python
SMem.DoSearch()
```
After finishing the analysis, this class will return several .pickle files:

  * *pos_up/dn.pickle* --> A (NFrames, NLipidspercell, NCellsX, NCellsY, 3) array, to store the lipid positions in the upper (up) or lower (dn) leaflet. Can be called as 
  ```python
  _ = SMem.POSu/d
  ```              
  * *sn1_up/dn.pickle* --> A (NFrames, NLipidspercell, NCellsX, NCellsY, 3) array, to store the lipid sn1 tail vectors in the upper (up) or lower (dn) leaflet. Can be called as 
  ```python
  _ = SMem.SN1u/d
  ```
  * *sn2_up/dn.pickle* --> A (NFrames, NLipidspercell, NCellsX, NCellsY, 3) array, to store the lipid sn2 tail vectors in the upper (up) or lower (dn) leaflet. Can be called as 
  ```python
  _ = SMem.SN2u/d
  ```
  * *rid_up/dn.pickle* --> A (NFrames, NLipidspercell, NCellsX, NCellsY, 1) array, to store the lipid residue index in the upper (up) or lower (dn) leaflet. Can be called as 
  ```python
  _ = SMem.RIDu/d
  ```
  * *rnm_up/dn.pickle* --> A (NFrames, NLipidspercell, NCellsX, NCellsY, 1) array, to store the lipid residue name in the upper (up) or lower (dn) leaflet. Can be called as 
  ```python
  _ = SMem.RNMu/d
  ```
  * *gridX.pickle* --> A (NCellsX+1) array to store the cell edges along the X dimension. Can be called as 
  ```python
  _ = SMem.gridX
  ```
  * *gridY.pickle* --> A (NCellsY+1) array to store the cell edges along the Y dimension. Can be called as 
  ```python
  _ = SMem.gridY
  ```
  
  ### HierarchyMem.Promed()
  
  This is the class used to promediate the properties in each grid cell, thus reducing the membrane's dimensionality and effectively converting it to a discrete surface per leaflet, containing a series of average properties at each point. It can be called as:
  ```python
  Prom_ = hm.Promed(Prop='_')
  ```
  
  The ```Prop``` argument is of ```str``` type and is used to define which Property is to be promediated. The available properties are the three starting characters of the names of the previously defined pickles, so  *pos, sn1 and sn2*. Note that one could also average the values of the resnames or the resids, as they are translated to be of the ```float``` type, but so far there are no further implementations for the averages obtained of such analyses.
  
  After the class is initialized, the averaging algorithm can be done by calling:
  ```python
  Prom_.GetPromed()
  ```
  
  When the analysis is finished a new *.pickle* file will be created, following a similar naming convention, such as if the **Position** is the selected property, the resulting pickle file will be:
  
  * *prom_**pos**.pickle* --> A (NFrames, NLeaflets, NCellsX, NCellsY, 3) array, storing the average position of the each leaflet at each XY point.
  
  Also, the result of the averaging algorithm can be called *in-script* by:
  ```python
  _ = Prom_.Result
  ```
  
  ### HierarchyMem.Normals()
  
  The class used to compute the normal vector to each point of the discretized membrane. In order to execute this class, **prom_pos.pickles must exist**, as they are the default input. To initialize the class and perform the analysis simply write:
  ```python
  Norms = hm.Normals()
  Norms.ComputeNormals()
  ```
  
  No arguments are needed for neither of the methods, and the analysis will run automatically. As in previous cases, a *.pickle* file will be generated:
  * *normals.pickle* --> A (NFrames, NLeaflets, NCellsX, NCellsY, 3) array, storing the normal vectors at each point of the two leaflets.
  
  Equivalently, one can call this array *in-script* via:
  ```python
  _ = Norms.Result
  ```
  
  ### HierarchyMem.LRS()
  
  This class is used to compute the Local Reference System at each point of the discretized membrane. The class can be initialized and executed such as:
  ```python
  _ = hm.LRS()
  _.ComputeLRS()
  ```
  
  No arguments are needed as the analyses are performed automatically given that the position averages and normals have already been calculated. If this isn't the case, the LRS **won't be able to be calculated**.
  
  The output *.pickle* file is:
  * *lrs.pickle* --> A (NFrames, NLeaflets, NCellsX, NCellsY, 3, 3) array, storing the three LRS vectors at each point of each leaflet.
  
  The result can be called *in-script* as:
  ```python
  LRS = _.Result
  ```
  
  
  
  
