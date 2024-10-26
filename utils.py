# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 14:23:52 2024

@author: User
"""


import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import distance
#from scipy.sparse.linalg import eigsh
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import time


class PlotLattice:
    def __init__(self, angle, size=30):
        # 創建一個圖和軸對象，只創建一次
        self.fig, self.ax = plt.subplots()
        self.fig.set_size_inches(10, 8.5)      
        self.angle = angle
        self.size = size

    def plot_structure(self,rot_layerp_sub_q, color='blue'):

        #plot layer p sublattice q
        self.ax.scatter(rot_layerp_sub_q[:,0], rot_layerp_sub_q[:,1], color=color, s=self.size)   
        self.ax.scatter(0,0,color="black",s=self.size*2)    #denote the origin(0,0)
        self.ax.set_title(f"Twisted Structure ~ {self.angle}°")
        self.ax.set_aspect("equal")
        plt.show()
        
    def plot_coincident(self, coincident_atoms):

        self.ax.scatter(coincident_atoms[:,0],coincident_atoms[:,1],
                        color="black",s=self.size*2) #plot coincident atoms
        plt.show()
        
    #Plotting the supercell 
    def plot_unitcell(self, vertex, atomb, atoma, atomc):
        vertex = np.append(vertex,atomb)
        vertex = np.append(vertex,atoma)
        vertex = np.append(vertex,atomc)
        vertexk = vertex.reshape(4,2) #reshaped vertex
        unitcell_x = [vertexk[0,0],vertexk[1,0],vertexk[2,0],vertexk[3,0],vertexk[0,0]]
        unitcell_y = [vertexk[0,1],vertexk[1,1],vertexk[2,1],vertexk[3,1],vertexk[0,1]]
        plt.plot(unitcell_x,unitcell_y,c='C0',lw=2)
        plt.show()
        return vertex, vertexk


#Counting atoms in supercell
class CountAtomNum:
    def __init__(self, atomo, atomb, atoma, atomc, nn, mm):
        """
        Initializes CountAtomNum with given atom arrays and configuration parameters.

        Parameters
        ----------
        atomo : numpy.ndarray
            The coordinates of the first atom type (e.g., O atoms).
        atoma : numpy.ndarray
            The coordinates of the second atom type (e.g., A atoms).
        atomb : numpy.ndarray
            The coordinates of the third atom type (e.g., B atoms).
        atomc : numpy.ndarray
            The coordinates of the fourth atom type (e.g., C atoms).
        nn : int
            The integer parameter n to construct twisted angle.
        mm : int
            The integer parameter m to construct twisted angle.
        """
        
        self.atomo = atomo
        self.atoma = atoma
        self.atomb = atomb
        self.atomc = atomc
        self.total_num = 0
        self.n = nn
        self.m = mm

    
    def count_atom_num(self, rot_layerp_sub_q, height, stick_point=None):
        
        polygon = Polygon([self.atomo, self.atomb, self.atoma, self.atomc])
        
        # create temporary rot_layer1&2 only contain atom's y coordinate > -0.1 for saving search time 
        temp_rot_layerp_sub_q = rot_layerp_sub_q[((rot_layerp_sub_q[:,1]>-0.1) & (rot_layerp_sub_q[:,0]>self.atomc[0]-0.1) & (rot_layerp_sub_q[:,0]<self.atomb[0]+0.1))] 
      
        # This way is more concise than above one but less clear,just restore two lines
        # Use list comprehension to filter points that are contained within the polygon
        #in_supercell_layer1_sub_a.extend([point for point in rot_layer1_sub_a if polygon.contains(Point(point[0], point[1]))])
        #in_supercell_layer2_sub_b.extend([point for point in rot_layer2_sub_b if polygon.contains(Point(point[0], point[1]))])
        lpq = np.array([point for point in temp_rot_layerp_sub_q if polygon.contains(Point(point[0], point[1]))])
        

        #for AA stacking: atomo should be included in in_supercell_layer1_sub_A and in_supercell_layer2_sub_A
        #for AB stacking: atomo should be included in in_supercell_layer1_sub_A and in_supercell_layer2_sub_B
        if stick_point:
            if self.n == self.m:
                lpq = self.atomo[np.newaxis,:]
                #l1a = atomo[np.newaxis,:]
                #l2a = atomo[np.newaxis,:]
            else:
                #AA stacking:
                lpq = np.vstack((self.atomo, lpq))
                #l2a = np.vstack((atomo, l2a))
                #AB stacking:
                #l1a = np.vstack((atomo, l1a))
                #l2b = np.vstack((atomo, l2b))
        
        del temp_rot_layerp_sub_q
            
        #add z component to each atom
        lpq = np.hstack((lpq,np.full((lpq.shape[0], 1), height)))
        return lpq
        


def POSCAR_generator(atomo,atoma,atomb,atomc,**kwargs):    
    """
    Generate POSCAR file based on given atoms and configuration.

    Parameters
    ----------
    atomo : numpy.ndarray
        The first atom's coordinates in the lattice.
    atoma : numpy.ndarray
        The second atom's coordinates in the lattice.
    atomb : numpy.ndarray
        The third atom's coordinates in the lattice.
    atomc : numpy.ndarray
        The fourth atom's coordinates in the lattice.

    **kwargs : dict
        A dictionary of additional parameters:
        
        - 'z_h' (float): the length of c-axis (vacuum included) 
        - 'material_name' (str): material name
        - 'angle' (float): The rotation angle between layers.
        - 'stack_conf' (str): Stacking configuration of the layers (e.g., 'AA', 'AB').
        - 'suffix' (str): File name suffix (e.g., '_tmp')
        - 'atom_types' (list[str]): element types 
        - 'total_num' (list[int]): Total number of atoms.
        - 'can' (list): A list of atomic configurations.
        - 'sub_index' (list[list[int]]): number of each type of atom
        
    Returns
    -------
    None.
    """
    z_h = kwargs.get('z_h', 35)
    mat_name = kwargs.get('material_name', 'TBSc2C')
    angle = kwargs.get('angle', 0)  # 如果 'angle' 不存在，則默認為 0
    stack_conf = kwargs.get('stack_conf', 'None')
    suffix = kwargs.get('suffix', '')
    atom_types = kwargs.get('type', ['C','Sc'])
    total_num = kwargs.get('total_num', [1,2])
    can = kwargs.get('can', None)
    sub_index = kwargs.get('sub_index', [[0,3],[1,2,4,5]])

    #Define lattice vertors of supercell
    l1 = np.array(atomb) - np.array(atomo)
    l2 = np.array(atomc) - np.array(atomo)
    L1 = [l1[0], l1[1], 0]  
    L2 = [l2[0], l2[1], 0]
    L3 = [0, 0, z_h]  #z方向 20為考慮真空層的厚度(12A)
    #nol= len(can)  #number of layer and sublattice
    
    path = f'POSCAR_{mat_name}_{int(angle*100)}_{stack_conf}'
    with open(path, 'w') as f:
        f.write(f'{mat_name} {stack_conf} stacking with {angle} twisted angle\n')
        f.write('1.0\n') #scaling factor
        f.write("{:12.8f} {:12.8f} {:12.8f}\n".format(L1[0], L1[1], L1[2]))
        f.write("{:12.8f} {:12.8f} {:12.8f}\n".format(L2[0], L2[1], L2[2]))
        f.write("{:12.8f} {:12.8f} {:12.8f}\n".format(L3[0], L3[1], L3[2]))
        for i in atom_types:
            f.write(f'{i} ')
        f.write('\n')
        for i in total_num:
            f.write(f'{i} ')
        f.write('\n')
        f.write('C\n')
        s1=time.time()
        #Coordinates for the each atom
        for i in sub_index[0]: # for sublattice A
            for j in range(len(can[i])):
                f.write(f"{can[i][j][0]:12.8f} {can[i][j][1]:12.8f} {can[i][j][2]:12.8f}\n")
        for i in sub_index[1]: # for sublattice B
            for j in range(len(can[i])):
                f.write(f"{can[i][j][0]:12.8f} {can[i][j][1]:12.8f} {can[i][j][2]:12.8f}\n")
        e1=time.time()
    print('POSCAR was written successfully.')
    #print(f'cost time: {e1-s1}')

