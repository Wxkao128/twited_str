# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 15:33:37 2024

@author: User
"""

import numpy as np
from scipy.spatial import distance

class Lattice:
    def __init__(self, file_name):
        #self.file = kwargs.get('file', None)  
        self.file = file_name
        self.a1 = None   # a-axis in real space 
        self.a2 = None   # b-axis in real space

        
        if self.file is not None:
            self.read_file()
        
    def read_file(self):
        array1 = []  # 用於存儲第3到第5行的內容
        array2 = []  # 用於存儲第9行開始的內容
        
        with open(self.file, 'r') as file:
            for i, line in enumerate(file):
                stripped_line = line.strip()  
                

                if 2 <= i <= 4: 
                    array1.append([float(x) for x in stripped_line.split()])
                

                if i >= 8:  
                    array2.append([float(x) for x in stripped_line.split()])
                
        # numpy array
        array1 = np.array(array1)
        array2 = np.array(array2)
        return array1, array2
    
    def get_data(self):
        lattice_vec, atom_coord = self.read_file()
        self.a1 = np.array([lattice_vec[0][0], 0.0, 0.0])
        self.a2 = np.array([self.a1[0]*np.cos(np.pi*2/3), self.a1[0]*np.sin(np.pi*2/3), 0.0])
        return self.a1, self.a2, atom_coord
        

class LayerSubAtom():
    def __init__(self, layerp_sub_q, **kwargs):
        
        #  nx ny
        self.nx = kwargs.get('nx', 20)  # default: 20
        self.ny = kwargs.get('ny', 20)  # default: 20
        self.t1 = kwargs.get('t1', 0)   # t1 (twisted angle)
        self.a1 = kwargs.get('a1', 0)
        self.a2 = kwargs.get('a2', 0)
        
        self.lp_atoms = []
        self.layerp_sub_q = layerp_sub_q
        
        self.rot_mat1 = np.array([[np.cos(self.t1/2),-np.sin(self.t1/2), 0],  #for layer 1
                                  [np.sin(self.t1/2), np.cos(self.t1/2), 0],
                                  [0,                 0,                 1]])
        
        self.rot_mat2 = np.array([[ np.cos(self.t1/2),np.sin(self.t1/2), 0],   #for layer 2
                                  [-np.sin(self.t1/2),np.cos(self.t1/2), 0],
                                  [0,                 0,                 1]])
        
    def translation(self):
        #nx = 40  #number of vector for x direction
        #ny = nx  #number of vector for y direction
        
        for i in range(-self.nx, self.nx+1):
            for j in range(-self.ny, self.ny+1):
                translation_vector = i * self.a1 + j * self.a2
                translated_atoms_lp = self.layerp_sub_q + translation_vector
                self.lp_atoms.append(translated_atoms_lp)
        return np.squeeze(np.array(self.lp_atoms))
    
    def rotation(self, rot, r_digit):
        lattice_layerp_sub_q = self.translation()
        
        # control rotation matrix
        if rot == 1:
            rot_layerp_sub_q = np.dot(lattice_layerp_sub_q, self.rot_mat1)
        elif rot == 2:
            rot_layerp_sub_q = np.dot(lattice_layerp_sub_q, self.rot_mat2)
            
        rot_layerp_sub_q = np.around(rot_layerp_sub_q,r_digit)
        return rot_layerp_sub_q


class TwistLayer:
    def __init__(self, *sublattices):
        self.sublattices = sublattices
    
    def concatenate_sublattices(self):
        """
        supercell
        """
        self.rot_layer = np.concatenate(self.sublattices)
        return self.rot_layer
    
    def coincident_atom(self, rot_layerp_sub_q, rot_layerr_sub_s):
        
        #Find the same coordinate in three layers and denote them as black dots
        #coincident_12 = np.intersect1d(rot_layer1.view(dtype), rot_layer2.view(dtype)) #combine layer1 and layer2 first
        # modified version: C coincident with C
        
        # read (x 和 y)
        rot_layerp_sub_q_xy = rot_layerp_sub_q[:, 0:2]
        rot_layerr_sub_s_xy = rot_layerr_sub_s[:, 0:2]
        self.coincident_12 = np.array([row for row in rot_layerp_sub_q_xy if any(np.all(np.isclose(row, rot_layerr_sub_s_xy), axis=1))])
       
        return self.coincident_12
    
    #contain the commands relate finding vertex of supercell
    def find_vertex(self): 
        
        if self.coincident_12 is None:
            raise ValueError("coincident_12 has not been calculated. Please call coincident_atom first.")
        
        
        #define a funciton for finding element which is closest to a specific point
        def find_nearest_vector(array, value):
            idx = np.array([np.linalg.norm(x+y) for (x,y) in array-value]).argmin()
            return array[idx]

        #Find the (0,0) or the closest point as start point of the supercell
        vertex = np.array([])
        pt = np.array([0,0])
        vertex = np.append(vertex,find_nearest_vector(self.coincident_12,pt)) #find closest O(0,0) point
        print('vertex =',vertex)
        #atom O as the supercell's reference point, very important!!!
        atomo = vertex
        
        leftt_o_max = self.coincident_12[np.where(self.coincident_12[:,0]<vertex[0])][:,0].max() #a maximum number in the  left of the atomO's x coordinate
        right_o_min = self.coincident_12[np.where(self.coincident_12[:,0]>vertex[0])][:,0].min() #a minimum number in the right of the atomO's x coordinate

        test1 = self.coincident_12[np.where(self.coincident_12[:,0]<vertex[0])] #for atomC 
        test2 = self.coincident_12[np.where(self.coincident_12[:,0]>vertex[0])] #for atomB

        test_array1 = test1[np.where(test1[:,0]==leftt_o_max)]
        test_array2 = test2[np.where(test2[:,0]==right_o_min)]
        test_y1 = test_array1[np.where(test_array1[:,1]>vertex[1])][:,1].min()
        test_y2 = test_array2[np.where(test_array2[:,1]>vertex[1])][:,1].min()

        #atom B  for test whether the supercell is diamond or hexagon
        Bx = test_array2[np.where(test_array2[:,1]>vertex[1])][:,0].min()
        By = test_array2[np.where(test_array2[:,1]>vertex[1])][:,1].min()
        atomb = np.array([Bx,By])
        #atom A' for test whether the supercell is diamond or hexagon
        test3 = self.coincident_12[np.where(self.coincident_12[:,0]==vertex[0])] #for atomAp
        Apy = test3[np.where(test3[:,1]>vertex[1])][:,1].min()
        Apx = vertex[0]
        atoma = np.array([Apx,Apy])
        #atom C  for test whether the supercell is diamond or hexagon
        Cx = test_array1[np.where(test_array1[:,1]>vertex[1])][:,0].max()
        Cy = test_array1[np.where(test_array1[:,1]>vertex[1])][:,1].min()
        atomc = np.array([Cx,Cy])
        
        if By == Cy:
            #pseudo code:
            #if L1 == L2 => this is still tilt diamond 
            #if not => this is hexagon(vertical hexagon) 
            if round(distance.euclidean(atoma, atomo),10) == round(distance.euclidean(atomb, atomo),10):
                #the supercell is tilt diamond shape as before
                print("supercell is 'horizontal diamond' ")
            else:
                if round(distance.euclidean(atoma, atomb),10) == round(distance.euclidean(atomo, atomb),10):
                    print("supercell is 'vertical diamond' ")
                else:
                    print("supercell is 'vertical hexagon' ")
                    if By > Apy:
                        #type 1 vertical hexagon A' is lower than B
                        Ay = test3[np.where(test3[:,1]>Apy)][:,1].min()
                        Ax = Apx
                        atoma = [Ax,Ay]
                    else:
                        #type 2 vertical hexagon A' is higher than B
                        Bpx = test2[np.where(test2[:,0]==Bx)][:,0].min()
                        Bpy = test2[np.where(test2[:,1]>By)][:,1].min()
                        atomb = [Bpx,Bpy]
                        
                        Cpx = test1[np.where(test1[:,0]==Cx)][:,0].min()
                        Cpy = test1[np.where(test1[:,1]>Cy)][:,1].min()
                        atomc = [Cpx,Cpy]
                        
                        Ay = test3[np.where(test3[:,1]>Apy)][:,1].min()
                        Ax = Apx
                        atoma = [Ax,Ay]
        else:
            #the supercell is hexagon shape, this hexagon is horizontal hexagon   
            print("supercell is 'horizontal hexagon' ")
            if Apy > By:
                #type 1 horizontal hexagon A' is higher than B
                Bpx = test2[np.where(test2[:,0]>Bx)][:,0].min()
                Bpy = test2[np.where(test2[:,1]==By)][:,1].max()
                atomb = [Bpx,Bpy]
                
                Cpx = test1[np.where(test1[:,0]<Cx)][:,0].max()
                Cpy = test1[np.where(test1[:,1]<Cy)][:,1].max()
                atomc = [Cpx,Cpy]
            else:
                #type 2 horizontal hexagon A' is equal high with B
                Bpx = test2[np.where(test2[:,0]>Bx)][:,0].min()
                Bpy = test2[np.where(test2[:,1]<By)][:,1].max()
                atomb = [Bpx,Bpy]
                
                Cpx = test1[np.where(test1[:,0]<Cx)][:,0].max()
                Cpy = Cy #in this case, Cpy = Cy = Bpy, so they are equal high
                atomc = [Cpx,Cpy]
                
        return atomo, atoma, atomb, atomc
    
    def save_atoms(self):
        pass
        # np.save('rot_layer1a.npy', rot_layer1_sub_a)
        # np.save('rot_layer1b.npy', rot_layer1_sub_b)
        # np.save('rot_layer2a.npy', rot_layer2_sub_a)
        # np.save('rot_layer2b.npy', rot_layer2_sub_b)





