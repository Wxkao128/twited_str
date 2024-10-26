# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 17:02:59 2024

@author: User
"""

import numpy as np
from matplotlib import pyplot as plt
import time

from lattice import Lattice, LayerSubAtom, TwistLayer
from utils import PlotLattice, CountAtomNum, POSCAR_generator

# ============================================================
#
# ======= Section 1: Setting parameters ======================
#
# ============================================================

n = 4 
m = 5   
nm = (m**2+n**2+4*m*n)/(2*(m**2+n**2+m*n))
t_a = np.arccos(nm) 
angle = round((t_a/np.pi)*180,2)  #twisted angle (degree)
default_str = False
round_digit = 12

# If necessary, you can enter the POSCAR file name
file = 'POSCAR_Sc2C_R3m.vasp'
#%%
# ============================================================
#
# ======= Section 2: Data Loading ============================
#
# ============================================================

if default_str == True:
    '''
    If default_str is True, the following structure will be used to generate 
    the AA-stacked twisted bilayer Sc2C.

    Structure details:
    - Material: Sc2C
    - Sublattice A: C
    - Sublattice B: Sc
    - Sublattice C: Sc

    Please ensure to consider these parameters when generating the structure.
    '''
    
    a1 = np.array([3.3251419067, 0.0])
    a2 = np.array([a1[0]*np.cos(np.pi*2/3), a1[0]*np.sin(np.pi*2/3)])
    
    layer1_sub_a = np.array([[0,0]]) #unit cell layer1 sublattice A
    layer1_sub_b = np.array([2/3*a1 + 1/3*a2]) #unit cell layer1 sublattice B   
    layer1_sub_c = np.array([1/3*a1 + 2/3*a2]) #unit cell layer1 sublattice C   
    
    layer2_sub_a = np.array([[0,0]]) #unit cell layer1 sublattice A
    layer2_sub_b = np.array([2/3*a1 + 1/3*a2]) #unit cell layer1 sublattice B   
    layer2_sub_c = np.array([1/3*a1 + 2/3*a2]) #unit cell layer1 sublattice C   

    params = { # 定義一個字典，包含所有參數
        'nx': 20,
        'ny': 20,
        't1': t_a,
        'a1': a1,
        'a2': a2
    }
    
elif default_str == False:
    
    a1, a2, atoms = Lattice(file).get_data()
    params = { # 定義一個字典，包含所有參數
        'nx': 20,
        'ny': 20,
        't1': t_a,
        'a1': a1,
        'a2': a2
    }

else:
    raise ValueError("Input must be True or False!")  # 拋出異常
    
atoms_tmp = atoms[:,0:2]  # get rid of z components temporarily
old_list = atoms[:,2].tolist()
index_order = [0, 3, 6, 1, 4, 7, 2, 5, 8]
h_list = [old_list[i] for i in index_order]
h_delta = np.mean([old_list[2]-old_list[1], old_list[1]-old_list[0]])*3

layer1_sub_1a = atoms_tmp[0]
layer1_sub_1b = atoms_tmp[3]
layer1_sub_1c = atoms_tmp[6]

layer1_sub_2a = atoms_tmp[1]
layer1_sub_2b = atoms_tmp[4]
layer1_sub_2c = atoms_tmp[7]

layer1_sub_3a = atoms_tmp[2]
layer1_sub_3b = atoms_tmp[5]
layer1_sub_3c = atoms_tmp[8]

layer2_sub_1a = atoms_tmp[0]
layer2_sub_1b = atoms_tmp[3]
layer2_sub_1c = atoms_tmp[6]

layer2_sub_2a = atoms_tmp[1]
layer2_sub_2b = atoms_tmp[4]
layer2_sub_2c = atoms_tmp[7]

layer2_sub_3a = atoms_tmp[2]
layer2_sub_3b = atoms_tmp[5]
layer2_sub_3c = atoms_tmp[8]

#%%    
# ============================================================
#
# ======= Step 3: Apply Data Transformations =================
#
# ============================================================

rot_layer1_sub_1a = LayerSubAtom(layer1_sub_1a,**params).rotation(rot=1, r_digit=round_digit)
rot_layer1_sub_1b = LayerSubAtom(layer1_sub_1b,**params).rotation(rot=1, r_digit=round_digit)
rot_layer1_sub_1c = LayerSubAtom(layer1_sub_1c,**params).rotation(rot=1, r_digit=round_digit)

rot_layer1_sub_2a = LayerSubAtom(layer1_sub_2a,**params).rotation(rot=1, r_digit=round_digit)
rot_layer1_sub_2b = LayerSubAtom(layer1_sub_2b,**params).rotation(rot=1, r_digit=round_digit)
rot_layer1_sub_2c = LayerSubAtom(layer1_sub_2c,**params).rotation(rot=1, r_digit=round_digit)

rot_layer1_sub_3a = LayerSubAtom(layer1_sub_3a,**params).rotation(rot=1, r_digit=round_digit)
rot_layer1_sub_3b = LayerSubAtom(layer1_sub_3b,**params).rotation(rot=1, r_digit=round_digit)
rot_layer1_sub_3c = LayerSubAtom(layer1_sub_3c,**params).rotation(rot=1, r_digit=round_digit)


rot_layer2_sub_1a = LayerSubAtom(layer2_sub_1a,**params).rotation(rot=2, r_digit=round_digit)
rot_layer2_sub_1b = LayerSubAtom(layer2_sub_1b,**params).rotation(rot=2, r_digit=round_digit)
rot_layer2_sub_1c = LayerSubAtom(layer2_sub_1c,**params).rotation(rot=2, r_digit=round_digit)

rot_layer2_sub_2a = LayerSubAtom(layer2_sub_2a,**params).rotation(rot=2, r_digit=round_digit)
rot_layer2_sub_2b = LayerSubAtom(layer2_sub_2b,**params).rotation(rot=2, r_digit=round_digit)
rot_layer2_sub_2c = LayerSubAtom(layer2_sub_2c,**params).rotation(rot=2, r_digit=round_digit)

rot_layer2_sub_3a = LayerSubAtom(layer2_sub_3a,**params).rotation(rot=2, r_digit=round_digit)
rot_layer2_sub_3b = LayerSubAtom(layer2_sub_3b,**params).rotation(rot=2, r_digit=round_digit)
rot_layer2_sub_3c = LayerSubAtom(layer2_sub_3c,**params).rotation(rot=2, r_digit=round_digit)


# 實例化
tw1 = TwistLayer(rot_layer1_sub_1a, rot_layer1_sub_1b, rot_layer1_sub_1c,
                 rot_layer1_sub_2a, rot_layer1_sub_2b, rot_layer1_sub_2c,
                 rot_layer1_sub_3a, rot_layer1_sub_3b, rot_layer1_sub_3c)

tw2 = TwistLayer(rot_layer2_sub_1a, rot_layer2_sub_1b, rot_layer2_sub_1c,
                 rot_layer2_sub_2a, rot_layer2_sub_2b, rot_layer2_sub_2c,
                 rot_layer2_sub_3a, rot_layer2_sub_3b, rot_layer2_sub_3c)

# 合併子晶格
rot_layer1 = tw1.concatenate_sublattices()
rot_layer2 = tw2.concatenate_sublattices()

# 計算共同原子
coincident_12 = tw1.coincident_atom(rot_layer1_sub_3a, rot_layer2_sub_3a)

# 獲取頂點
atomO, atomA, atomB, atomC = tw1.find_vertex()
#%%
# ============================================================
#
# ======= Step 4: If necessary, plot the lattice structure ===
#
# ============================================================

pl = PlotLattice(angle)
pl.plot_structure(rot_layer1_sub_1a, color='r')
pl.plot_structure(rot_layer1_sub_1b, color='r')
pl.plot_structure(rot_layer1_sub_1c, color='r')
pl.plot_structure(rot_layer2_sub_1a, color='b')
pl.plot_structure(rot_layer2_sub_1b, color='b')
pl.plot_structure(rot_layer2_sub_1c, color='b')
pl.plot_unitcell(atomO, atomB, atomA, atomC )
pl.plot_coincident(coincident_12)
#%%
# ============================================================
#
# ======= Step 5: Counting # of atoms and creat POSCAR =======
#
# ============================================================

can_tmp = CountAtomNum(atomO, atomB, atomA, atomC, n, m)
atom_layers = [rot_layer1_sub_1a, rot_layer1_sub_1b, rot_layer1_sub_1c,
               rot_layer1_sub_2a, rot_layer1_sub_2b, rot_layer1_sub_2c,
               rot_layer1_sub_3a, rot_layer1_sub_3b, rot_layer1_sub_3c,
               
               rot_layer2_sub_1a, rot_layer2_sub_1b, rot_layer2_sub_1c,
               rot_layer2_sub_2a, rot_layer2_sub_2b, rot_layer2_sub_2c,
               rot_layer2_sub_3a, rot_layer2_sub_3b, rot_layer2_sub_3c]


h_values = h_list + [x+h_delta for x in h_list ]
stick_points = [None, None, True, None, True, None, True, None, None, 
                None, None, True, None, True, None, True, None, None]

can = [can_tmp.count_atom_num(layer, h, stick_point=sp) 
           for layer, h, sp in zip(atom_layers, h_values, stick_points)]

# 計算 total_num
num_c  = sum(len(can[i]) for i in [0,3,6,9,12,15])
num_sc = sum(len(can[i]) for i in [1,2,4,5,7,8,10,11,13,14,16,17])

params = { # 定義一個字典，包含所有參數
    'z_h': 45.0,
    'material_name': 'TBSc2C_R3m',
    'angle': angle,
    'stack_conf': 'AA', #'AA', 'AB', ...
    'suffix': '',
    'atom_types': ['C','Sc'],
    'total_num': [num_c, num_sc],
    'can': can,
    'sub_index': [[0,3,6,9,12,15],[1,2,4,5,7,8,10,11,13,14,16,17]]
}

POSCAR_generator(atomO,atomA,atomB,atomC, **params)