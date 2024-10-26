# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 15:47:03 2024

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

n = 1 
m = 2   
nm = (m**2+n**2+4*m*n)/(2*(m**2+n**2+m*n))
t_a = np.arccos(nm) 
angle = round((t_a/np.pi)*180,2)  #twisted angle (degree)
default_str = True
round_digit = 12

# If necessary, you can enter the POSCAR file name
file = 'POSCAR_Sc2C_R3m.vasp'

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
    
# ============================================================
#
# ======= Step 3: Apply Data Transformations =================
#
# ============================================================

h1 = 10.000000000   # layer1: C
h2 =  8.885497078   # layer1: Sc-down
h3 = 11.320742815   # layer1: Sc-top

h4 = 15.628324672   # layer2: C
h5 = 14.307581856   # layer2: Sc-down
h6 = 16.742827966   # layer2: Sc-top

rot_layer1_sub_a = LayerSubAtom(layer1_sub_a,**params).rotation(rot=1, r_digit=round_digit)
rot_layer1_sub_b = LayerSubAtom(layer1_sub_b,**params).rotation(rot=1, r_digit=round_digit)
rot_layer1_sub_c = LayerSubAtom(layer1_sub_c,**params).rotation(rot=1, r_digit=round_digit)

rot_layer2_sub_a = LayerSubAtom(layer2_sub_a,**params).rotation(rot=2, r_digit=round_digit)
rot_layer2_sub_b = LayerSubAtom(layer2_sub_b,**params).rotation(rot=2, r_digit=round_digit)
rot_layer2_sub_c = LayerSubAtom(layer2_sub_c,**params).rotation(rot=2, r_digit=round_digit)

# 實例化
tw1 = TwistLayer(rot_layer1_sub_a, rot_layer1_sub_b, rot_layer1_sub_c)

# 合併子晶格
rot_layer1 = tw1.concatenate_sublattices()
rot_layer2 = TwistLayer(rot_layer2_sub_a,
                        rot_layer2_sub_b,
                        rot_layer2_sub_c
                        ).concatenate_sublattices()

# 計算共同原子
coincident_12 = tw1.coincident_atom(rot_layer1_sub_a, rot_layer2_sub_a)

# 獲取頂點
atomO, atomA, atomB, atomC = tw1.find_vertex()

# ============================================================
#
# ======= Step 4: If necessary, plot the lattice structure ===
#
# ============================================================

pl = PlotLattice(angle)
pl.plot_structure(rot_layer1_sub_a, color='r')
pl.plot_structure(rot_layer1_sub_b, color='r')
pl.plot_structure(rot_layer1_sub_c, color='r')
pl.plot_structure(rot_layer2_sub_a, color='b')
pl.plot_structure(rot_layer2_sub_b, color='b')
pl.plot_structure(rot_layer2_sub_c, color='b')
pl.plot_unitcell(atomO, atomB, atomA, atomC )
pl.plot_coincident(coincident_12)

# ============================================================
#
# ======= Step 5: Counting # of atoms and creat POSCAR =======
#
# ============================================================

can_tmp = CountAtomNum(atomO, atomB, atomA, atomC, n, m)
atom_layers = [rot_layer1_sub_a, rot_layer1_sub_b, rot_layer1_sub_c, 
               rot_layer2_sub_a, rot_layer2_sub_b, rot_layer2_sub_c]
h_values = [h1, h2, h3, h4, h5, h6]
stick_points = [True, None, None, True, None, None]

can = [can_tmp.count_atom_num(layer, h, stick_point=sp) 
           for layer, h, sp in zip(atom_layers, h_values, stick_points)]

# 計算 total_num
total_num = sum(len(can[i]) for i in range(len(can)))
#can = [total_num] + results

params = { # 定義一個字典，包含所有參數
    'z_h': 25,
    'material_name': 'TBSc2C',
    'angle': angle,
    'stack_conf': 'AA', #'AA', 'AB', ...
    'suffix': '',
    'atom_types': ['C','Sc'],
    'total_num': total_num,
    'can': can,
    'sub_index': [[0,3],[1,2,4,5]]
}

POSCAR_generator(atomO,atomA,atomB,atomC, **params)