#!/usr/bin/env python
#==============================================================================#
#                                                                              #
#              Band structure plotting tool (Modified version)                 #
#                         with decomposed function                             #
#                                                                              #
#==============================================================================#

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
#from scipy.spatial import distance

start_time = time.time() #denote the start time

with open('OUTCAR', "r") as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        if "reciprocal lattice vectors" in line:
            reciprocal_vectors = np.array([float(num.split()[3+i]) for num in lines[i+1:i+4] for i in range(3)]).reshape(3,3)
        if "E-f" in line:
            Ef = float(line.split()[2])
            print("=================================================================")
            print('Ef =',Ef,'eV')
            print("=================================================================")
            break
#Ef = -0.0823
# replace '\n' as '\n\t\t\t\t' in string
print('reciprocal lat vector =',str(reciprocal_vectors).replace('\n','\n\t\t\t'))
print("=================================================================")

#We have to do some task for k point for plotting band
h = open('KPOINTS','r')
num_lines = sum(1 for line in open('KPOINTS'))
line_num = 0
hsp = []       #collection of high symmetry points
hsp_coord = [] #collection of high symmetry points' coordinate
for i in h.readlines():
    line_num += 1
    if line_num == 2:
        k_num = int(i.split()[0])
        print("k_num =",k_num)
    if line_num == 5:
        hsp.append(str(i.split()[-1]))
        #hsp_coord.append([kx,ky,kz])
        hsp_coord.append([float(i.split()[0]),float(i.split()[1]),float(i.split()[2])])

    if line_num > 5 and line_num%3==0:
        hsp.append(str(i.split()[-1]))
        #hsp_coord.append([kx,ky,kz])
        hsp_coord.append([float(i.split()[0]),float(i.split()[1]),float(i.split()[2])])

hsp = np.array(hsp)
hsp_coord = np.array(hsp_coord).reshape(-1,3)
print('hsp =',hsp)                                               #names of all the high symmetry points
print('hsp_coord   =  ',str(hsp_coord).replace('\n','\n\t\t'))   #coordinates of all the high symmetry points
print("=================================================================")
#We have to organize all the eigenvalues
with open('EIGENVAL', 'r') as f:
    for i, line in enumerate(f):
        i += 1  #line number
        if i == 6:
            kpt_num = int(line.split()[1])
            eig_num = int(line.split()[2])
            print('kpt_num =',kpt_num,f" real kpoint number is {kpt_num-(len(hsp)-2)} ")
            print('eig_num =',eig_num,"number of bands")
            print("=================================================================")
            #kpt = np.zeros((kpt_num,4))        # k-point array
            eig = np.zeros(kpt_num*eig_num)    # all eigenvalues array
            j = 0 # jth element in eig for iteration

        elif i > 6:
            if (i-6) % (eig_num+2) == 2:
                pass
            elif (i-6) % (eig_num+2) == 1:
                pass
            else:
                #line.split()
                eig[j] = float(line.split()[1])-Ef  #shift the energy for value = Ef
                j += 1

eig = eig.reshape((kpt_num,eig_num))  #reshape all eigenvalues array
#eig = eig.reshape((eig_num,kpt_num)).T
end1 = time.time()
#print('read file time:',end1-start_time) #Move to #1.

# calculate the distance of each k-path section
distances = np.linalg.norm(np.diff(np.dot(hsp_coord,reciprocal_vectors), axis=0), axis=1)

# calculate the total length of k-path
total_length = np.sum(distances)

# normalized each k-path section
norm_dis = distances/total_length

# cumulative k-path section
cumulative_dis = np.cumsum(norm_dis)
cumulative_dis = np.append(np.array([0]),cumulative_dis)

k_path = np.array([])
for i in range(len(cumulative_dis)-1):
    sub_arr = np.linspace(cumulative_dis[i], cumulative_dis[i+1], k_num-1,endpoint=False)
    k_path = np.concatenate((k_path,sub_arr))
k_path = np.concatenate((k_path,np.array([1])))

duplicate_kpt = len(hsp)-2  #number of duplicate k-point
k_path_section =len(hsp)-1  #number of k-path section

for i in range(k_path_section-1,0,-1):
    #kpt = np.delete(kpt, int(kpt_num/k_path_section)*i, axis=0)
    eig = np.delete(eig, k_num*i, axis=0)

k_path = np.repeat(k_path,eig_num).reshape(-1,eig_num) # Convert a one-dimensional array to a two-dimensional array

# Calculating band gap
mask_abv_fermi = eig > 0
mask_bel_fermi = eig < 0

conduct_bnd = eig[mask_abv_fermi].min()
valance_bnd = eig[mask_bel_fermi].max()
band_gap = conduct_bnd - valance_bnd
print(f"Band gap   =   {round(band_gap,6)} eV")

indice1 = np.argwhere(eig == conduct_bnd)
indice2 = np.argwhere(eig == valance_bnd)
if indice1[0][0] == indice2[0][0] and band_gap >0:
    print("Band gap type: Direct Band gap")
elif indice1[0][0] != indice2[0][0] and band_gap >0:
    print("Band gap type: Indirect Band gap")
else:
    print("Band gap type: No Band gap")
print("=================================================================")
end2 = time.time() #denote the end time
#print('data processing time:',end2-end1) #Move to #2.

#plotting
fig, ax = plt.subplots(figsize=(10,6))

# Original method:
#for i in range(eig_num):
#    ax.plot(k_path,eig.T[i],color='blue')            #plot each band

# k_path and eig are both 2d array, we can avoid using for-loop (save time!)
# plot band structure:
ax.plot(k_path,eig,color='grey',zorder=0)
ax.plot([], label='DFT', c='grey')
#print(k_path.shape,eig.shape) # for checking

for i in range(len(hsp)):
    ax.axvline(x=cumulative_dis[i],ls=':',color='black')    #plot vertical line in each hsp

ax.set_xticks(cumulative_dis)
ax.set_xticklabels(hsp, fontsize=12)
ax.set_xlabel("K point")
ax.set_ylabel("eV")
ax.axhline(y=0,ls=':',label="Ef")                    #denote the Fermi level
#ax.set_title(" Band Structure")
ax.legend(loc ='best')

end_time = time.time() #denote the end time
print('read    file    time:',round(end1-start_time,6)) #1.
print('data processing time:',round(end2-end1,6))       #2.
print('plotting        time:',round(end_time-end2,6))
print('Total   spent   time:',round(end_time-start_time,6))
print("=================================================================")
#plt.show()
#%%
t1 = time.time() 

# Read content line by line and process one line at a time 
with open('PROCAR', 'r') as file:
    band_data = None
    ion_data = []
    line_num = 0
    k_th_pnt = 0
    bnd_th = 0
    
    
    for line in file:
        line_num += 1
        
        #print(line_num)
        if "k-point   290 :" in line:
            break  # 如果找到，則跳出循環
        line = line.strip()  # Remove newlines and spaces at the end of lines
        if line_num==2:
            info = line.split("#")
            num_kpt = int(info[1].split(":")[1])
            num_bnd = int(info[2].split(":")[1])
            num_ion = int(info[3].split(":")[1])
            kp_coord = np.zeros((num_kpt,3))        # array for k-point coordinates
            eigenval = np.zeros((num_kpt,num_bnd))  # array for eigenvalues, plot band 
        
        if line_num > 2:
            
            # define k_th_pnt index, it means this is the _th k-point
            k_th_pnt = (line_num-2) // ((5+num_ion)*num_bnd+3) + 1 
            if (line_num-2) % ((5+num_ion)*num_bnd+3) == 0:
                k_th_pnt -= 1
                
            # define bnd_th index, it means this is the _th band
            bnd_th = ((line_num-2) % ((5+num_ion)*num_bnd+3) -3) // (5+num_ion) + 1

            if ((line_num-2) % ((5+num_ion)*num_bnd+3) -3) % (5+num_ion) == 0:
                bnd_th -= 1
                
            if (line_num-2) % ((5+num_ion)*num_bnd+3) < 4:
                bnd_th = 0
                if (line_num-2) % ((5+num_ion)*num_bnd+3) == 2:
                    #print('123: ',kp_coord, kp_coord.shape)
                    #print(line.split(':')[1].split('weight')[0].split(" ")[4])
                    kp_coord[k_th_pnt-1][0] = line.split(':')[1].split('weight')[0].split(" ")[4]
                    kp_coord[k_th_pnt-1][1] = line.split(':')[1].split('weight')[0].split(" ")[5]
                    kp_coord[k_th_pnt-1][2] = line.split(':')[1].split('weight')[0].split(" ")[6]
                    #print(kp_coord[k_th_pnt-1])
                    
            if (line_num-2) % ((5+num_ion)*num_bnd+3) == 0:
                bnd_th = 8
                
                
            #print(f'k_th_pnt: {k_th_pnt}')
            #print(f'  bnd_th: {bnd_th}')
            


        if bnd_th > 0:
            if line.startswith("band"):
                eigenval[k_th_pnt-1][bnd_th-1] = float(line.split('#')[1].split('energy')[1])
                #print(eigenval[k_th_pnt-1][bnd_th-1])
            
            
            # test part, need to know the length of this list
            if line_num == 9: # 9th line is the firt line contain the weight of each orbital
                a = line
                #print(a.split('  '),len(a.split('  ')),sep='\n')
                
                num_orb = len(a.split('  '))
                weight_arr = np.zeros((num_kpt,num_bnd,num_ion,num_orb-1))
                
            if ((line_num-2)%((5+num_ion)*num_bnd+3)-3) > 0 and \
               (((line_num-2)%((5+num_ion)*num_bnd+3)-3)%(5+num_ion) > 3) and \
               (((line_num-2)%((5+num_ion)*num_bnd+3)-3)%(5+num_ion) < 4+num_ion):
                   
                # define ion_index, it means this is the _th ion
                ion_index = ((line_num-2)%((5+num_ion)*num_bnd+3)-3)%(5+num_ion) - 4
                
                b = line.split('  ')
                float_b = [float(item) for item in b]
                weight_arr[k_th_pnt-1][bnd_th-1][ion_index][:] = float_b[1:]
                #print(weight_arr[k_th_pnt-1][bnd_th-1][ion_index][:])
        
                
        '''
        # for checking
        if line_num >80:
            break
        '''    
        
t2 = time.time()
print(f'read PROCAR cost: {round(t2-t1,6)}s')
#%%

eigenval = eigenval.T # for plot the band, need to transpose first
eigenval = np.delete(eigenval, [i*k_num for i in range(1,1+duplicate_kpt)], axis=1)-Ef

# 绘制提取出的 k_path 和 eigenval 值
weight_arr = np.delete(weight_arr,[i*k_num for i in range(1,1+duplicate_kpt)], axis=0)

bnd_max = eigenval.shape[0]
select_bnd = np.array([x for x in range(0,bnd_max)])    # band index
#select_bnd = np.array([6,7,8,9,10])
#select_ion = [0,1]     # ion_1, ion_2
select_ion = np.array([x for x in range(0,6)]) 
#select_ion = np.array([2,3,4,5])
selected_orbitals = np.array([x for x in range(0,9)]) 
#selected_orbitals = np.array([0,1,2,3])  # s, py, pz, px, dxy, dyz, dz2, dxz, x2-y2
magnification = 500


t1 = time.perf_counter()

# setting energy range for plot projection band
ub = 1.5   #default: eigenval.max()
lb = -1.5  #default: eigenval.min()

ub = 3
lb = -3

# s     py     pz     px    dxy    dyz    dz2    dxz  x2-y2
'''
for i in select_bnd:
    for j in select_ion:
        row_indices = np.where((lb < eigenval[i]) & (eigenval[i] < ub))
        selected_eigenval = eigenval[i][row_indices]
        selected_k_path = k_path.T[0][row_indices]
        if selected_eigenval.size != 0:         
            ax.scatter(selected_k_path, selected_eigenval, c='r',         s=weight_arr[:,i,j,0][row_indices] * magnification, zorder=1)  # s orbital
            ax.scatter(selected_k_path, selected_eigenval, c='limegreen', s=weight_arr[:,i,j,1][row_indices] * magnification, zorder=1)  # py
            ax.scatter(selected_k_path, selected_eigenval, c='b',         s=weight_arr[:,i,j,3][row_indices] * magnification, zorder=1)  # px
            ax.scatter(selected_k_path, selected_eigenval, c='cyan',      s=weight_arr[:,i,j,2][row_indices] * magnification, zorder=1)  # pz
            ax.scatter(selected_k_path, selected_eigenval, c=cl,      s=weight_arr[:,i,j,4][row_indices] * magnification, zorder=1)  # dxy 
            ax.scatter(selected_k_path, selected_eigenval, c='red',      s=weight_arr[:,i,j,5][row_indices] * magnification, zorder=1)  # dyz
            ax.scatter(selected_k_path, selected_eigenval, c='orange',      s=weight_arr[:,i,j,6][row_indices] * magnification, zorder=1)  # dz2
            ax.scatter(selected_k_path, selected_eigenval, c='purple',      s=weight_arr[:,i,j,7][row_indices] * magnification, zorder=1)  # dxz
            ax.scatter(selected_k_path, selected_eigenval, c='gray',      s=weight_arr[:,i,j,8][row_indices] * magnification, zorder=1)  # x2-y2
'''

def plot_orbitals(ax, selected_k_path, selected_eigenval, weight_arr, row_indices, magnification, selected_orbitals):
    # 定義不同軌道對應的顏色
    # s     py     pz     px    dxy    dyz    dz2    dxz  x2-y2
    orbital_colors = ['y', 'limegreen', 'cyan', 'b',  'magenta', 'red', 'orange', 'purple', 'gray']
    
    # 根據 selected_orbitals 中的軌道索引進行繪圖
    for orbital in selected_orbitals:
        ax.scatter(
            selected_k_path,
            selected_eigenval,
            c=orbital_colors[orbital],
            s=weight_arr[:, i, j, orbital][row_indices] * magnification,
            zorder=1
        )

# 在主循環中使用 plot_orbitals 函數
for i in select_bnd:
    for j in select_ion:
        row_indices = np.where((lb < eigenval[i]) & (eigenval[i] < ub))
        selected_eigenval = eigenval[i][row_indices]
        selected_k_path = k_path.T[0][row_indices]
        
        if selected_eigenval.size != 0:
            plot_orbitals(ax, selected_k_path, selected_eigenval, weight_arr, row_indices, magnification, selected_orbitals)

           
# dummy legend   
# 定義顏色和標籤
orbital_colors_labels = {
    'y': 's',
    'limegreen': 'py',
    'b': 'px',
    'cyan': 'pz',
    'magenta': 'dxy',
    'red': 'dyz',
    'orange': 'dz2',
    'purple': 'dx2',
    'gray': 'x2-y2'
}

# 使用循環生成圖例
for color, label in orbital_colors_labels.items():
    ax.scatter(-100, -100, c=color, label=label)

ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

ax.set_title('Projected Band Structure')
ax.set_xlim(0 - 0.02, 1 + 0.02)
ax.set_ylim(lb, ub)  # 這裡的 lb 和 ub 是你的變量
#plt.ylim(-3,4)
plt.show()

t2 = time.perf_counter()
print(f'Plot projected band cost time: {t2-t1:.6f} sec')




