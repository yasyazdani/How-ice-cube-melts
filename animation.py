import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

T_B = 20
C_B = 0
ww = 60
vec_scale_factor = 100  # maximum velocity divided by this value gives the velocity represented by the diagonal unit length
vec_interval = 10  # The interval between velocity vectors
vec_color = 'black'
f_min = 0.35
f_max = 0.55  # The range where the water ice interface is identified
f_color = 'white'
f_size = 10



snapshot_file = 'snapshots_08-08-2024_w60rmp_T20_C0_HR/snapshots_08-08-2024_w60rmp_T20_C0_HR_s1.h5'
animation_name = '08-08-2024_w60rmp_T20_C0_HR'

# Create directory for PNG files
output_dir = f'animation_frames_T{T_B}_C{C_B}_w{ww}'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with h5py.File(snapshot_file, mode='r') as file:
    Temp = file['tasks']['Temperature'][:]    
    time_list = file['scales']['sim_time'][:]
    Temp_list = Temp[:]
    Temp_t = []
    for matrix in Temp_list:
        matrix_array = np.array(matrix)
        matrix_t_array = matrix_array.T
        matrix_t = (matrix_t_array * T_B).tolist()
        Temp_t.append(matrix_t[:])


# Assuming Temp is a 3D NumPy array with shape (T, X, Z)
t_dim, x_dim, z_dim = Temp.shape

# Create arrays representing the coordinates
x = np.arange(x_dim)
z = np.arange(z_dim)

# Convert arrays to lists
Temp_list = Temp[:]
x_list = x[:]
z_list = z[:]

# Subsample the coordinates
x_v_list = x[::vec_interval]
z_v_list = z[::vec_interval]

# Create meshgrids
X_list, Z_list = np.meshgrid(x_list, z_list)
X_V_list, Z_V_list = np.meshgrid(x_v_list, z_v_list)


# Load the snapshots from the HDF5 file
with h5py.File(snapshot_file, mode='r') as file:
    xvel_list = file['tasks']['x velocity'][:]
    yvel_list = file['tasks']['y velocity'][:] 
    zvel_list = file['tasks']['z velocity'][:]


xvel_t = []
for matrix in xvel_list:
    matrix_array = np.array(matrix)
    matrix_t_array = matrix_array.T
    matrix_t = []
    for line in matrix_t_array:
        matrix_t.append(line[::vec_interval])
    xvel_t.append(matrix_t[::vec_interval])
xvel_array = np.array(xvel_t)

zvel_t = []
for matrix in zvel_list:
    matrix_array = np.array(matrix)
    matrix_t_array = matrix_array.T
    matrix_t = []
    for line in matrix_t_array:
        matrix_t.append(line[::vec_interval])
    zvel_t.append(matrix_t[::vec_interval])
zvel_array = np.array(zvel_t)

yvel_t = []
for matrix in yvel_list:
    matrix_array = np.array(matrix)
    matrix_t_array = matrix_array.T
    matrix_t = []
    for line in matrix_t_array:
        matrix_t.append(line[::vec_interval])
    yvel_t.append(matrix_t[::vec_interval])
yvel_array = np.array(yvel_t)


magvel = (xvel_array ** 2 + zvel_array ** 2 + yvel_array ** 2) ** .5
lw = (magvel / np.max(magvel))
magvel_max = np.max(magvel)
magvel_scale = magvel_max / vec_scale_factor


with h5py.File(snapshot_file, mode='r') as file:
     phase_list = file['tasks']['Phase'][:]

phase_t = []
for matrix in phase_list:
    matrix_array = np.array(matrix)
    matrix_t_array = matrix_array.T
    matrix_t = matrix_t_array.tolist()
    phase_t.append(matrix_t)


x_f_list = []
z_f_list = []
phase_plot_list = []
for t_id in range(len(phase_t)):
    x_f_list.append([])
    z_f_list.append([])
    for i in range(len(phase_t[t_id])):  # i is the index of z_f_list
        for j in range(len(phase_t[t_id][i])):  # j is the index of x_f_list
            if f_max > phase_t[t_id][i][j] > f_min:
                z_f_list[t_id].append(z_list[i])
                x_f_list[t_id].append(x_list[j])

color = 'jet'
cmax = (np.max(Temp_t)) * T_B
cmin = (np.min(Temp_t)) * T_B

fig, ax = plt.subplots(figsize=(10, 4))
image = ax.pcolormesh(X_list, Z_list, Temp_t[0], shading='nearest', cmap=color, vmin=0, vmax=20)
scatter = ax.scatter(x_f_list[0], z_f_list[0], color=f_color, s=f_size)
xvel_half = 0.5 * xvel_array[0]
zvel_half = 0.5 * zvel_array[0]
quiver = ax.quiver(X_V_list - xvel_half, Z_V_list - zvel_half, xvel_array[0], zvel_array[0], angles='xy', scale_units='xy', scale=magvel_scale, color=vec_color)

ax.set_xlabel('x')
ax.set_ylabel('z')
ax.set_title(f'C:{C_B}(g/kg) T:{T_B}(C) w:{ww}(rmp) latitude:10')

plt.tight_layout()

time_text = None

for frame in range(len(Temp_t)):
    for coll in ax.collections:
        coll.remove()
    image = ax.pcolormesh(X_list, Z_list, Temp_t[frame], shading='nearest', cmap=color)
    scatter = ax.scatter(x_f_list[frame], z_f_list[frame], color=f_color, s=f_size)
    u_half = 0.5 * xvel_array[frame]
    w_half = 0.5 * zvel_array[frame]
    quiver = ax.quiver(X_V_list - u_half, Z_V_list - w_half, xvel_array[frame], zvel_array[frame], angles='xy', scale_units='xy', scale=magvel_scale, color=vec_color)
    
    if time_text:
        time_text.remove()
    time_text = fig.text(0.95, 0.01, f'Time: {time_list[frame]:.2f} s', ha='right', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    
    plt.savefig(os.path.join(output_dir, f'{animation_name}_frame_{frame:04d}.png'))

plt.close()
