import numpy as np
import matplotlib.pyplot as plt 
from scipy.interpolate import griddata

# configurations : VV i.e. same polarisation for incoming and outgoing light 
# configurations : HV i.e. same polarisation for incoming and outgoing light 
config = 'VV' 
offset_angle = np.pi *0.5
dir_propogation = '001' # To be used while saving file
E_pol_plane_axes = [0,1] # 0,1,2 for x, y, z. 
raman_tensor_file = 'raman_tensor.npy'
phonon_energies_file = 'phonon_energies.npy'
raman_int_out_file=f'summed_intensity_{config}.dat'
dump_int_summed_over_pol_angles=True

# Choose energy window for phonons
start_e = 25 # in cm-1  
end_e = 200
angle_max = np.pi   # in rad
angle_max_degrees = angle_max * 180.0 * (1.0/np.pi) # in degrees

angle_grid = np.linspace(0,angle_max,400)
e_pad = 25 # To account for broadening from phonons outside the window
gamma = 5 # Broadening in cm-1 
energy_grid = np.linspace(start_e,end_e,400) # in cm-1 


#####################################

def gauss(energy,strength,gamma,energy_grid):
    lineshape = strength * np.exp(-0.5*((energy_grid-energy)**2/gamma**2))
    return(lineshape)

raman_tensor = np.load(raman_tensor_file,allow_pickle=True)
phonon_energies = np.load(phonon_energies_file,allow_pickle=True)  * 1.05

# Use the subset of phonons belonging to energy window
subset = np.argwhere(np.logical_and(phonon_energies > start_e - e_pad, phonon_energies < end_e + e_pad ))
subset = subset[:,0]

raman_tensor = raman_tensor[subset,...]
phonon_energies = phonon_energies[subset]

##########

e_field_in = np.zeros((len(angle_grid),3))
e_field_out = np.zeros((len(angle_grid),3))

e_field_in[:,E_pol_plane_axes[0]] = np.sin(angle_grid+offset_angle)
e_field_in[:,E_pol_plane_axes[1]] = np.cos(angle_grid+offset_angle)

if config == 'VV':
    # VV configuration 
    # Incoming and outgoing light have the same polarization 
    e_field_out[:,E_pol_plane_axes[0]] = np.sin(angle_grid+offset_angle) # VV configuration 
    e_field_out[:,E_pol_plane_axes[1]] = np.cos(angle_grid+offset_angle)
else:
    e_field_out[:,E_pol_plane_axes[0]] = np.sin(angle_grid+offset_angle+np.pi*0.5) # HV configuration
    e_field_out[:,E_pol_plane_axes[1]] = np.cos(angle_grid+offset_angle+np.pi*0.5)

raman_int = np.abs(np.einsum('tm,pmn,tn -> pt',e_field_in,raman_tensor,e_field_out))**2


intensity = np.zeros((len(energy_grid),len(angle_grid)))
for a in range(len(angle_grid)):
    for p, energy in enumerate(phonon_energies):
        intensity[:,a] += gauss(energy,raman_int[p,a],gamma,energy_grid)


if os.path.exists('./heatmaps'):
    pass
else:
    os.system("mkdir heatmaps")

fig, ax = plt.subplots(figsize=(1.5,1.5))


#x_ticks= np.linspace(0,len(energy_grid),5)
#x_tick_labels=np.linspace(start_e,end_e,5)
# Calculate the correct indices for your desired energy values
target_energies = np.array([50, 100, 150])
x_ticks = np.searchsorted(energy_grid, target_energies)
x_tick_labels = ['50', '100', '150']

plt.xticks(ticks=x_ticks,labels=x_tick_labels)


target_angles = np.array([0, 90, 180])
y_ticks = np.searchsorted(angle_grid * (180.0/np.pi), target_angles)
y_tick_labels = ['0', '90', '180']

#y_ticks= np.linspace(0,len(angle_grid),3)
#y_tick_labels=np.linspace(0,angle_max_degrees,3)
plt.yticks(ticks=y_ticks,labels=y_tick_labels)

plt.xlabel(r'Raman shift (cm$^{-1}$)',labelpad=1)
plt.ylabel(r'Angle ($\circ$)',labelpad=1)

plt.imshow(intensity.T,origin='lower',cmap='viridis')

heatmap_file_name = './heatmaps/heatmap_' + config + dir_propogation + '.png'
#plt.tight_layout(pad=0.3)

#plt.tight_layout()
plt.savefig(heatmap_file_name,dpi=400,bbox_inches='tight',pad_inches=0.05)