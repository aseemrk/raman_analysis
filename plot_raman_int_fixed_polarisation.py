import numpy as np
import matplotlib.pyplot as plt 

raman_tensor_file = 'raman_tensor.npy'
phonon_energies_file = 'phonon_energies.npy'
gamma = 1.5 # Broadening in cm-1 
start_e = 0.0 # in cm-1  
end_e = 200 
energy_grid = np.linspace(start_e,end_e,1000) # in cm-1 
# Function for broadening 
def gauss(energy,strength,gamma,energy_grid):
    lineshape = strength * np.exp(-0.5*((energy_grid-energy)**2/gamma**2))
    return(lineshape)

raman_tensor = np.load(raman_tensor_file,allow_pickle=True)
phonon_energies = np.load(phonon_energies_file,allow_pickle=True)

e_field_in = np.array([1.0,0.0,0.0])
e_field_out = np.array([1.0,0.0,0.0])

raman_int = np.einsum('m,pmn,n -> p',e_field_in,raman_tensor,e_field_out)**2

intensity = 0.0
for p, energy in enumerate(phonon_energies):
    intensity += gauss(energy,raman_int[p],gamma,energy_grid)

np.savetxt('raman_intensity_ZXXZ.dat',np.column_stack((energy_grid,intensity)))

plt.plot(energy_grid,intensity)
plt.xlabel('Energy (1/cm)')
plt.ylabel('Raman intensity (arb.u.)')

plt.savefig('raman_intensity.pdf')
