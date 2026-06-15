import numpy as np 
import netCDF4 as nc 
import os 
#import torch as to

cm_Ha = (1.0 / (8065.610420 * 27.211407953))
Ha_eV = 27.211407953

exec(open('modes.inp').read())  # reads input file (python compliant)


##############################################################

n_atoms = len(atoms)            # Number of atoms in unit cell 
n_modes = len(atoms) * 3        # Number of phonon modes 3

def ph_eigs(eigv,atoms):
    n_atoms = len(atoms)            # Number of atoms in unit cell 
    n_modes = len(atoms) * 3        # Number of phonon modes 3
    mode_file = open(eigv,'r')    # Reading the input file
    lines = mode_file.readlines()       # Storing it in lines variable
    modes = np.zeros((n_modes,n_atoms,3)) # last dim for projection along axes
    energy = np.zeros((n_modes))
    m=-1
    for l,line in enumerate(lines):
        if 'freq' in line:
            m += 1
            energy[m] = np.abs(np.float64(line.split()[7]))
            for i in range(0,n_atoms):
                line_string =lines[l+i+1].split()
                modes[m,i,0] = np.float64(line_string[1]) / np.sqrt(m_atom[i])
                modes[m,i,1] = np.float64(line_string[3]) / np.sqrt(m_atom[i])
                modes[m,i,2] = np.float64(line_string[5]) / np.sqrt(m_atom[i])
    return(modes,energy)


########### get eigenvectors ##############

modes = ph_eigs(eigv,atoms)[0]
modes_lr = ph_eigs(eigv_lr,atoms)[0]

energy = ph_eigs(eigv,atoms)[1] 
energy = energy * cm_Ha # Convert from cm-1 to Ha
energy_lr = ph_eigs(eigv_lr,atoms)[1]
energy_lr = energy_lr * cm_Ha # Convert from cm-1 to Ha

# reshape the eigenvectors as a square matrix
modes = np.reshape(modes,(n_modes,n_modes)) # mode, natom*axes (xyz xyz .. natom^th *xyz)
modes_lr = np.reshape(modes_lr,(n_modes,n_modes)) # mode, natom*axes (xyz xyz .. natom^th *xyz)

# Invert the matrix , now it can be used for conversion from phonon to cartesian basis 
conv_mat = np.linalg.inv(modes)

###########################################

# Import exciton-phonon matrix elements
elph_nc = nc.Dataset(elph_file,'r')
elph = elph_nc['exc_elph'][...].data #(nq, nbranch, sf, si, Re_Im_part)
elph = elph[...,0] + 1j*elph[...,1] 

elph_prefac_ramfun = np.sqrt(2*energy) / Ha_eV 
elph = np.einsum('qbvc,b -> qbvc',elph,elph_prefac_ramfun)

print("Einsum (1) begins")
# Convert to (atom, axis) perturbation basis
elph_cart = np.einsum('ab,qbvc -> qavc',conv_mat,elph)

print("Einsum (2) begins")
# Use phonon eig. displ. for specific polar mode to compute elph mat.
elph_lr = np.einsum('ab,qbvc -> qavc',modes_lr,elph_cart)

elph_lr_prefac_ramfun = 1.0 / ( np.sqrt(2*energy_lr) / Ha_eV )
elph_lr = np.einsum('qbvc,b -> qbvc',elph_lr,elph_lr_prefac_ramfun)


os.system(f'cp {elph_file} {elph_file_lr}') # create a new netcdf file to dump the new elph matrix

elph_lr_nc = nc.Dataset(elph_file_lr,'r+')

elph_lr_nc['exc_elph'][...,0] = np.real(elph_lr[...])
elph_lr_nc['exc_elph'][...,1] = np.imag(elph_lr[...])

elph_lr_nc.close()

###### Update phonon energies in QE elph netcdf file ########3


ph_qe = nc.Dataset(ph_file_qe, "w", format="NETCDF4")
phonons = ph_qe.createDimension("phonons", n_modes)
cmplx = ph_qe.createDimension("cmplx", 2)
atoms = ph_qe.createDimension("atoms", n_atoms)
axes = ph_qe.createDimension("axes", 3)


ph_ene = ph_qe.createVariable("PH_FREQS1","f8",("phonons"))
pol_vecs = ph_qe.createVariable("POLARIZATION_VECTORS","f8",(axes, atoms, phonons, cmplx))

ph_qe['PH_FREQS1'][...] = energy_lr**2 
ph_qe['POLARIZATION_VECTORS'][...] = 0.0

ph_qe.close()

##########################################




