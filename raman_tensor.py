import numpy as np 
import os 

# Eigenvector files, output of dynmat.x
eigv = 'eigd.dat' #  eigen-displacements of modes
read_atoms_scf_out=True
scf_out = 'scf.out'
dyn_file = 'iba_pbbr3.dynG'          
output_file_suffix = ''
######## Some constants #######

cm_Ha = (1.0 / (8065.610420 * 27.211407953))
Ha_eV = 27.211407953

###########################################
def read_atoms(scf_out):
    file_scf_out = open(scf_out,'r')
    lines = file_scf_out.readlines()
    atoms=[]
    for l, line in enumerate(lines):
        #print(line.split())
        if "number of atoms" in line:
            n_atoms = int(line.split()[4])
        if "site n.     atom                  positions (alat units)" in line:
            for n in range(n_atoms):
                atoms.append(lines[l+1+n].split()[1])
    print("Number of atoms in unit cell: ",n_atoms)
    return(atoms)
#########################################

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
                modes[m,i,0] = np.float64(line_string[1]) 
                modes[m,i,1] = np.float64(line_string[3]) 
                modes[m,i,2] = np.float64(line_string[5])
    return(modes,energy)

#############################################

def raman_cart(dyn_file,n_atoms):
    raman_cart = np.zeros((n_atoms,3,3,3))
    # 3, 3, 3 -> atom movement in xyz, Efield incoming, same outgoing
    file_text = open(dyn_file,'r')
    lines = file_text.readlines()
    for l,line in enumerate(lines):
        if "Raman tensor" in line:
            for n in range(n_atoms):
                print("Reading Raman tensor for atom ",n, ' of ', n_atoms)
                # Perterbation polarsation 1
                raman_cart[n,0,0,:] = np.array(lines[l+3+12*n].split(),dtype=np.float64)
                raman_cart[n,0,1,:] = np.array(lines[l+4+12*n].split(),dtype=np.float64)
                raman_cart[n,0,2,:] = np.array(lines[l+5+12*n].split(),dtype=np.float64)
                # Perterbation polarsation 2
                raman_cart[n,1,0,:] = np.array(lines[l+7+12*n].split(),dtype=np.float64)
                raman_cart[n,1,1,:] = np.array(lines[l+8+12*n].split(),dtype=np.float64)
                raman_cart[n,1,2,:] = np.array(lines[l+9+12*n].split(),dtype=np.float64)
                # Perterbation polarsation 3
                raman_cart[n,2,0,:] = np.array(lines[l+11+12*n].split(),dtype=np.float64)
                raman_cart[n,2,1,:] = np.array(lines[l+12+12*n].split(),dtype=np.float64)
                raman_cart[n,2,2,:] = np.array(lines[l+13+12*n].split(),dtype=np.float64)
    return(raman_cart)


############################################

# Atomic symbols of all atoms in unit cell
print("Reading atoms from scf output")
if read_atoms_scf_out==True:
    # Read atoms from scf.out file
    atoms = read_atoms(scf_out)
else:
    # Or specify here
    atoms = ['Bi','Fe','Fe','Bi','O','O','O','O','O','O']

n_atoms = len(atoms)            # Number of atoms in unit cell 
n_modes = len(atoms) * 3        # Number of phonon modes 3


print("Reading Raman tensor in atom-perturbation basis from dynG file")
raman_cart = raman_cart(dyn_file,n_atoms)
raman_cart = raman_cart.astype(np.float64)
#print(raman_cart[0:2,...])

########### get eigenvectors ##############

print("Reading eigendisplacements calculated from dynmat.x output")
modes = ph_eigs(eigv,atoms)[0]
energy = ph_eigs(eigv,atoms)[1] 
#energy = energy * cm_Ha # Convert from cm-1 to Ha


########### Compute Raman tensor in phonon basis ########

print("Computing Raman tensor in phonon basis")
raman_phon = np.einsum('pax, axmn -> pmn',modes,raman_cart)

# Remove numerical noise
raman_phon = np.where(np.abs(raman_phon) < 10**(-10), 0.0, raman_phon)

print("Dumping Raman tensor and phonon energies")
np.save('raman_tensor'+output_file_suffix+'.npy',raman_phon,allow_pickle=True)
np.save('phonon_energies'+output_file_suffix+'.npy',energy,allow_pickle=True)

