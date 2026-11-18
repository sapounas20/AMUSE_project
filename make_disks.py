from amuse.lab import *


# Add disks to cluster... 

def add_disks(stars): 
# Container for all disk particles

    all_disks = Particles()

    for stars in clusters:
        
        # Set disk properties relative to the host star
        # what do we want these values to be??? 
        r_min = 5.0 | units.au
        r_max = 100.0 | units.au
        disk_mass = 0.05 * stars.mass

    converter = nbody_system.nbody_to_si(star.mass, r_min)
    disk = ProtoPlanetaryDisk(500, # (500(N) particles per disk) 
                          convert_nbody=converter, Rmin=1.0, Rmax=(r_max / r_min), discfraction=0.05,).result

    # Optional: Apply a random inclination and position angle
    theta = np.random.uniform(0, np.pi)
    phi = np.random.uniform(0, 2 * np.pi)
    disk.rotate(theta, phi, 0.0)

    # Shift disk to the host star's phase-space coordinates
    disk.position += star.position
    disk.velocity += star.velocity

    all_disks.add_particles(disk)