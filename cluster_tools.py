#%matplotlib inline

from amuse.lab import *
from amuse.ic.fractalcluster import new_fractal_cluster_model
from amuse.ic.kroupa import new_kroupa_mass_distribution
from amuse.community.ph4.interface import Ph4
from amuse.units import nbody_system
from matplotlib import pyplot as plt
import numpy as np


### This .py file is a conglomerate of tools that we will use for our cluster evolution! 
### you can import them as follows- 
### from cluster_tools import *instert name of function* 
### this is helpful because it allows for our main notebook
### to be much less crowded & clear in what it is doing... 


def make_cluster(n_stars=1000, r_vir=1.0 | units.pc,
                 fractal_dimension=1.6, seed=42,virial_ratio=0.5):

    # the definition and affects of fractal dimension?
    np.random.seed(seed)
    # 0.08 MSun is Hydrogen-burning limit, otherwise the stars would be brown dwarfs
    masses = new_kroupa_mass_distribution(
        n_stars, mass_min=0.08 | units.MSun, mass_max=100 | units.MSun)
    converter = nbody_system.nbody_to_si(masses.sum(), r_vir)
    stars = new_fractal_cluster_model(
        n_stars, fractal_dimension=fractal_dimension,
        convert_nbody=converter, random_seed=seed)
    stars.mass = masses

    #reset the mass-center
    stars.move_to_center()
    #scale the position and velocity to guarantee the Virial_ratio=0.5
    stars.scale_to_standard(convert_nbody=converter, virial_ratio=virial_ratio)
    return stars, converter


# Use Ph4 to study the dynamics of cluster in 2 Myr

def evolve_gravity(stars, converter, t_end=2 | units.Myr, n_snap=4):
    stars = stars.copy()                      
    gravity = Ph4(converter)
    # set soften parameter to compress the expensive computation (maybe not necessary)
    gravity.parameters.epsilon_squared = (100 | units.au)**2
    gravity.particles.add_particles(stars)
    channel = gravity.particles.new_channel_to(stars)

    E0 = gravity.kinetic_energy + gravity.potential_energy
    #make snapshots
    snaps = []
    for t in np.linspace(0, t_end.value_in(units.Myr), n_snap) | units.Myr:
        gravity.evolve_model(t)
        channel.copy()
        snaps.append((t, stars.copy()))
    E1 = gravity.kinetic_energy + gravity.potential_energy
    # check how much the system deviates from energy conservation
    print(f"relative error dE/E = {abs((E1 - E0) / E0):.1e}")
    gravity.stop()
    return snaps





### UV-source proximity and irradiation-environment diagnostics

#These analyses require `clusters` and the high-density Ph4 snapshots in `snaps` from the preceding #cells. Distances are **3D**, measured in parsecs. Disk hosts are stars below $2\,M_\odot$; sources #are stars at or above $8\,M_\odot$. The **0.1 pc** cutoff is only a comparison threshold, **not** an #evaporation boundary.

def _uv_separation_squared_pc2(stars, source_mass=8, disk_mass_limit=2):
    """Squared 3D host-to-UV-source separations, in pc^2.

    Columns are UV sources (mass >= source_mass MSun); rows are disk hosts
    (mass < disk_mass_limit MSun). Returns a plain NumPy array.
    """
    hosts = stars[stars.mass < (disk_mass_limit | units.MSun)]
    sources = stars[stars.mass >= (source_mass | units.MSun)]
    if len(hosts) == 0:
        raise ValueError("No eligible disk hosts in this snapshot")
    if len(sources) == 0:
        raise ValueError("No massive UV sources in this snapshot")

    def positions_pc(particles):
        return np.column_stack((
            particles.x.value_in(units.pc),
            particles.y.value_in(units.pc),
            particles.z.value_in(units.pc),
        ))

    host_pos = positions_pc(hosts)
    source_pos = positions_pc(sources)
    delta = host_pos[:, None, :] - source_pos[None, :, :]
    return np.sum(delta**2, axis=2)




# Define diagnostics explicitly so they do not depend on undocumented
# helper functions inside cluster_tools.
def _xyz_pc(particles):
    return np.column_stack((
        particles.x.value_in(units.pc),
        particles.y.value_in(units.pc),
        particles.z.value_in(units.pc),
    ))

def _disk_host_source_distance_squared(stars, disk_mass_limit=2,
                                       source_mass_limit=8):
    hosts = stars[stars.mass < (disk_mass_limit | units.MSun)]
    sources = stars[stars.mass >= (source_mass_limit | units.MSun)]
    if len(sources) == 0:
        raise ValueError("No stars above the UV-source mass threshold")
    if len(hosts) == 0:
        raise ValueError("No stars below the disk-host mass threshold")
    delta = _xyz_pc(hosts)[:, None, :] - _xyz_pc(sources)[None, :, :]
    return np.sum(delta**2, axis=2)  # pc^2, numerical values

def nearest_uv_distance(stars):
    return np.sqrt(_disk_host_source_distance_squared(stars).min(axis=1))

def exposure_proxy(stars):
    distance_sq_pc2 = _disk_host_source_distance_squared(stars)
    return np.sum(1.0 / np.maximum(distance_sq_pc2, 1e-12), axis=1)