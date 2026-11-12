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
                 fractal_dimension=1.6, seed=42):

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
    stars.scale_to_standard(convert_nbody=converter, virial_ratio=0.5)
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


def exposure_proxy(stars, source_mass=8, disk_mass_limit=2):
    """Dimensionless sum of (1 pc / d)^2 over all massive UV sources.

    This is a geometric *proxy*, assuming identical UV luminosities for
    all massive stars; it is NOT the physical FUV flux or a mass-loss rate.
    """
    d2 = _uv_separation_squared_pc2(stars, source_mass, disk_mass_limit)
    # The numerical floor prevents division by zero if two positions coincide.
    return np.sum(1.0 / np.maximum(d2, 1e-12), axis=1)


#def nearest_uv_distance(stars, source_mass=8, disk_mass_limit=2):
    """Distance to the nearest massive star for each disk host, in pc."""
    #d2 = _uv_separation_squared_pc2(stars, source_mass, disk_mass_limit)
    #return np.sqrt(d2.min(axis=1))

def nearest_uv_distance(stars, source_mass=8):
    """3D distance to nearest massive star, for each disk host, in pc."""

    hosts = stars[stars.mass < (2 | units.MSun)]
    sources = stars[stars.mass >= (source_mass | units.MSun)]

    if len(sources) == 0:
        raise ValueError("No massive UV sources in this cluster")

    def positions(particles):
        return np.column_stack([
            particles.x.value_in(units.pc),
            particles.y.value_in(units.pc),
            particles.z.value_in(units.pc)
        ])

    host_pos = positions(hosts)
    source_pos = positions(sources)

    separation = np.linalg.norm(
        host_pos[:, None, :] - source_pos[None, :, :],
        axis=2
    )

    return separation.min(axis=1)