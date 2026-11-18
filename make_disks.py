import numpy as np
from amuse.lab import *
from amuse.ext.protodisk import ProtoPlanetaryDisk


def add_disks(stars, fraction=0.5, n_disk=500,
              r_min=5 | units.au, r_max=100 | units.au, disk_mass=0.05):
    """Give a disk to `fraction` of the stars below 2 MSun. Returns the disk particles."""

    # Several clusters (a list, or your `clusters` dict) -> do each one
    if isinstance(stars, dict):
        return {name: add_disks(s, fraction, n_disk, r_min, r_max, disk_mass)
                for name, (s, conv) in stars.items()}
    if isinstance(stars, list):
        return [add_disks(s, fraction, n_disk, r_min, r_max, disk_mass) for s in stars]

    # Randomly pick which low-mass stars get a disk
    low_mass = stars[stars.mass < 2 | units.MSun]
    n_hosts = int(fraction * len(low_mass))
    hosts = low_mass.random_sample(n_hosts)

    all_disks = Particles()
    for star in hosts:
        converter = nbody_system.nbody_to_si(star.mass, r_min)
        disk = ProtoPlanetaryDisk(n_disk, convert_nbody=converter,
                                  Rmin=1.0, Rmax=r_max / r_min,
                                  discfraction=disk_mass).result

        # Random orientation, then move the disk onto its star
        disk.rotate(np.random.uniform(0, np.pi), np.random.uniform(0, 2 * np.pi), 0.0)
        disk.position += star.position
        disk.velocity += star.velocity
        disk.host_key = star.key       # remember which star owns the disk

        all_disks.add_particles(disk)

    return all_disks