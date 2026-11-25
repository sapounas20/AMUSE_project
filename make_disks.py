import numpy as np
from amuse.lab import *
from amuse.ext.protodisk import ProtoPlanetaryDisk


# Initialize a population of protoplanetary disks around low-mass stars.
#
# - Select stars with masses below 2 solar masses (Msun).
# - Randomly assign disks to a fraction of these stars (default: 50%).
# - Set each disk's initial mass to 5% of its host star's mass.
# - Set the initial disk radius to 100 AU.
# - Store the host star's identity to track its disk during evolution.
#
# Each disk is represented by its mass and radius, rather than
# individual gas particles, to reduce computational cost.

def initialize_disk_population(stars, fraction=0.5, disk_mass_fraction=0.05,
                               radius_au=100.0, seed=123,
                               host_mass_limit=2.0):
    """Select low-mass stars and attach a scalar disk mass/radius to each.

    Host indices refer to the original AMUSE particle order and must be
    preserved in the Ph4 snapshots (evolve_gravity uses a channel into a
    fixed-order copy, so that holds for the supplied cluster_tools.py).
    Reseed identically for each Q to compare the same host *indices*.
    """
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be in [0,1]")

    masses = np.asarray(stars.mass.value_in(units.MSun), dtype=float)
    eligible = np.flatnonzero(masses < host_mass_limit)
    if len(eligible) == 0:
        raise ValueError("No stars below host_mass_limit")
    n_hosts = round(fraction * len(eligible))
    if n_hosts == 0:
        raise ValueError("Fraction selects no disks: increase fraction")

    rng = np.random.default_rng(seed)
    indices = rng.choice(eligible, size=n_hosts, replace=False)
    initial_mass = disk_mass_fraction * masses[indices]
    return {
        "host_indices": indices,
        "host_keys": np.asarray(stars.key)[indices].copy(),
        "host_masses_msun": masses[indices].copy(),
        "initial_mass_msun": initial_mass,
        "initial_radius_au": np.full(n_hosts, float(radius_au)),
        "n_stars": len(stars),
    }





# Calculate the relative UV radiation received by each disk.
#
# - Identify massive stars (M >= 8 Msun) as UV radiation sources.
# - Calculate the 3D distance between each disk host and UV source.
# - Estimate radiation exposure using the inverse-square law:
#       I = sum[(M_uv / 8 Msun)^3.5 / d^2]
#   where d is measured in parsecs.
# - More massive stars contribute more strongly to the exposure.
# - Nearby UV sources have a greater effect than distant ones.
# - Apply a minimum separation to avoid numerical divergence.
#
# This is a dimensionless, mass-weighted UV exposure proxy,
# not a physically calibrated FUV flux.


def relative_uv_intensity(stars, host_indices, uv_mass_limit=8.0,
                          luminosity_exponent=3.5,
                          minimum_separation_pc=0.001):
    """Dimensionless mass-weighted inverse-square UV *proxy*, NOT FUV flux.

    I_i = sum_j (M_j/8 Msun)**alpha * (1 pc/d_ij)**2,
    over source stars with M_j >= 8 Msun. The power-law is illustrative.
    """
    masses = np.asarray(stars.mass.value_in(units.MSun), dtype=float)
    source_indices = np.flatnonzero(masses >= uv_mass_limit)
    if len(source_indices) == 0:
        return np.zeros(len(host_indices), dtype=float)

    coords = np.column_stack((
        stars.x.value_in(units.pc),
        stars.y.value_in(units.pc),
        stars.z.value_in(units.pc),
    ))
    host_pos = coords[host_indices]
    source_pos = coords[source_indices]
    d2_pc2 = np.sum((host_pos[:, None, :] - source_pos[None, :, :])**2,
                    axis=2)
    d2_pc2 = np.maximum(d2_pc2, minimum_separation_pc**2)

    weights = (masses[source_indices] / 8.0) ** luminosity_exponent
    return np.sum(weights[None, :] / d2_pc2, axis=1)

# Simulate the gradual evaporation of protoplanetary disks.
#
# - Use snapshots from the Ph4 gravitational simulation.
# - Track each disk as its host star moves through the cluster.
# - Calculate the UV exposure received at each snapshot.
# - Reduce disk mass according to the approximate mass-loss law:
#       dM/dt = -(M / tau) * sqrt(I / I_ref)
#   where:
#       M     = current disk mass
#       tau   = characteristic erosion timescale
#       I     = relative UV exposure
#       I_ref = reference exposure
# - Estimate mass loss between consecutive snapshots using
#   the average exposure-dependent erosion rate.
# - Update the disk radius as mass decreases:
#       R = R_initial * (M / M_initial)^0.5
# - Prevent disk masses from becoming negative.
#
# Returns the evolution of disk masses, radii, remaining
# mass fractions, and UV exposure over time.
#
# NOTE: This is a simplified photoevaporation model.
# Physical mass-loss prescriptions and viscous disk evolution
# (e.g. using FRIED and VADER) are not yet implemented.


def evolve_disk_population(snapshots, disks, reference_intensity=100.0,
                           erosion_timescale_myr=1.0, radius_exponent=0.5):
    """Evolve scalar disk masses using an explicitly *toy* erosion law.

    dM/dt = -M/tau * sqrt(I / I_ref).
    Between snapshots, a trapezoidal average of sqrt(I/I_ref) is used.
    This has a positive analytic solution; no disk can acquire negative mass.
    Radii vary as R/R0=(M/M0)**radius_exponent (heuristic only).
    """
    if reference_intensity <= 0 or erosion_timescale_myr <= 0:
        raise ValueError("Reference intensity and timescale must be positive")
    if radius_exponent < 0:
        raise ValueError("radius_exponent must be non-negative")
    if len(snapshots) == 0:
        raise ValueError("No snapshots were supplied")

    indices = disks["host_indices"]
    masses0 = disks["initial_mass_msun"]
    radii0 = disks["initial_radius_au"]
    times = np.asarray([t.value_in(units.Myr) for t, _ in snapshots],
                       dtype=float)
    if len(times) > 1 and np.any(np.diff(times) <= 0):
        raise ValueError("Snapshot times must be strictly increasing")

    intensities = []
    for _, stars in snapshots:
        if len(stars) != disks["n_stars"] or not np.array_equal(
                np.asarray(stars.key)[indices], disks["host_keys"]):
            raise ValueError("Snapshot star order/keys changed: host mapping unsafe")
        intensities.append(relative_uv_intensity(stars, indices))
    intensities = np.asarray(intensities)

    # The integral of 1/tau_eff determines the remaining mass fraction.
    rate = np.sqrt(intensities / reference_intensity) / erosion_timescale_myr
    cumulative = np.zeros(len(indices), dtype=float)
    mass_history = [masses0.copy()]
    for k in range(1, len(times)):
        dt = times[k] - times[k-1]
        cumulative += dt * (rate[k-1] + rate[k]) / 2.0
        mass_history.append(masses0 * np.exp(-cumulative))

    mass_history = np.asarray(mass_history)
    remaining = mass_history / masses0[None, :]
    radius_history = radii0[None, :] * remaining**radius_exponent

    return {
        "times_myr": times,
        "mass_msun": mass_history,
        "radius_au": radius_history,
        "remaining_fraction": remaining,
        "relative_uv_intensity": intensities,
        "host_indices": indices,
        "host_keys": disks["host_keys"],
        "initial_mass_msun": masses0,
    }







### Max's Add_Disk Function that is Not Used with Adonis's Code- 

# Generate particle-based protoplanetary disks using AMUSE.
#
# - Select a fraction of stars with M < 2 Msun.
# - Create a disk of gas particles around each selected star.
# - Assign disk mass and inner/outer radii.
# - Randomize disk orientation.
# - Translate disk positions and velocities to match the host star.
# - Store the host star's key for disk identification.
#
# Unlike the simplified disk-population model, this function
# creates individual particles representing the disk structure.
# It does not evolve the disks or calculate photoevaporation.

def add_disks(stars, fraction=0.5, n_disk=500,
              r_min=5 | units.au, r_max=100 | units.au, disk_mass=0.05):
    """Give a disk to `fraction` of the stars below 2 MSun. Returns the disk particles."""

    # Several clusters (a list, or your `clusters` dict) -> do each one
    if isinstance(stars, dict):
        return {name: add_disks(s, fraction, n_disk, r_min, r_max, disk_mass)
                for name, (s, conv) in stars.items()}
    # one cluster (but we dont use it) 
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

