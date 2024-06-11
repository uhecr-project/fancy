'''Class that handles forward simulations of GMF deflections (lensing / weighted vMF maps)'''

import os
import numpy as np
import h5py
import pickle as pickle

from astropy.coordinates import SkyCoord

from fancy.utils.package_data import (
    get_path_to_lens,
)

try:
    import crpropa
except:
    crpropa = None

class GMFLensing:

    '''
    Class that handles forward simulations of GMF deflections (lensing / weighted vMF maps)
    '''

    lens_names = {
        "JF12":"JF12full_Gamale"
    }

    def __init__(self, gmf_model : str = "JF12"):

        '''
        Class that handles forward simulations of GMF deflections (lensing / weighted vMF maps)

        :param gmf_model: GMF model considered
        '''

        if crpropa == None:
            raise ImportError("CRPropa must be installed to use this functionality.")

        self.gmf_model = gmf_model
        path_to_lens = str(get_path_to_lens(self.lens_names[self.gmf_model]))

        # read in GMF lens if we have GMF enabled
        if gmf_model == "JF12":
            self.gmf_lens = crpropa.MagneticLens(path_to_lens)
        else:
            raise NotImplementedError(f"Lensing for GMF model {gmf_model} not yet implemented.")
        
    def apply_lens_with_particles(self, rigidities, coordinates : SkyCoord, disable_gmf=False):
        '''
        Apply GMF lensing by sampling & re-sampling. Returns same number of sampled events at earth as SkyCoord objects

        :param rigidities: rigidities from particle samples in EV
        :param coordinates: arrival directions of samples in SkyCoord
        :param disable_gmf: force no GMF lensing, which will resample from the initial map
        '''
        # now GMF lensing
        particle_map = crpropa.ParticleMapsContainer()
        Nsamples = coordinates.shape[0]

        for i in range(Nsamples):
            # make coordinate system consistent
            coord_gb_xyz = -1 * coordinates[i].cartesian.xyz.value
            vector3d_gb = crpropa.Vector3d(*coord_gb_xyz)
            
            # adding rigidities instead of energy
            particle_map.addParticle(
                crpropa.nucleusId(1,1), rigidities[i] * crpropa.EeV, vector3d_gb
            )

        # lens 
        if disable_gmf:
            particle_map.applyLens(self.gmf_lens)
        
        # sample back same number of particles at earth
        _, _, glon_earth, glat_earth = particle_map.getRandomParticles(int(Nsamples))

        return SkyCoord(glon_earth, glat_earth, frame="galactic", representation_type="unitspherical")
    
    def apply_lens_to_map(self, weighted_map, R : float, disable_gmf=False):
        '''
        Apply GMF lensing from weighted healpy map

        :param weighted_map: map of normalised counts that represent an event distribution at each coordinate. **must be of Pixelisation order 6 (NPIX = 49152) following CRPropa conventions**
        :param R: rigidity in EV
        :param disable_gmf: force no GMF lensing, which will return the same map
        '''

        # make sure dimensionality is of order 6
        if len(weighted_map) != 49152:
            raise ValueError(f"Dimension of weighted map must be of order 6 (NPIX = {49152})!")

        # also make sure weighted map is normalised
        assert np.sum(weighted_map) < 1.01 and np.sum(weighted_map) > 0.99, f"sum of weights = {np.sum(weighted_map)} != 1"

        # create maps container and add weights to it
        particles = crpropa.ParticleMapsContainer()
        particles.addWeights(R * crpropa.EeV, weighted_map)

        # apply lensing
        if disable_gmf:
            particles.applyLens(R * crpropa.EeV, self.gmf_lens)

        # obtain the lensed weights
        lensed_weighted_map = particles.getWeights(crpropa.nucleusId(1, 1), R * crpropa.EeV)

        assert np.sum(lensed_weighted_map) < 1.01 and np.sum(lensed_weighted_map) > 0.99, f"sum of weights = {np.sum(lensed_weighted_map)} != 1"

        return lensed_weighted_map