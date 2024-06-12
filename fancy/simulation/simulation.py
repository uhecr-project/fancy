'''Class to manage simulations for UHECR propagation & detector effects'''

import os
import numpy as np
import pickle as pickle
import h5py
import datetime

from scipy.interpolate import RegularGridInterpolator, CubicSpline

from astropy.coordinates import SkyCoord, AltAz
import astropy.units as u
from astropy.time import Time

from fancy import Data
from fancy.physics.gmf import GMFBackPropagation, GMFLensing
from fancy.detector.exposure import m_dec
from vMF import sample_vMF
import tempfile

class Simulation:

    '''
    Handles the generation of simulation samples 
    '''

    truth_input_keys = ["f", "alpha_s", "alpha_b", "alpha", "log10_L", "Bigmf"]

    def __init__(self, data : Data, energy_loss_table_file : str, exposure_table_file : str, gmf_model : str = "None"):

        '''
        Handles the generation of simulation samples 

        :param data: Data object from fancy
        :param energy_loss_table_file: file for energy tables
        :param exposure_table_file: file where exposure tables are contained
        '''

        self.detector_type = data.detector.label
        self.mass_group = data.detector.mass_group
        self.source_type = data.source.label
        self.gmf_model = gmf_model

        # source parameters
        self.Dsrcs = data.source.distance * u.Mpc
        self.Nsrcs = data.source.N

        self.data = data
        
        # make sure files do exist
        if not os.path.exists(energy_loss_table_file):
             raise FileNotFoundError(f"Energy loss tables have not been generated. Construct them first!")
        
        if not os.path.exists(exposure_table_file):
             raise FileNotFoundError(f"Exposure tables have not been generated. Construct them first!")
        
        # initialise the grids
        self._initialise_grids(energy_loss_table_file, exposure_table_file)
        

    def _initialise_grids(self, energy_loss_table_file : str, exposure_table_file : str, dlR = 0.001):
        '''
        Initialise grids used for simulation
        
        :param energy_loss_table_file: file for energy tables
        :param exposure_table_file: file where exposure tables are contained
        :param dlR: discretisation in logarithmic bins of log10(EV), default 0.001
        '''

        with h5py.File(energy_loss_table_file, "r") as f:
            # find the relevant group
            config_label = f"{self.detector_type}_mg{self.mass_group}"
            self.alpha_grid = f[config_label]["alpha_grid"][()]
            self.distances_grid = f[config_label]["distances_grid"][()] * u.Mpc
            log10_Rgrid = f[config_label]["log10_rigidities"][()] 

            # create separate grid for sampling
            lRbins = np.arange(np.log10(self.data.detector.Rth), np.log10(self.data.detector.Rth_max), dlR)
            self.rigidity_widths = (10**lRbins[1:] - 10**lRbins[:-1])  * u.EV
            self.rigidities = 10**(0.5 * (lRbins[1:] + lRbins[:-1])) * u.EV
            
            if self.mass_group != 1:
                log10_arrspect_grid = f[config_label]["log10_arrspect_grid"][()]

                # interpolate for the rigidity grid used here
                f_log10_arrspect = CubicSpline(y=log10_arrspect_grid, x=log10_Rgrid, axis=1)
                self.log10_arrspects_grid = f_log10_arrspect(np.log10(self.rigidities.value)) * (1 / u.EV)

            else:
                # get arrival rigidities as a function of distance and source rigidity
                Rarr_grid = f[config_label]["Rarr_grid"][()] * u.EV

                # interpolate for the rigidity grid used here
                self.f_Rarr = CubicSpline(y=Rarr_grid, x=10**log10_Rgrid, axis=1)

            # also get the expected energies
            self.log10_Eexs_grid = f[config_label]["log10_Eexs_grid"][()] * u.EeV

        self.Nalphas = len(self.alpha_grid)
        self.Ndistances = len(self.distances_grid)
        self.NRs = len(self.rigidities)

        # now read the exposure table file
        with h5py.File(exposure_table_file, "r") as f:
            # find the relevant group
            config_label = f"{self.detector_type}_mg{self.mass_group}"
            self.log10_Bigmf_grid = f[config_label]["log10_Bigmf_grid"][()] * u.nG
            self.log10_wexp_src_grid = f[config_label]["log10_wexp_src_grid"][()] * u.km**2 * u.yr
            self.log10_wexp_bg_grid = f[config_label]["log10_wexp_bg_grid"][()] * u.km**2 * u.yr

        self.NBigmfs = len(self.log10_Bigmf_grid)

    def set_truths(self, input_dict : dict, truth_outfile : str):
        '''
        Set & compute simulation truths based on truths
        
        :param input_dict: dictionary containing truth of [alpha_s, alpha_b, log10_L, f, Bigmf]
        :param truth_outfile: output file for truths
        '''

        # first assert that the inputs keys match the ones the class definition
        assert np.all([k_input in self.truth_input_keys for k_input in input_dict.keys()]), "Truth inputs do not match."

        alpha_s = input_dict["alpha_s"]
        alpha_b = input_dict["alpha_b"]
        Bigmf = input_dict["Bigmf"] * u.nG
        L = 10**input_dict["log10_L"] * (u.EeV * u.yr**-1)
        f = input_dict["f"]

        # calculate the number of expected events from all sources using weighted exposure * total flux
        wexps_src = np.zeros(self.Nsrcs) * (u.km**2 * u.yr)  
        Fs_per_Ls = np.zeros(self.Nsrcs) * (u.km**-2 * u.EeV**-1)  

        for id, Dsrc in enumerate(self.Dsrcs):  
            dmax_idx = np.digitize(Dsrc, self.distances_grid, right=True)

            f_log10_wexp_src = RegularGridInterpolator((self.alpha_grid, self.log10_Bigmf_grid), self.log10_wexp_src_grid[id,...])
            log10_wexp_src = f_log10_wexp_src((alpha_s, np.log10(Bigmf.value)))
            wexps_src[id] = 10.0**log10_wexp_src * (u.km**2 * u.yr)  

            f_log10_Eexs = CubicSpline(x=self.alpha_grid, y=self.log10_Eexs_grid[dmax_idx,:])
            Eex = 10.0**f_log10_Eexs(alpha_s) * u.EeV
            Fs_per_Ls[id] = 1 / (4 * np.pi * Dsrc.to(u.km)**2) / Eex

        Nex_src = L * (np.sum(wexps_src * Fs_per_Ls))

        # calculate expected events from background using source fraction
        Nex_bg = Nex_src * ((1 - f) / f)
        Nex = Nex_src + Nex_bg
        print(f"Nex: {Nex:.3f}, Nex_src: {Nex_src:.3f}, Nex_bg: {Nex_bg:.3f}")

        # convert to integer using np.round (TODO: strictly should be Poisson, edit later)
        # store as object since we need to use this for sampling
        self.Nuhecrs_arr = np.zeros(self.Nsrcs + 1, dtype=int)  # type: ignore
        self.Nuhecrs_arr[:self.Nsrcs] = int(np.round(L * wexps_src * Fs_per_Ls))
        self.Nuhecrs_arr[self.Nsrcs] = int(np.round(Nex_bg))
        print(f"Nuhecrs: {np.sum(self.Nuhecrs_arr):d}, Nuhecrs_src: {np.sum(self.Nuhecrs_arr[:-1]):d}, Nuhecrs_bg: {self.Nuhecrs_arr[-1]:d}")
        self.Nuhecrs = np.sum(self.Nuhecrs_arr)

        # now calculate the flux
        Fs = np.sum(Fs_per_Ls) * L
        # for background flux, interpolate weighted exposure and calculate using Nex_bg
        f_log10_wexp_bg = CubicSpline(x=self.alpha_grid, y=self.log10_wexp_bg_grid)  # NB: take any index for Bigmf since no dependence on it
        F0 = Nex_bg / (10.0**f_log10_wexp_bg(alpha_b) * (u.km**2 * u.yr))
        FT = Fs + F0  # total flux
        f1 = Fs / FT  # source fraction before detection
        print(f"FT: {FT:.3f}, Fs: {Fs:.3f}, F0: {F0:.3f}, f1 = Fs / FT: {f1:.3f}")

        # create dictionaryu of truths
        self.truth_dict = {
            "alpha_s":alpha_s,
            "alpha_b":input_dict["alpha_b"],
            "f":input_dict["f"],
            "f1":f1,
            "L":10.0**input_dict["log10_L"],
            "log10_L":input_dict["log10_L"],
            "Bigmf":input_dict["Bigmf"],
            "F0":F0.value,
            "log10_F0":np.log10(F0.value),
            "Fs":Fs.value,
            "FT":FT.value,
            "Nex":Nex,
            "Nsrc":Nex_src,
            "Nbg":Nex_bg
        }

        print(f"Storing truths to {truth_outfile}")
        with open(truth_outfile, "wb") as file:
            pickle.dump(self.truth_dict, file, protocol=-1)

    def sample_events(self, sampling_factor : int=100):
        '''
        Sample rigidities / energies and arrival directions at the Galactic boundary

        :param sampling_factor: factor to multiply with number of UHECRs to simulate for sampling
        '''
        Nsamples_arr = np.full(self.Nsrcs+1, sampling_factor, dtype=int) * self.Nuhecrs_arr

        '''Sampling rigidities at GB'''
        # sample using rng.choice
        rng = np.random.default_rng()

        sampled_rigidities = []

        if self.mass_group != 1:
            # precompute background spectrum & its probability
            bg_spectrum_grid = bounded_power_law(self.rigidities.value, self.truth_dict["alpha_b"], self.data.detector.Rth, self.data.detector.Rth_max)
            probs_bg = bg_spectrum_grid * self.rigidity_widths.value /  np.sum(bg_spectrum_grid * self.rigidity_widths.value)

            for id, Nsamples in enumerate(Nsamples_arr):

                if id < self.Nsrcs:
                    didx = np.digitize(self.Dsrcs[id], self.distances_grid, right=True)   

                    # similar to the background spectrum
                    f_log10_arrspect = CubicSpline(y=self.log10_arrspects_grid[didx,:,:], x=self.alpha_grid, axis=1)
                    src_spectrum_grid = 10.0**f_log10_arrspect(self.truth_dict["alpha_s"])
                    probs_src = src_spectrum_grid * self.rigidity_widths.value / np.sum(src_spectrum_grid * self.rigidity_widths.value)

                    # sample using random.choice
                    sampled_rigidities.append(rng.choice(self.rigidities, size=Nsamples, p=probs_src))
                else:
                    sampled_rigidities.append(rng.choice(self.rigidities, size=Nsamples, p=probs_bg))

        else:  # MG1, sample via power-law distribution only bounded by minimum (TODO: make this bounded power law instead)
            src_spectrum_grid = lbounded_power_law(self.rigidities.value, self.truth_dict["alpha_s"], self.data.detector.Rth)
            probs_src = src_spectrum_grid * self.rigidity_widths.value / np.sum(src_spectrum_grid * self.rigidity_widths.value)

            bg_spectrum_grid = lbounded_power_law(self.rigidities.value, self.truth_dict["alpha_b"], self.data.detector.Rth)
            probs_bg = bg_spectrum_grid * self.rigidity_widths.value / np.sum(bg_spectrum_grid * self.rigidity_widths.value)

            for id, Nsamples in enumerate(Nsamples_arr):
                if id < self.Nsrcs:

                    # sample using random.choice
                    Rsrcs_samples = rng.choice(self.rigidities, size=Nsamples, p=probs_src)
                    # store arrival rigidities, computed via interpolation
                    sampled_rigidities.append(self.f_Rarr(Rsrcs_samples.value)[id,:] * u.EV)
                else:
                    # otherwise no losses
                    sampled_rigidities.append(rng.choice(self.energies, size=Nsamples, p=probs_bg))

        '''Sampling for arrival directions'''
        sampled_coords_gb = []

        for id, Nsamples in enumerate(Nsamples_arr):

            if id < self.Nsrcs:
                # compute kappa_igmf
                kappa_igmfs = kappa_igmf_vec(sampled_rigidities[id] * u.EV, self.truth_dict["Bigmf"], self.Dsrcs[id])

                sampled_vectors_src = np.zeros((Nsamples, 3))
                for i in range(Nsamples):
                    sampled_vectors_src[i,:] = sample_vMF(self.data.source.coord[id].cartesian.xyz.value, kappa_igmfs[i], num_samples=1)
                
                sampled_coords_gb.append(SkyCoord(sampled_vectors_src, frame='galactic', representation_type='cartesian'))
            
            else:
                sampled_vectors_bg = sample_vMF(np.array([1,0,0]), 0.0, num_samples=Nsamples)   # uniform sampling
                sampled_coords_gb.append(SkyCoord(sampled_vectors_bg, frame='galactic', representation_type='cartesian'))

        return sampled_rigidities, sampled_coords_gb

    def apply_lens(self, sampled_rigidities, sampled_coords):
        '''
        Apply GMF lens by sampling & re-sampling of particles

        :param sampled_rigidites: rigidities after sampling (list of np arrays)
        :param sampled_coords: coordinates after sampling (list of SkyCoord)
        '''

        if self.gmf_model == "None":
            print(f"GMF is disabled. Will not run this code and return the initial samples")
            return sampled_coords
        
        # initialise GMFLENs
        gmflens = GMFLensing(self.gmf_model)

        sampled_coords_earth = []
        for id, sampled_coord in enumerate(sampled_coords):
            sampled_coords_earth.append(gmflens.apply_lens_with_particles(sampled_rigidities[id], sampled_coord))

        return sampled_coords_earth


    def apply_detector_cuts(self, sampled_rigidities, sampled_coords, deltaR=None, sigma_dir=None):
        '''
        Apply detector cuts to sampled events at Earth

        :param sampled_rigidites: rigidities after sampling (list of np arrays)
        :param sampled_coords: coordinates after sampling (list of SkyCoord)
        :param delta: force rigidity uncertainty in %
        :param sigma_dir: force angular reconstruction uncertainty in deg
        '''
        self.rigidities_det = np.zeros(self.Nuhecrs)
        glons_det = np.zeros(self.Nuhecrs)
        glats_det = np.zeros(self.Nuhecrs)
        self.exposure_factors = np.zeros(self.Nuhecrs)

        # some uncertainty parameters that can be forced
        kappa_d = self.data.detector.kappa_d if sigma_dir == None else 7552 * sigma_dir**-2
        deltaR = self.data.detector.rigidity_uncertainty if deltaR == None else deltaR

        rng_det = np.random.default_rng()
        uhecr_idx = 0

        for id, Nuhecrs_per_src in enumerate(self.Nuhecrs_arr):
            if id != len(self.Nuhecrs_arr)-1:
                print(f"Current Source: {self.data.source.name[id]}")
            else:
                print("Background Case")

            count_per_src = 0
            iters = 0

            while count_per_src < Nuhecrs_per_src:
                # shuffle indices from samples
                sample_idces = np.arange(len(sampled_coords[id]))
                rng_det.shuffle(sample_idces)

                for i in sample_idces:
                    
                    # angular reconstruction uncertainty
                    coord_earth = sampled_coords[id][i]

                    # sample reconstruction uncertainty using vMF
                    reconstr_uv = sample_vMF(coord_earth.cartesian.xyz.value, kappa_d, num_samples=1)[0]
                    reconstr_uv /= np.linalg.norm(reconstr_uv)
                    reconst_coord = SkyCoord(*reconstr_uv, representation_type="cartesian", frame="galactic")
                    reconst_coord.transform_to("icrs")

                    # evaluate exposure function at that declination -> construct pdet
                    m_omega = m_dec(reconst_coord.icrs.dec.rad, self.data.detector.params)
                    pdet = m_omega / self.data.detector.exposure_max
                    accept = rng_det.choice([0, 1], p = [pdet, 1 - pdet])  # use binomial distribution to sample that UHECR

                    # rigidity reconstruction
                    rig = sampled_rigidities[id][i]
                    reconst_rig = rng_det.normal(rig, deltaR * rig)

                    # if particle is within exposure & within cuts then append
                    if (accept == 0) and (reconst_rig >= self.data.detector.Rth) and (reconst_rig <= self.data.detector.Rth_max):
                        self.rigidities_det[uhecr_idx] = reconst_rig
                        reconst_coord.transform_to("galactic")
                        reconst_coord.representation_type = "unitspherical"
                        glons_det[uhecr_idx] = reconst_coord.galactic.l.deg
                        glats_det[uhecr_idx] = reconst_coord.galactic.b.deg
                        self.exposure_factors[uhecr_idx] = m_omega * self.data.detector.alpha_T / self.data.detector.M

                        uhecr_idx += 1
                        count_per_src += 1
                    
                    # break when we have the UHECRs contributing for this particular source
                    if count_per_src >= Nuhecrs_per_src:
                        print(count_per_src)
                        break

                    iters += 1

                if iters % 100 == 0:
                    print(f"Counts / src = {count_per_src}")

            if uhecr_idx > self.Nuhecrs:
                raise Exception(f"something wrong with indexing: {uhecr_idx}, {self.Nuhecrs}")

        self.coords_det = SkyCoord(glons_det * u.deg, glats_det * u.deg, frame="galactic")

        if np.any(self.rigidities_det == 0):
            raise ValueError(f"zero value detected in rigidity computation: {np.where(self.rigidities_det == 0)}")

    def backpropagate_events(self, Nsamples_bt : int=100, njobs=4):
        '''
        Backpropagate the sampled & exposure-applied events at Earth back to the GB.

        :param Nsamples_bt: the number of samples for backpropagation simulation
        '''
        
        # first write data to temporary file such that Data can read it
        outfile = tempfile.mkstemp()[1]
        with h5py.File(outfile, "w") as f:
            data_gr = f.create_group(self.detector_type)
            data_gr.create_dataset("energy", data=self.rigidities_det * self.data.detector.meanZ)
            data_gr.create_dataset("rigidity", data=self.rigidities_det)
            data_gr.create_dataset("exposure", data=self.exposure_factors)
            data_gr.create_dataset("glon", data=self.coords_det.galactic.l.deg)
            data_gr.create_dataset("glat", data=self.coords_det.galactic.b.deg)
            data_gr.create_dataset("theta", data=np.full(self.Nuhecrs, 70))  # stub
            data_gr.create_dataset("year", data=np.full(self.Nuhecrs, self.data.detector.start_year, dtype=int))  # stub
            data_gr.create_dataset("day", data=np.ones(self.Nuhecrs, dtype=int))  # stub

        # add this to the data object
        self.data.add_uhecr(outfile, label=self.detector_type, gmf_model="None")

        # now perform GMF back propagation
        gmfbackprop = GMFBackPropagation(self.data, self.gmf_model)
        gmfbackprop.run_backpropagation(Nsamples_bt, njobs=njobs)
        gmfbackprop.compute_kappa_gmf()

        # set properties
        if self.gmf_model != "None":
            self.kappa_gmfs = gmfbackprop.kappa_gmfs
            self.thetaPs = gmfbackprop.thetaPs
            self.glons_gb = gmfbackprop.uhecr_coords_gb.galactic.l.deg
            self.glats_gb = gmfbackprop.uhecr_coords_gb.galactic.b.deg


    def save(self, outfile : str):
        '''
        Save the simulation file as a new UHECR file.

        :param outfile: the output file as an UHECR file
        '''
        # simulate for zenith angles for compatibility
        zeniths_sim, years_sim, days_sim = self._simulate_zenith_angles(self.coords_det.transform_to("icrs"))

        # compute energy & ICRS coordinates also for compatibility
        # energies computed from multiplying with mean charge from mass group
        energies_det = self.rigidities_det * self.data.detector.meanZ

        # get ra, dec, glon, glat
        glons_det, glats_det = self.coords_det.galactic.l.deg, self.coords_det.galactic.b.deg
        c_icrs = self.coords_det.transform_to("icrs")
        ras_det, decs_det = c_icrs.ra.deg, c_icrs.dec.deg


        with h5py.File(outfile, "a") as file:
            if self.detector_type in list(file.keys()):
                del file[self.detector_type]
            simulated_data = file.create_group(f"{self.detector_type}")
            simulated_data.create_dataset("day", data=days_sim)
            simulated_data.create_dataset("year", data=years_sim)
            simulated_data.create_dataset("theta", data=zeniths_sim)
            simulated_data.create_dataset("rigidity", data=self.rigidities_det)
            simulated_data.create_dataset("energy", data=energies_det)
            simulated_data.create_dataset("ra", data=ras_det)
            simulated_data.create_dataset("dec", data=decs_det)
            simulated_data.create_dataset("glat", data=glats_det)
            simulated_data.create_dataset("glon", data=glons_det)
            simulated_data.create_dataset("exposure", data=self.exposure_factors)

            gmfdefl_datas_grp = simulated_data.create_group("gmf")
            config_key = f"{self.gmf_model}_mg{self.mass_group}"
            if config_key in gmfdefl_datas_grp.keys():
                del gmfdefl_datas_grp[config_key]
            gmfdefl_datas_config_grp = gmfdefl_datas_grp.create_group(config_key)
            gmfdefl_datas_config_grp.create_dataset("kappa_gmf", data=self.kappa_gmfs)
            gmfdefl_datas_config_grp.create_dataset("thetaP", data=self.thetaPs)
            gmfdefl_datas_config_grp.create_dataset("glons_gb", data=self.glons_gb)
            gmfdefl_datas_config_grp.create_dataset("glats_gb", data=self.glats_gb)

    # convert starting period to decimal year
    def _year_fraction(self, date):
        start = datetime.date(date.year, 1, 1).toordinal()
        year_length = datetime.date(date.year+1, 1, 1).toordinal() - start
        return date.year + float(date.toordinal() - start) / year_length

    def _simulate_zenith_angles(self, c_icrs):
        '''Simulate zenith angles, using ICRS SKyCoord'''
        years = []
        days = []
        times = []
        zenith_angles = []
        stuck = []

        k = 0
        first = True
        for d in c_icrs:
            za = 99
            i = 0
            while za > self.data.detector.threshold_zenith_angle.rad:
                dt = np.random.exponential(1 / self.Nuhecrs)
                if first:
                    t = self._year_fraction(self.data.detector.period_start) + dt
                else:
                    t = times[-1] + dt
                tdy = Time(t, format="decimalyear")
                c_altaz = d.transform_to(AltAz(obstime=tdy, location=self.data.detector.location))
                za = np.pi/2 - c_altaz.alt.rad

                i += 1
                if i > 100:
                    za = self.data.detector.threshold_zenith_angle.rad
                    stuck.append(1)

            # convert decimal years to year & days
            year, year_frac = divmod(t, 1)
            day = np.round(year_frac * 365.2425) # round to nearest even value
            years.append(int(year))
            days.append(int(day))

            # append time also for the algorithm
            times.append(t)
            first = False
            zenith_angles.append(za)
            k += 1
            # print(j , za)

        if len(stuck) > 1:
            print(
                "Warning: % of zenith angles stuck is",
                len(stuck) / len(zenith_angles) * 100,
            )

        return np.array(zenith_angles), np.array(years), np.array(days)

def bounded_power_law(x, alpha, xmin, xmax):

    '''bounded power law spectrum'''
    if alpha != 1.0:
        norm = (1.0 - alpha) / (xmax**(1.0 - alpha) - xmin**(1.0 - alpha))
    else:
        norm = 1.0 / (np.log(xmax) - np.log(xmin))
    
    return norm * x**(-alpha)

def lbounded_power_law(x, alpha, xmin):
    '''Lower bounded power law spectrum'''
    return (alpha - 1.0) * xmin**(alpha - 1) * x**(-alpha)

def theta_igmf(R, Bigmf, D, lc=1):
    '''
    Deflection angle for IGMF in degrees

    :param R: rigidity in EV
    :param Bigmf: IGMF magnetic field strength in nG
    :param D: distance of the source in Mpc
    :param lc: coherence length in Mpc (default 1 Mpc)
    '''
    return (2.3 * (50 * u.EV / R) * (Bigmf / (1 * u.nG)) * np.sqrt(D / (10 * u.Mpc)) * np.sqrt(lc)) * u.deg

def kappa_igmf(R, Bigmf, D, lc=1, kappa_max = 1e6):
    '''
    Deflection parameter for IGMF. Calculated using the approximate formula (for k >> 1)

    :param R: rigidity in EV
    :param Bigmf: IGMF magnetic field strength in nG
    :param D: distance of the source in Mpc
    :param lc: coherence length in Mpc (default 1 Mpc)
    :param kappa_max: some maximum threshold value for high kappa (== super small angles)
    '''
    k = (7552 * (theta_igmf(R, Bigmf, D, lc) / (1 * u.deg))**-2).value
    return k if k < kappa_max else kappa_max

def kappa_igmf_vec(Rs, Bigmf, D, lc=1, kappa_max = 1e6):
    return np.array([kappa_igmf(R, Bigmf, D, lc, kappa_max) for R in Rs])

