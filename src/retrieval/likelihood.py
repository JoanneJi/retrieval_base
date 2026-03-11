import numpy as np
from scipy.special import loggamma

class Covariance:
     
    def __init__(self, err): 
        self.err = err
        self.cov_reset() # set up covariance matrix
        
    def cov_reset(self): # make diagonal covariance matrix from uncertainties
        self.cov = self.err**2

    def get_logdet(self): # log of determinant
        self.logdet = np.sum(np.log(self.cov)) 
        return self.logdet

    def solve(self, b): # Solve: cov*x = b, for x (x = cov^{-1}*b)
        return 1/self.cov * b # if diagonal matrix, only invert the diagonal

class LogLikelihood:

    def __init__(self, target=None, covariance=None, retrieval_object=None, scale_flux=True, scale_err=True, alpha=2, N_phi=1):
        """
        Initialize LogLikelihood.
        
        Can be initialized either with:
        - target and covariance (new interface)
        - retrieval_object (old interface, for backward compatibility)
        """
        # Support both old and new interfaces
        if retrieval_object is not None:
            # Old interface: retrieval_object
            self.target = None
            self.d_flux = retrieval_object.data_flux
            self.d_mask = retrieval_object.mask_isfinite
            self.N_params = retrieval_object.n_params
            self.cov = covariance if covariance is not None else Covariance(err=retrieval_object.data_err[self.d_mask])
        elif target is not None:
            # New interface: target and covariance
            self.d_flux = target.fl_flat
            self.d_mask = target.mask
            self.N_params = None  # Will be set later if needed
            self.cov = covariance
            self.target = target
        else:
            raise ValueError("Must provide either (target, covariance) or retrieval_object")

        self.chips_mode = getattr(self.target, 'chips_mode', False) if self.target is not None else False
        if self.chips_mode and self.target is not None:
            # Chip boundaries: flat index start and length for each chip
            self._chip_lengths = [len(w) for w in self.target.wl]
            self._chip_starts = np.cumsum([0] + self._chip_lengths)[:-1]
            self._n_chips = len(self._chip_lengths)
        else:
            self._chip_starts = None
            self._chip_lengths = None
            self._n_chips = None

        self.scale_flux = scale_flux
        self.scale_err = scale_err
        self.N_d = self.d_mask.sum()  # number of degrees of freedom / valid datapoints
        self.alpha = alpha  # from Ruffio+2019
        self.N_phi = N_phi  # number of linear scaling parameters
        
    def __call__(self, m_flux, Cov=None):
        """
        Calculate log-likelihood.

        When target is in chips_mode, one flux-scaling factor (phi) is solved per chip
        to reduce detector-to-detector offsets; otherwise a single global phi is used.
        """
        if Cov is None:
            Cov = self.cov

        self.ln_L = 0.0
        self.chi2_0 = 0.0
        self.m_flux_phi = np.nan * np.ones_like(self.d_flux)  # scaled model flux

        if self.chips_mode:
            # Per-chip phi: one scaling factor per chip (same order as target.wl)
            self.phi = []
            N_d_total = 0
            for i in range(self._n_chips):
                start = self._chip_starts[i]
                length = self._chip_lengths[i]
                end = start + length
                d_flux_i = self.d_flux[start:end]
                m_flux_i = m_flux[start:end]
                mask_i = self.d_mask[start:end]
                N_d_i = int(mask_i.sum())
                if N_d_i == 0:
                    self.phi.append(1.0)
                    self.m_flux_phi[start:end] = m_flux_i
                    continue
                d_i = d_flux_i[mask_i]
                m_i = m_flux_i[mask_i]
                err_i = self.target.err_flat[start:end][mask_i]
                Cov_i = Covariance(err=err_i)
                _, phi_i = self.get_flux_scaling(d_i, m_i, Cov_i)
                self.phi.append(phi_i)
                self.m_flux_phi[start:end] = phi_i * m_flux_i
                residuals_i = d_flux_i - (phi_i * m_flux_i)
                chi2_i = np.dot(residuals_i[mask_i].T, Cov_i.solve(residuals_i[mask_i]))
                self.chi2_0 += chi2_i
                N_d_total += N_d_i
                logdet_cov_0_i = Cov_i.get_logdet()
                MT_inv_cov_0_M = np.dot(m_i.T, Cov_i.solve(m_i))
                logdet_MT_inv_cov_0_M = np.log(MT_inv_cov_0_M)
                self.ln_L += -0.5 * (N_d_i - self.N_phi) * np.log(2 * np.pi) + loggamma(
                    0.5 * (N_d_i - self.N_phi + self.alpha - 1)
                )
                self.ln_L += -0.5 * (
                    logdet_cov_0_i + logdet_MT_inv_cov_0_M
                    + (N_d_i - self.N_phi + self.alpha - 1) * np.log(chi2_i)
                )
            self.N_d = N_d_total
            self.chi2_0_red = self.chi2_0 / self.N_d if self.N_d > 0 else np.nan
            if self.scale_err and self.N_d > 0:
                self.s2 = self.get_err_scaling(self.chi2_0, self.N_d)
            else:
                self.s2 = 1.0
            return self.ln_L

        # Single global phi (original behaviour)
        N_d = self.d_mask.sum()
        d_flux = self.d_flux[self.d_mask]
        m_flux_masked = m_flux[self.d_mask]
        if self.scale_flux:
            self.m_flux_phi[self.d_mask], self.phi = self.get_flux_scaling(d_flux, m_flux_masked, Cov)
        residuals_phi = (self.d_flux - self.m_flux_phi)
        chi2_0 = np.dot(residuals_phi[self.d_mask].T, Cov.solve(residuals_phi[self.d_mask]))
        logdet_MT_inv_cov_0_M = 0
        if self.scale_flux:
            MT_inv_cov_0_M = np.dot(m_flux_masked.T, Cov.solve(m_flux_masked))
            logdet_MT_inv_cov_0_M = np.log(MT_inv_cov_0_M)
        if self.scale_err:
            self.s2 = self.get_err_scaling(chi2_0, N_d)
        logdet_cov_0 = Cov.get_logdet()
        self.ln_L += -0.5 * (N_d - self.N_phi) * np.log(2 * np.pi) + loggamma(
            0.5 * (N_d - self.N_phi + self.alpha - 1)
        )
        self.ln_L += -0.5 * (
            logdet_cov_0 + logdet_MT_inv_cov_0_M
            + (N_d - self.N_phi + self.alpha - 1) * np.log(chi2_0)
        )
        self.chi2_0 = chi2_0
        self.chi2_0_red = self.chi2_0 / self.N_d
        return self.ln_L

    def get_flux_scaling(self, d_flux, m_flux, cov): 
        # Solve for linear scaling parameter phi: (M^T * cov^-1 * M) * phi = M^T * cov^-1 * d
        lhs = np.dot(m_flux.T, cov.solve(m_flux)) # Left-hand side
        rhs = np.dot(m_flux.T, cov.solve(d_flux)) # Right-hand side
        phi = rhs / lhs # Optimal linear scaling factor
        return np.dot(m_flux, phi), phi # Return scaled model flux + scaling factors

    def get_err_scaling(self, chi_squared_scaled, N):
        s2 = np.sqrt(1/N * chi_squared_scaled)
        return s2 # uncertainty scaling that maximizes log-likelihood
