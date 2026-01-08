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
            self.d_flux = retrieval_object.data_flux
            self.d_mask = retrieval_object.mask_isfinite
            self.N_params = retrieval_object.n_params
            self.cov = covariance if covariance is not None else Covariance(err=retrieval_object.data_err[self.d_mask])
        elif target is not None:
            # New interface: target and covariance
            self.d_flux = target.fl
            self.d_mask = target.mask
            self.N_params = None  # Will be set later if needed
            self.cov = covariance
        else:
            raise ValueError("Must provide either (target, covariance) or retrieval_object")
        
        self.scale_flux = scale_flux
        self.scale_err = scale_err
        self.N_d = self.d_mask.sum()  # number of degrees of freedom / valid datapoints
        self.alpha = alpha  # from Ruffio+2019
        self.N_phi = N_phi  # number of linear scaling parameters
        
    def __call__(self, m_flux, Cov=None):
        """
        Calculate log-likelihood.
        
        Args:
            m_flux: Model flux (1D array, same length as data)
            Cov: Covariance object (optional, uses self.cov if not provided)
        """
        if Cov is None:
            Cov = self.cov
        
        self.ln_L = 0.0
        self.chi2_0 = 0.0
        self.m_flux_phi = np.nan * np.ones_like(self.d_flux)  # scaled model flux

        N_d = self.d_mask.sum()  # Number of (valid) data points
        d_flux = self.d_flux[self.d_mask]  # data flux
        m_flux_masked = m_flux[self.d_mask]  # model flux
        
        if self.scale_flux:  # Find the optimal phi-vector to match the observed spectrum
            self.m_flux_phi[self.d_mask], self.phi = self.get_flux_scaling(d_flux, m_flux_masked, Cov)

        residuals_phi = (self.d_flux - self.m_flux_phi)  # Residuals wrt scaled model
        inv_cov_0_residuals_phi = Cov.solve(residuals_phi[self.d_mask]) 
        chi2_0 = np.dot(residuals_phi[self.d_mask].T, inv_cov_0_residuals_phi)  # Chi-squared for the optimal linear scaling
        logdet_MT_inv_cov_0_M = 0

        if self.scale_flux:
            inv_cov_0_M = Cov.solve(m_flux_masked)  # Covariance matrix of phi
            MT_inv_cov_0_M = np.dot(m_flux_masked.T, inv_cov_0_M)
            logdet_MT_inv_cov_0_M = np.log(MT_inv_cov_0_M)  # (log)-determinant of the phi-covariance matrix

        if self.scale_err: 
            self.s2 = self.get_err_scaling(chi2_0, N_d)  # Scale variance to maximize log-likelihood
        logdet_cov_0 = Cov.get_logdet()  # Get log of determinant (log prevents over/under-flow)
        self.ln_L += -1/2*(N_d-self.N_phi) * np.log(2*np.pi) + loggamma(1/2*(N_d-self.N_phi+self.alpha-1))  # see Ruffio+2019
        self.ln_L += -1/2*(logdet_cov_0+logdet_MT_inv_cov_0_M+(N_d-self.N_phi+self.alpha-1)*np.log(chi2_0))
        self.chi2_0 += chi2_0
        # Reduced chi-squared (take degrees of freedom into account)
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
