#
#
#
from esys.escript import *
from esys.weipa import saveSilo
from esys.escript.minimizer import CostFunction, MinimizerException
from .tools import setupERTPDE
import logging
import numpy as np
from esys.escript.pdetools import Locator,  ArithmeticTuple
from .inversionsIP import IPMisfitCostFunction

class IPConductivityModelTemplate(object):
    """
    this a template for class for providing an electric conductivity model for IP inversion
    the basic functionality is that it provides values for secondary conductivity sigma_0 as function of normalized chargeability M_n but may extend
    to other models such as temperature dependencies
    """

    def __init__(self, T_min=1, T_max=1000, **kargs):
        """
        :param T_min: lower cut-off temperature
        :param T_max: upper cut-off temperature

        """
        self.T_min = T_min
        self.T_max = T_max

    def getDCConductivity(self, Mn):
        raise NotImplementedError

    def getDsigma_0DMn(self, Mn):
        raise NotImplementedError

    def getDMnDsigma_0(self, sigm0):
        raise NotImplementedError


class ConductivityModelByTemperature(IPConductivityModelTemplate):
    """
    temperature based conductivity model
    """

    def __init__(self, sigma_0_ref=1.52e-4, Mn_ref=5.054e-5, T_ref=15, P_0=2.9, P_Mn=2.5, T_min=1, T_max=1000, **kargs):
        """
        :param T_ref: reference temperature
        :param sigma_0_ref: secondary conductivity at reference temperature
        :param Mn_ref: chargeability at at reference temperature
        :param P_0: exponent of secondary conductivity  temperature model
        :param P_Mn: exponent of chargeability temperature model
        :param T_min: lower cut-off temperature
        :param T_max: upper cut-off temperature
        """
        super().__init__(T_min=T_min, T_max=T_max, **kargs)
        self.T_ref = T_ref
        self.sigma_0_ref = sigma_0_ref
        self.Mn_ref = Mn_ref
        self.P_0 = float(P_0)
        self.P_Mn = float(P_Mn)

    def getInstantanousConductivity(self, T):
        """
        return the secondary conductivity sigma_0 for given temperature T or given chargeability Mn
        """
        sigma_oo = self.getChargeability(T=T) + self.getDCConductivity(T=T)
        return sigma_oo

    def getDCConductivity(self, T=None, Mn=None):
        """
        return the secondary conductivity sigma_0 for given temperature T or given chargeability Mn
        """
        if not T is None:
            sigma_0 = self.sigma_0_ref * (T / self.T_ref) ** self.P_0
        elif not Mn is None:
            sigma_0 = self.sigma_0_ref * (Mn / self.Mn_ref) ** (self.P_0 / self.P_Mn)
        return sigma_0

    def getTemperature(self, Mn=None, sigma_0=None):
        """
        return the temperature for  given secondary conductivity sigma_0  or given  chargeability Mn
        """
        if not Mn is None:
            T = self.T_ref * (Mn / self.Mn_ref) ** (1. / self.P_Mn)
        elif not sigma_0 is None:
            T = self.T_ref * (sigma_0 / self.sigma_0_ref) ** (1. / self.P_0)
        return T

    def getChargeability(self, T=None, sigma_0=None):
        """
        return the chargeability for given secondary conductivity sigma_0  or for given temperature T
        """
        if not T is None:
            Mn = self.Mn_ref * (T / self.T_ref) ** self.P_Mn
        elif not sigma_0 is None:
            Mn = self.Mn_ref * (sigma_0 / self.sigma_0_ref) ** (self.P_Mn / self.P_0)
        return Mn

    def getDsigma_0DMn(self, Mn):
        Dsigma_0DMn = (self.P_0 / self.P_Mn) * (self.sigma_0_ref / self.Mn_ref) * (Mn / self.Mn_ref) ** (
                    self.P_0 / self.P_Mn - 1)
        return Dsigma_0DMn

    def getDMnDsigma_0(self, sigma_0):
        DMnDsigma_0 = (self.P_Mn / self.P_0) * (self.Mn_ref / self.sigma_0_ref) * (sigma_0 / self.sigma_0_ref) ** (
                    self.P_Mn / self.P_0 - 1)
        return DMnDsigma_0

    def getDsigma_0DT(self, T):
        Dsigma_0DT = self.P_0 * (self.sigma_0_ref / self.T_ref) * (T / self.T_ref) ** (self.P_0 - 1)
        return Dsigma_0DT

    def getDMnDT(self, T):
        DMnDT = self.P_Mn * (self.Mn_ref / self.T_ref) * (T / self.T_ref) ** (self.P_Mn - 1)
        return DMnDT


class InversionIPByTemperature(IPMisfitCostFunction):
    """
    Class to run a IP inversion for conductivity (sigma_0, DC) and normalized chargeabilty (Mn=sigma_oo-sigma_0)
    using H2-regularization on M = grad(m-m_ref) (as 6 components) with m=[m[0], m[1]]:

             m[0]=log(sigma/sigma_ref), m[1]=log(Mn/Mn_ref)

    To recover m from we solve

             (grad(v), grad(m[i]) = (grad(v), M[i*3:(i+1)*3])

    cross-gradient over V is added to cost function with weighting factor theta (=0 by default)
    """

    def __init__(self, domain, data, maskZeroPotential = None,
                 conductivity_model=IPConductivityModelTemplate(),
                 surface_temperature = None, mask_surface_temperature = None,
                 sigma_src=None, pde_tol=1e-8, stationsFMT="e%s", length_scale=None,
                 useLogMisfitDC=False, dataRTolDC=1e-4, useLogMisfitIP=False, dataRTolIP=1e-4,
                 weightingMisfitDC=1, w1=1, reg_tol=None,
                 conductivity=1., logger=None, **kargs):
        """
        :param domain: PDE & inversion domain
        :param data: survey data, is `fingal.SurveyData`. Resistence and secondary potential data are required.
        :param maskZeroPotential: mask of locations where electric potential is set to zero.
        :param conductivity_model: conductivity model
        :param sigma_src: background conductivity, used to calculate the source potentials. If not set sigma_0_ref
        is used.
        :

        :param pde_tol: tolerance for solving the forward PDEs and recovering m from M.
        :param stationsFMT: format string to convert station id to mesh label
        :param m_ref: reference property function
        :param zero_mean_m: constrain m by zero mean.
        :param useLogMisfitDC: if set logarithm of DC data is used in misfit.
        :param useLogMisfitIP: if set logarithm of secondary potential (IP) data is used in misfit.
        :param dataRTolDC: relative tolerance for damping small DC data on misfit
        :param dataRTolIP: relative tolerance for damping small secondary potential data in misfit
        :param weightingMisfitDC: weighting factor for DC data in misfit. weighting factor for IP data is one.
        :param sigma_0_ref: reference DC conductivity
        :param Mn_ref: reference Mn conductivity
        :param w1: regularization weighting factor(s) (scalar or numpy.ndarray)
        :param length_scale: length scale, w0=(1/length scale)**2 is weighting factor for |grad m|^2 = |M|^2 in
                            the regularization term. If `None`, the term is dropped.
        :param theta: weigthing factor x-grad term
        :param fixTop: if set m[0]-m_ref[0] and m[1]-m_ref[1] are fixed at the top of the domain
                        rather than just on the left, right, front, back and bottom face.
        :param logclip: value of m are clipped to stay between -logclip and +logclip
        :param m_epsilon: threshold for small m, grad(m) values.
        :param reg_tol: tolerance for PDE solve for regularization.
        :param save_memory: if not set, the three PDEs for the three conponents of the inversion unknwon
                            with the different boundary conditions are held. Otherwise, boundary conditions
                            are updated for each component which requires refactorization which obviously
                            requires more time.
        :param logger: the logger, if not set, 'fingal.IPInversion.H2' is used.
        """
        if sigma_src == None:
            sigma_src = conductivity_model.sigma_0_ref
        if logger is None:
            self.logger = logging.getLogger('fingal.IPInversionByTemperature')
        else:
            self.logger = logger
        assert length_scale > 0, "Length scale must be positive."
        super().__init__(domain=domain, data=data, sigma_src=sigma_src, pde_tol=pde_tol,
                         maskZeroPotential=maskZeroPotential, stationsFMT=stationsFMT,
                         useLogMisfitDC=useLogMisfitDC, dataRTolDC=dataRTolDC,
                         useLogMisfitIP=useLogMisfitIP, dataRTolIP=dataRTolIP,
                         weightingMisfitDC=weightingMisfitDC,
                         logger=logger, **kargs)
        self.conductivity_model = conductivity_model
        if length_scale is None:
            raise ValueError("Length scale must be given")
        else:
            self.a = (1. / length_scale) ** 2
        self.logger.info("Length scale factor is %s." % (str(length_scale)))
        # PDE to recover temperature:
        if surface_temperature is None:
            surface_temperature = Scalar( self.conductivity_model.T_ref, Solution(domain))
        if mask_surface_temperature is None:
            z =Solution(domain).getX()[2]
            mask_surface_temperature = whereZero(z-sup(z))

        self.Tpde = setupERTPDE(self.domain)
        self.Tpde.getSolverOptions().setTolerance(pde_tol)
        optionsG = self.Tpde.getSolverOptions()
        from esys.escript.linearPDEs import SolverOptions
        optionsG.setSolverMethod(SolverOptions.DIRECT)
        self.Tpde.setValue(A=conductivity * kronecker(3), q=mask_surface_temperature)
        ## get the initial temperature:
        self.Tpde.setValue(r=surface_temperature)
        self.T_background = self.Tpde.getSolution()
        self.logger.debug("Background Temperature = %s" % (str(self.T_background)))
        self.Tpde.setValue(r=Data())
        # ... regularization
        if not reg_tol:
            reg_tol = min(sqrt(pde_tol), 1e-3)
        self.logger.debug(f'Tolerance for solving regularization PDE is set to {reg_tol}')
        self.Hpde = setupERTPDE(self.domain)
        self.Hpde.getSolverOptions().setTolerance(reg_tol)
        self.Hpde.setValue(A=kronecker(3), D=self.a)

        self.setW1(w1)
    #====
    def getTemperature(self, M):
        """
        returns the temperature from property function M
        """
        self.Tpde.setValue(X=M, Y =Data())
        T = self.Tpde.getSolution() + self.T_background
        T = clip(T, minval=self.conductivity_model.T_min, maxval=self.conductivity_model.T_max)
        return T

    def setW1(self, w1):
        self.w1 = w1
        self.logger.debug(f'w1 = {self.w1:g}')

    def getSigma0(self, T, applyInterploation=False):
        """
        get the sigma_0 from temperature T
        """
        if applyInterploation:
            iT = interpolate(T, Function(self.domain))
        else:
            iT = T
        sigma_0 =  self.conductivity_model.getDCConductivity(T=iT)
        return sigma_0

    def getMn(self, T, applyInterploation=False):
        if applyInterploation:
            iT= interpolate(T, Function(self.domain))
        else:
            iT= T
        Mn  =  self.conductivity_model.getChargeability(T=iT)
        return Mn

    def getDsigma0DT(self, sigma_0, T):
        iT = interpolate(T, sigma_0.getFunctionSpace())
        return self.conductivity_model.getDsigma_0DT(T=iT)

    def getDMnDT(self, Mn, T):
        iT = interpolate(T, Mn.getFunctionSpace())
        return self.conductivity_model.getDMnDT(T=iT)

    def extractPropertyFunction(self, M):
        return M

    def getArguments(self, M):
        """
        precalculation
        """
        M2 = self.extractPropertyFunction(M)
        iM = interpolate(M2, Function(self.domain))
        T=self.getTemperature(iM)
        iT = interpolate(T, Function(self.domain))
        iT_stations = self.grabValuesAtStations(T)
        self.logger.debug("Temperature = %s" % (str(iT)))
        #self.logger.debug("Temperature at stations = %s" % (str(iT_stations)))

        isigma_0 = self.getSigma0(iT)
        isigma_0_stations = self.getSigma0(iT_stations)
        iMn = self.getMn(iT)
        args2 = self.getIPModelAndResponse(isigma_0, isigma_0_stations, iMn)
        return iT, isigma_0, isigma_0_stations, iMn, args2

    def getValue(self, M, iT, isigma_0, isigma_0_stations, iMn, args2):
        """
        return the value of the cost function
        """
        misfit_DC, misfit_IP = self.getMisfit(*args2)
        gM = grad(M, where=iT.getFunctionSpace())
        iM = interpolate(M, iT.getFunctionSpace())
        R = 1. / 2. * integrate(self.w1 * (length(gM) ** 2 + self.a * length(iM) ** 2) )
        V = R + misfit_DC + misfit_IP
        self.logger.debug(
                f'misfit ERT, IP; reg, total \t=  {misfit_DC:e}, {misfit_IP:e};  {R:e} = {V:e}')
        self.logger.debug(
                f'ratios ERT, IP; reg \t=  {misfit_DC / V * 100:g}, {misfit_IP / V * 100:g};  {R / V * 100:g}')

        return V

    #=======================================
    def getGradient(self, M, iT, isigma_0, isigma_0_stations, iMn, args2):
        """
        returns the gradient of the cost function. Overwrites `getGradient` of `MeteredCostFunction`
        """
        gM = grad(M, where=iT.getFunctionSpace())
        iM = interpolate(M, iT.getFunctionSpace())

        X = self.w1 * gM
        Y = self.w1 * self.a * iM

        DMisfitDsigma_0, DMisfitDMn = self.getDMisfit(isigma_0, iMn, *args2)
        Dsigma_0DT = self.getDsigma0DT(isigma_0, iT)
        DMnDT = self.getDMnDT(iMn, iT)
        Ystar = DMisfitDMn * DMnDT + DMisfitDsigma_0 * Dsigma_0DT
        self.Tpde.setValue(Y = Ystar, X=Data())
        Tstar = self.Tpde.getSolution()
        Y += grad(Tstar, where=Y.getFunctionSpace())
        return ArithmeticTuple(Y, X)

    def getInverseHessianApproximation(self, r,  iT, isigma_0, isigma_0_stations, iMn, args, initializeHessian=False):
        """
        returns an approximation of inverse of the Hessian. Overwrites `getInverseHessianApproximation` of `MeteredCostFunction`
        """
        P = Data(0., (3,), Solution(self.domain))
        for k in [0, 1, 2]:
            self.Hpde.setValue(X=r[1][k], Y=r[0][k])
            P[k] = self.Hpde.getSolution() / self.w1
            self.logger.debug(f"search direction component {k} = {str(P[k])}.")
        return P

    def getDualProduct(self, M, r):
        """
        dual product of gradient `r` with increment `V`. Overwrites `getDualProduct` of `MeteredCostFunction`
        """
        return integrate(inner(r[0], M) + inner(r[1], grad(M)))

    def getNorm(self, M):
        """
        returns the norm of property function `m`. Overwrites `getNorm` of `MeteredCostFunction`
        """
        return Lsup(M)

    def getSqueezeFactor(self, M, p):
        return None

