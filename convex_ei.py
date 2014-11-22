import numpy as np
from scipy.special import erfc

from openmdao.main.api import Component
from openmdao.lib.datatypes.api import Array, Instance, Float
from openmdao.main.uncertain_distributions import NormalDistribution


class ConvexEI(Component): 
    """Convex EI function, using thetas and R""" 

    target = Float(0, iotype="in", desc="Current lowest found value of the objective.")
    thetas = Array(iotype="in", desc="theta values of the kriging model")
    R = Array(iotype="in", desc="correlation matrix of the kriging model")

    current = Instance(NormalDistribution, iotype="in",
                       desc="The Normal Distribution of the predicted value "
                            "for some function at some point where you wish to"
                            " calculate the EI.")

    EI = Float(0.0, iotype="out",
               desc="The expected improvement of the predicted_value " + \
                    "in current.")

    PI = Float(0.0, iotype="out",
               desc="The probability of improvement of the predicted_value" + \
                    " in current.")

    def execute(self):  # currently using the regular EI calculation. You need to modify" 
        
        mu = self.current.mu
        sigma = self.current.sigma
        target = self.target

        try:
            np.seterr(divide='raise')
            self.PI = 0.5*erfc(-(1./2.**.5)*((target-mu)/sigma))

            T1 = (target-mu)*.5*(erfc(-(target-mu)/(sigma*2.**.5)))
            T2 = sigma*((1./((2.*np.pi)**.5))*np.exp(-0.5*((target-mu)/sigma)**2.))
            self.EI = np.abs(T1+T2)

        except (ValueError, ZeroDivisionError, FloatingPointError):
            self.EI = 0
            self.PI = 0
