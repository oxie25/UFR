"""

Oliver Xie, Olsen Group at Massachusetts Institute of Technology, 2025
@author oxie25

"""

import numpy as np
import pandas as pd
from scipy import optimize
import matplotlib.pyplot as plt
from ..model.idac_model import factorial

# Currently, only binary mixtures are supported
class Combinatorial():
    def __init__(self, mix, model, p):
        # Some constants
        self.v0 = 0.01517 # the -CH2- reference volume used by Abrams and Prausnitz (m3/kmol)
        self.s0 = 2.5 * 10**8 # the -CH2- reference area used by Abrams and Prausnitz (m2/kmol)
        self.z = 10 # coordination number used by Abrams and Prausnitz
        self.mix = mix

        # Select but do not instantiate model
        if model == 'FH':
            self.model = self.flory
        elif model == 'FV':
            self.model = self.elbro
        elif model == 'mod_FH':
            self.model = self.mod_flory
        elif model == 'mod_FV':
            self.model = self.mod_elbro
        elif model == 'SG':
            self.model = self.sg
        elif model == 'GK-FV':
            self.model = self.gk_fv
        elif model == 'none':
            self.model = self.none # This is for testing only, doesn't implement a combinatorial
        else:
            print(f'Combinatorial model of {model} not recognized')

        if mix == 2:
            # Unpack binary
            pA, pB = p

            # Unpack the molecular parameters, this is a dataframe item so has column labels 'van der waals volume (m3/kmol)', 'van der waals area (m2/kmol)', 'H donor sites', 'H acceptor sites', 'A', 'B', 'C', 'D', 'Equation'
            # VdW volume and area
            self.vdw_volume_A = pA['van der waals volume (m3/kmol)']
            self.vdw_volume_B = pB['van der waals volume (m3/kmol)']
            self.vdw_area_A = pA['van der waals area (m2/kmol)']
            self.vdw_area_B = pB['van der waals area (m2/kmol)']
            
            # Danner coefficients
            if model == 'FV' or model == 'mod_FV' or model == 'GK-FV':
                self.danner_coeff_A = pA[['A', 'B', 'C', 'D', 'Equation']].to_numpy()
                self.danner_coeff_B = pB[['A', 'B', 'C', 'D', 'Equation']].to_numpy()

    def daubert_danner(self, T, danner_coeff):
        # Danner coeffs are in order of A, B, C, D, eqn_num
        # Calculate the molar volume im m3/kmol at the prescribed temperature T in Kelvin
        # Determine which equation to use
        A, B, C, D, eqn_num = danner_coeff
        try:
            if eqn_num == 105:
                factor = 1 + (1 - (T / C))**D
                rho = A / B**factor # in units of kmol/m3
                #if np.any(np.isnan(factor)) or np.any(np.isinf(factor)):
                #    print(f"Invalid value encountered in power for base: {T}, exponent: {D}")
            elif eqn_num == 100:
                rho = A + B*T + C*T**2 + D*T**3 # E * T**4
            elif eqn_num == 116:
                tau = (1 - (T / 647.096)) # water
                rho = A + B*tau**0.35 + C * tau**(2/3) + D*tau
            else:
                print(f"Unknown equation number, reported value is: {eqn_num}")

        except RuntimeWarning as e:
            print(f"RuntimeWarning: {e}")
            print(f"Invalid value encountered in power for base: {T / C}, exponent: {D}")

        v = (1 / rho)

        return v
    
    def flory(self, x, invT, degree):
        v_A = self.vdw_volume_A
        v_B = self.vdw_volume_B

        if self.mix == 2:
            xA, xB = x

            if degree == 0:
                # Calculate the free energy (not excess)
                denom = v_A * xA + v_B * xB
                g = xA * np.log((v_A * xA / denom)) + xB * np.log(v_B * xB / denom)

                return g

            elif degree == 1:
                # Calculate the activity coefficients (first deriv of EXCESS gibbs free energy)
                denom = v_A * xA + v_B * xB

                ln_y_A = np.log( (v_A / denom) ) + 1 - (v_A / denom)
                ln_y_B = np.log( (v_B / denom) ) + 1 - (v_B / denom)

                return ln_y_A, ln_y_B

    
    def mod_flory(self, x, invT, degree):
        v_A = self.vdw_volume_A
        v_B = self.vdw_volume_B

        if self.mix == 2:
            xA, xB = x

            if degree == 0:
                # Calculate the free energy (not excess)
                denom = v_A**(2/3) * xA + v_B**(2/3) * xB
                g = xA * np.log((v_A**(2/3) * xA / denom)) + xB * np.log(v_B**(2/3) * xB / denom)

                return g

            elif degree == 1:
                # Calculate the activity coefficients (first deriv of EXCESS gibbs free energy)
                denom = v_A**(2/3) * xA + v_B**(2/3) * xB

                ln_y_A = np.log( (v_A**(2/3) / denom) ) + 1 - (v_A**(2/3) / denom)
                ln_y_B = np.log( (v_B**(2/3) / denom) ) + 1 - (v_B**(2/3) / denom)

                return ln_y_A, ln_y_B
            
    def elbro(self, x, invT, degree):
        T = 1 / invT
        vol_calc_A = self.daubert_danner(T, self.danner_coeff_A)
        vol_calc_B = self.daubert_danner(T, self.danner_coeff_B)

        fv_A = vol_calc_A - self.vdw_volume_A
        fv_B = vol_calc_B - self.vdw_volume_B

        if self.mix == 2:
            xA, xB = x

            if degree == 0:
                # Calculate the free energy (not excess)
                denom = fv_A * xA + fv_B * xB
                g = xA * np.log((fv_A * xA / denom)) + xB * np.log(fv_B * xB / denom)

                return g
            elif degree == 1:
                # Calculate the activity coefficients (first deriv of EXCESS gibbs free energy)
                denom = fv_A * xA + fv_B * xB

                ln_y_A = np.log( (fv_A / denom) ) + 1 - (fv_A / denom)
                ln_y_B = np.log( (fv_B / denom) ) + 1 - (fv_B / denom)

                return ln_y_A, ln_y_B
        

    def mod_elbro(self, x, invT, degree):
        T = 1 / invT
        vol_calc_A = self.daubert_danner(T, self.danner_coeff_A)
        vol_calc_B = self.daubert_danner(T, self.danner_coeff_B)

        fv_A = vol_calc_A - self.vdw_volume_A
        fv_B = vol_calc_B - self.vdw_volume_B

        if self.mix == 2:
            xA, xB = x

            if degree == 0:
                # Calculate the free energy (not excess)
                denom = fv_A**(2/3) * xA + fv_B**(2/3) * xB
                g = xA * np.log((fv_A**(2/3) * xA / denom)) + xB * np.log(fv_B**(2/3) * xB / denom)

                return g
            elif degree == 1:
                # Calculate the activity coefficients (first deriv of EXCESS gibbs free energy)
                denom = fv_A**(2/3) * xA + fv_B**(2/3) * xB

                ln_y_A = np.log( (fv_A**(2/3) / denom) ) + 1 - (fv_A**(2/3) / denom)
                ln_y_B = np.log( (fv_B**(2/3) / denom) ) + 1 - (fv_B**(2/3) / denom)

                return ln_y_A, ln_y_B
        
        
    def sg(self, x, invT, degree):
        T = 1 / invT
        r_A = self.vdw_volume_A / self.v0
        r_B = self.vdw_volume_B / self.v0
        q_A = self.vdw_area_A / self.s0
        q_B = self.vdw_area_B / self.s0

        if self.mix == 2:
            xA, xB = x

            if degree == 0:
                # Calculate the free energy (not excess)
                denom_r = r_A * xA + r_B * xB
                denom_q = q_A * xA + q_B * xB
                g = xA * np.log(r_A * xA / denom_r) + xB * np.log(r_B * xB / denom_r) + self.z/2 * (q_A * xA * np.log((q_A * xA / denom_q) / (r_A * xA / denom_r)) + q_B * xB * np.log((q_B * xB / denom_q) / (r_B * xB / denom_r)))

                return g
            elif degree == 1:
                # Calculate the activity coefficients (first deriv of EXCESS gibbs free energy)
                denom_r = r_A * xA + r_B * xB
                denom_q = q_A * xA + q_B * xB

                ln_y_A = np.log( (r_A / denom_r) ) + 1 - (r_A / denom_r) - self.z/2 * q_A * (np.log((r_A/denom_r)/(q_A/denom_q)) + 1 - (r_A/denom_r)/(q_A/denom_q))
                ln_y_B = np.log( (r_B / denom_r) ) + 1 - (r_B / denom_r) - self.z/2 * q_B * (np.log((r_B/denom_r)/(q_B/denom_q)) + 1 - (r_B/denom_r)/(q_B/denom_q))

                return ln_y_A, ln_y_B
            
    def gk_fv(self, x, invT, degree):
        T = 1 / invT
        r_A = self.vdw_volume_A / self.v0
        r_B = self.vdw_volume_B / self.v0
        q_A = self.vdw_area_A / self.s0
        q_B = self.vdw_area_B / self.s0

        T = 1 / invT
        vol_calc_A = self.daubert_danner(T, self.danner_coeff_A)
        vol_calc_B = self.daubert_danner(T, self.danner_coeff_B)

        r_fv_A = (vol_calc_A - self.vdw_volume_A) / self.v0
        r_fv_B = (vol_calc_B - self.vdw_volume_B) / self.v0

        if self.mix == 2:
            xA, xB = x

            if degree == 0:
                # Calculate the free energy (not excess)
                denom_r = r_A * xA + r_B * xB
                denom_r_fv = r_fv_A * xA + r_fv_B * xB
                denom_q = q_A * xA + q_B * xB

                g = xA * np.log(r_fv_A * xA / denom_r_fv) + xB * np.log(r_fv_B * xB / denom_r_fv) + self.z/2 * (q_A * xA * np.log((q_A * xA / denom_q) / (r_A * xA / denom_r)) + q_B * xB * np.log((q_B * xB / denom_q) / (r_B * xB / denom_r)))

                return g
            elif degree == 1:
                # Calculate the activity coefficients (first deriv of EXCESS gibbs free energy)
                denom_r = r_A * xA + r_B * xB
                denom_r_fv = r_fv_A * xA + r_fv_B * xB
                denom_q = q_A * xA + q_B * xB

                ln_y_A = np.log( (r_fv_A / denom_r_fv) ) + 1 - (r_fv_A / denom_r_fv) - self.z/2 * q_A * (np.log((r_A/denom_r)/(q_A/denom_q)) + 1 - (r_A/denom_r)/(q_A/denom_q))
                ln_y_B = np.log( (r_fv_B / denom_r_fv) ) + 1 - (r_fv_B / denom_r_fv) - self.z/2 * q_B * (np.log((r_B/denom_r)/(q_B/denom_q)) + 1 - (r_B/denom_r)/(q_B/denom_q))

                return ln_y_A, ln_y_B
            
    def none(self, x, invT, degree):
        # this is a dummy function for having no combinatorial model - for testing
        if self.mix == 2:
            xA, xB = x
            
            return np.zeros_like(xA)


class Residual():
    def __init__(self, mix, model, A, P, Q, Alpha, mol_param, temp_mat, dist_mat, temp_exponent):
        # Some constants
        self.v0 = 0.01517 # the -CH2- reference volume used by Abrams and Prausnitz (m3/kmol)
        self.s0 = 2.5 * 10**8 # the -CH2- reference area used by Abrams and Prausnitz (m2/kmol)
        self.z = 10 # coordination number used by Abrams and Prausnitz
        self.mix = mix

        # Select but do not instantiate model
        if model == 'NRTL':
            self.model = self.nrtl
        elif model == 'mod_UNIQUAC':
            self.model = self.mod_uniquac
        elif model == 'UNIQUAC':
            self.model = self.uniquac
        elif model == 'Wilson':
            self.model = self.wilson
        else:
            print(f'Residual model of {model} not recognized')
        
        if mix == 2:
            # Unpack and store as numpy
            A_A, A_B = A
            P_A, P_B = P
            mol_param_A, mol_param_B = mol_param

            Q_A, Q_B = Q

            self.A_A = A_A.to_numpy()
            self.A_B = A_B.to_numpy()
            self.P_A = P_A.to_numpy()
            self.P_B = P_B.to_numpy()
            self.Q_A = np.clip(Q_A.to_numpy(), a_min = 0, a_max = np.inf)
            self.Q_B = np.clip(Q_B.to_numpy(), a_min = 0, a_max = np.inf)
            
            # The only two values we need from mol_param are r and q
            self.rA = mol_param_A['van der waals volume (m3/kmol)'] / self.v0
            self.rB = mol_param_B['van der waals volume (m3/kmol)'] / self.v0
            self.qA = mol_param_A['van der waals area (m2/kmol)'] / self.s0
            self.qB = mol_param_B['van der waals area (m2/kmol)'] / self.s0

            # Transform temp_mat into the correct transformation matrix
            T = self.temp_layer(temp_mat)
            O = self.dist_layer(dist_mat)
            
            # Pre-calculate the energetic parameters uAB, uBA, and alpha
            # Calculate a self-consistent P
            O_expanded = np.expand_dims(O, axis = 0) # Dimension of 1 x Nparam x 2

            # Always stack product on difference
            A_prod = np.clip(self.A_A, a_min = 0, a_max = np.inf) * np.clip(self.A_B, a_min = 0, a_max = np.inf) # Clip to be between 0 and infinity
            A_dif = (self.A_A - self.A_B)**2
            A_stack = np.stack((A_prod, A_dif), axis = 1) # Nparam x 2
            A_stack = np.expand_dims(A_stack, axis = 0) # 1 x Nparam x 2
            A_comb = np.sum(A_stack * O_expanded, axis = -1) # Sum in last dimension (across 2), results in 1 x Nparam
            A_comb_temp = np.matmul(A_comb, T) # T is Nparam x Ntemp
            self.A_AB = A_comb_temp

            PA_prod = np.clip(self.A_A, a_min = 0, a_max = np.inf) * np.clip(self.A_A, a_min = 0, a_max = np.inf) # Clip to be between 0 and infinity
            PA_dif = (self.A_A - self.A_A)**2
            PA_stack = np.stack((PA_prod, PA_dif), axis = 1) # Nparam x 2
            PA_stack = np.expand_dims(PA_stack, axis = 0) # 1 x Nparam x 2
            PA_comb = np.sum(PA_stack * O_expanded, axis = -1) # Sum in last dimension (across 2), results in 1 x Nparam
            PA_comb_temp = np.matmul(PA_comb, T) # T is Nparam x Ntemp
            
            PB_prod = np.clip(self.A_B, a_min = 0, a_max = np.inf) * np.clip(self.A_B, a_min = 0, a_max = np.inf) # Clip to be between 0 and infinity
            PB_dif = (self.A_B - self.A_B)**2
            PB_stack = np.stack((PB_prod, PB_dif), axis = 1) # Nparam x 2
            PB_stack = np.expand_dims(PB_stack, axis = 0) # 1 x Nparam x 2
            PB_comb = np.sum(PB_stack * O_expanded, axis = -1) # Sum in last dimension (across 2), results in 1 x Nparam
            PB_comb_temp = np.matmul(PB_comb, T) # T is Nparam x Ntemp

            # Overwrite P_A and P_B
            self.P_A = PA_comb_temp
            self.P_B = PB_comb_temp

            # Subtract out the self-interaction parameter
            self.Uab = A_comb_temp.flatten() - self.P_B.flatten() # 1D ready for multiplying with temp
            self.Uba = A_comb_temp.flatten() - self.P_A.flatten() # 1D ready for multiplying with temp

            # Create an exponent tensor of shape (Ntemp) using passed value
            if T.shape[1] == 1: # If ony one temp dim, then create only a single column which gets 1/T
                self.exponents = np.ones(T.shape[1])
            else:
                self.exponents = temp_exponent.flatten()

            if np.min(temp_exponent) < 0 and np.max(temp_exponent) == 1:
                # Negative temperature powers means that we are doing a Taylor series expansion in T. Calculate and multiply the correct coefficients
                taylor_coeffs = np.array([factorial(x) for x in np.absolute(temp_exponent - 1)])
                self.taylor_coeffs = 1 / taylor_coeffs.flatten()  # Shape (1, Ntemp)
                #print(f"Taylor coefficients: {self.taylor_coeffs}")
            elif np.min(temp_exponent) == 0 and np.max(temp_exponent) > 0:
                # Positive temperature powers means that we are doing a Taylor series expansion in 1/T. Calculate and multiply the correct coefficients
                taylor_coeffs = np.array([factorial(x) for x in np.absolute(temp_exponent)])
                self.taylor_coeffs = 1 / taylor_coeffs.flatten()  # Shape (1, Ntemp)
                #print(f"Taylor coefficients: {self.taylor_coeffs}")
            else:
                self.taylor_coeffs = None

            # NRTL specific items
            if model == 'NRTL':
                Alpha_A, Alpha_B = Alpha
                self.Alpha_A = np.clip(Alpha_A, a_min = 0, a_max = 1)
                self.Alpha_B = np.clip(Alpha_B, a_min = 0, a_max = 1)
                self.alpha = np.sum( (self.Alpha_A - self.Alpha_B)**2 )

    def temp_layer(self, temp_mat):
        T = np.exp(temp_mat) / np.sum(np.exp(temp_mat), axis = 1, keepdims = 1) # Keep dim
        
        return T
    
    def dist_layer(self, dist_mat):
        # Only softmax for now
        O = np.exp(dist_mat) / np.sum(np.exp(dist_mat), axis = 1, keepdims = 1) # Keep dim, sum across 2nd dimension

        return O

    def uniquac(self, x, invT, degree):
        if self.taylor_coeffs is not None:
            Tmat = invT.reshape(1, 1, -1)**self.exponents.reshape(-1, 1, 1) * self.taylor_coeffs.reshape(-1, 1, 1)
        else:
            Tmat = invT.reshape(1, 1, -1)**self.exponents.reshape(-1, 1, 1) # array of nT x 1 x npts
        
        if self.mix == 2:
            xA, xB = x
            theta = self.qA * xA / (self.qA * xA + self.qB * xB)
            phi = self.rA * xA / (self.rA * xA + self.rB * xB)

            uab = np.sum(self.Uab.reshape(-1, 1, 1) * Tmat, axis = 0).flatten() # element-wise multiply and sum in the first dim, return to a 1D array
            uba = np.sum(self.Uba.reshape(-1, 1, 1) * Tmat, axis = 0).flatten()

            if degree == 0:
                g = -self.qA * xA * np.log(theta + (1-theta) * np.exp(-uba)) - self.qB * xB * np.log((1-theta) + theta * np.exp(-uab))

                return g

            elif degree == 1:
                ln_y_A = self.qA * ( 1 - np.log(theta + (1-theta) * np.exp(-uba)) - theta / (theta + (1-theta)*np.exp(-uba)) - (1-theta)*np.exp(-uab) / ((1-theta) + theta*np.exp(-uab)))
                ln_y_B = self.qB * ( 1 - np.log((1-theta) + theta * np.exp(-uab)) - (1-theta) / ((1-theta) + theta*np.exp(-uab)) - theta*np.exp(-uba) / (theta + (1-theta)*np.exp(-uba)))

                return ln_y_A, ln_y_B
            
            elif degree == 'param':
                param_dict = {'Uba': self.Uba, 'Uab': self.Uab, 'uaa': self.P_A, 'ubb': self.P_B, 'uab': self.A_AB}

                return param_dict
            
    def mod_uniquac(self, x, invT, degree):
        # degree marks whether we want 0, 1st, 2nd or 3rd derivative
        # Generate theta and phi

        if self.taylor_coeffs is not None:
            Tmat = invT.reshape(1, 1, -1)**self.exponents.reshape(-1, 1, 1) * self.taylor_coeffs.reshape(-1, 1, 1)
        else:
            Tmat = invT.reshape(1, 1, -1)**self.exponents.reshape(-1, 1, 1) # array of nT x 1 x npts

        if self.mix == 2:
            Q_A = self.Q_A #* self.qA
            Q_B = self.Q_B #* self.qB

            xA, xB = x
            theta = Q_A * xA / (Q_A * xA + Q_B * xB)#self.qA * xA / (self.qA * xA + self.qB * xB)
            phi = self.rA * xA / (self.rA * xA + self.rB * xB)

            uab = np.sum(self.Uab.reshape(-1, 1, 1) * Tmat, axis = 0).flatten() # element-wise multiply and sum in the first dim, return to a 1D array
            uba = np.sum(self.Uba.reshape(-1, 1, 1) * Tmat, axis = 0).flatten()

            if degree == 0:
                g = -self.Q_A * xA * np.log(theta + (1-theta) * np.exp(-uba)) - self.Q_B * xB * np.log((1-theta) + theta * np.exp(-uab))

                return g
    
            elif degree == 1:
                ln_y_A = Q_A * ( 1 - np.log(theta + (1-theta) * np.exp(-uba)) - theta / (theta + (1-theta)*np.exp(-uba)) - (1-theta)*np.exp(-uab) / ((1-theta) + theta*np.exp(-uab)))
                ln_y_B = Q_B * ( 1 - np.log((1-theta) + theta * np.exp(-uab)) - (1-theta) / ((1-theta) + theta*np.exp(-uab)) - theta*np.exp(-uba) / (theta + (1-theta)*np.exp(-uba)))

                return ln_y_A, ln_y_B
        
            elif degree == 'param':
                param_dict = {'Uba': self.Uba, 'Uab': self.Uab, 'Qa': Q_A, 'Qb': Q_B, 'uaa': self.P_A, 'ubb': self.P_B, 'uab': self.A_AB}
                return param_dict

    def wilson(self, x, invT, degree):
        if self.taylor_coeffs is not None:
            Tmat = invT.reshape(1, 1, -1)**self.exponents.reshape(-1, 1, 1) * self.taylor_coeffs.reshape(-1, 1, 1)
        else:
            Tmat = invT.reshape(1, 1, -1)**self.exponents.reshape(-1, 1, 1) # array of nT x 1 x npts

        if self.mix == 2:
            xA, xB = x
            theta = self.qA * xA / (self.qA * xA + self.qB * xB)
            phi = self.rA * xA / (self.rA * xA + self.rB * xB)

            uab = np.sum(self.Uab.reshape(-1, 1, 1) * Tmat, axis = 0).flatten() # element-wise multiply and sum in the first dim, return to a 1D array
            uba = np.sum(self.Uba.reshape(-1, 1, 1) * Tmat, axis = 0).flatten()

            if degree == 0:
                g = -self.rA * xA * np.log(phi + (1-phi) * np.exp(-uba)) - self.rB * xB * np.log((1-phi) + phi * np.exp(-uab))

                return g

            elif degree == 1:
                ln_y_A = self.rA * ( 1 - np.log(phi + (1-phi) * np.exp(-uba)) - phi / (phi + (1-phi)*np.exp(-uba)) - (1-phi)*np.exp(-uab) / ((1-phi) + phi*np.exp(-uab)))
                ln_y_B = self.rB * ( 1 - np.log((1-phi) + phi * np.exp(-uab)) - (1-phi) / ((1-phi) + phi*np.exp(-uab)) - phi*np.exp(-uba) / (phi + (1-phi)*np.exp(-uba)))

                return ln_y_A, ln_y_B
        
            elif degree == 'param':
                param_dict = {'Uba': self.Uba, 'Uab': self.Uab, 'uaa': self.P_A, 'ubb': self.P_B, 'uab': self.A_AB}
                return param_dict
            

    def nrtl(self, x, invT, degree):
        if self.taylor_coeffs is not None:
            Tmat = invT.reshape(1, 1, -1)**self.exponents.reshape(-1, 1, 1) * self.taylor_coeffs.reshape(-1, 1, 1)
        else:
            Tmat = invT.reshape(1, 1, -1)**self.exponents.reshape(-1, 1, 1) # array of nT x 1 x npts
            
        if self.mix == 2:
            xA, xB = x
            uab = np.sum(self.Uab.reshape(-1, 1, 1) * Tmat, axis = 0).flatten() # element-wise multiply and sum in the first dim, return to a 1D array
            uba = np.sum(self.Uba.reshape(-1, 1, 1) * Tmat, axis = 0).flatten()
            alpha = self.alpha

            if degree == 0:
                g = xA * xB * uba * np.exp(-alpha * uba) / (xA + xB * np.exp(-alpha * uba)) + xA * xB * uab * np.exp(-alpha * uab) / (xB + xA * np.exp(-alpha * uab))

                return g
            
            elif degree == 1:
                ln_y_A = xB**2 * ( uba * np.exp(-2 * alpha * uba) / (xA + xB * np.exp(-alpha * uba))**2 + uab * np.exp(-alpha * uab) / (xB + xA * np.exp(-alpha * uab))**2 )
                ln_y_B = xA**2 * ( uab * np.exp(-2 * alpha * uab) / (xB + xA * np.exp(-alpha * uab))**2 + uba * np.exp(-alpha * uba) / (xA + xB * np.exp(-alpha * uba))**2 )
                return ln_y_A, ln_y_B
            
            elif degree == 'param':
                param_dict = {'alpha': alpha, 'Uba': self.Uba, 'Uab': self.Uab, 'uaa': self.P_A, 'ubb': self.P_B, 'uab': self.A_AB}
                return param_dict
            
class Association():
    def __init__(self, mix, model, D, mol_param):
        # Some constants
        self.v0 = 0.01517 # the -CH2- reference volume used by Abrams and Prausnitz (m3/kmol)
        self.s0 = 2.5 * 10**8 # the -CH2- reference area used by Abrams and Prausnitz (m2/kmol)
        self.z = 10 # coordination number used by Abrams and Prausnitz

        # Select but do not instantiate model
        if model == 'wertheim':
            self.model = self.wertheim
        elif model == 'none':
            self.model = None
        else:
            print(f'Association model of {model} not recognized')

        # Unpack and store as numpy
        D_A, D_B = D
        mol_param_A, mol_param_B = mol_param

        self.D_A = D_A.to_numpy()
        self.D_B = D_B.to_numpy()

        # We need r, q, N, rho
        self.rA = mol_param_A['van der waals volume (m3/kmol)'] / self.v0
        self.rB = mol_param_B['van der waals volume (m3/kmol)'] / self.v0
        self.qA = mol_param_A['van der waals area (m2/kmol)'] / self.s0
        self.qB = mol_param_B['van der waals area (m2/kmol)'] / self.s0
        self.N_a_A = mol_param_A['H acceptor sites']
        self.N_d_A = mol_param_A['H donor sites']
        self.N_a_B = mol_param_B['H acceptor sites']
        self.N_d_B = mol_param_B['H donor sites']

        self.mix = mix

    def softplus(self, x):
        return np.log(1 + np.exp(x))

    def check_solution(self, solution, solver, lower_bound = 0, upper_bound = 1):
        # Check if the solver was successful
        is_successful = solution.success
        message = solution.message
        
        # Check if all values are within the specified bounds
        within_bounds = all(lower_bound <= x <= upper_bound for x in solution.x)
        
        # Additional feedback
        if is_successful and within_bounds:
            pass
            # print(f"{solver}: Solution is successful and within bounds.")
        elif not is_successful:
            pass
            #print(f"{solver}: Solution is not successful: {message}")
        else:
            pass
            #print(f"{solver}: Solution is successful but has values out of bounds.")

        # Optionally, return the checks for programmatic use
        return is_successful, within_bounds

    def wertheim(self, x, invT, degree):
        # Wertheim association term only (solved using the fsolve method - most accurate)
        # Calculate the Wertheim term
        # values of acceptor/donor
        
        if self.mix == 2:
            x_A, x_B = x
            D_A = self.softplus(self.D_A)
            D_B = self.softplus(self.D_B)
            a_A = D_A[0]
            d_A = D_A[1]
            a_B = D_B[0]
            d_B = D_B[1]

            rA = self.rA
            rB = self.rB
            # Calculate the normalized densities
            rho_a_A_m = self.N_a_A * x_A / (rA * x_A + rB * x_B)
            rho_d_A_m = self.N_d_A * x_A / (rA * x_A + rB * x_B)
            rho_a_B_m = self.N_a_B * x_B / (rA * x_A + rB * x_B)
            rho_d_B_m = self.N_d_B * x_B / (rA * x_A + rB * x_B)

            # Pure normalized densities
            rho_a_A_p = self.N_a_A / rA
            rho_d_A_p = self.N_d_A / rA
            rho_a_B_p = self.N_a_B / rB
            rho_d_B_p = self.N_d_B / rB
        
            if degree == 0:
                # Calculate the association Delta function constant
                Delta_H2O = 0.034 * (np.exp(1960 * invT) - 1) # array due to T

                # Association term, from Lin et al, d_H2O and a_H2O are 1
                Delta_aA_dA = a_A * d_A * Delta_H2O
                Delta_aB_dB = a_B * d_B * Delta_H2O
                Delta_aA_dB = a_A * d_B * Delta_H2O
                Delta_aB_dA = a_B * d_A * Delta_H2O

                # We need to solve a system of coupled equations. No easy explicit form for non-infinite dilution
                def fXpure(x, rho, delta):
                    xd, xa = x
                    rho_d_p, rho_a_p = rho
                    # System of coupled equations
                    eq1 = 1 / (1 + rho_a_p * xa * delta) - xd
                    eq2 = 1 / (1 + rho_d_p * xd * delta) - xa
                    return [eq1, eq2]
                
                def fXmix(x, rho, delta):
                    xd_A, xa_A, xd_B, xa_B = x
                    rho_d_A_m, rho_a_A_m, rho_d_B_m, rho_a_B_m = rho
                    delta_aAdB, delta_aBdA, delta_aAdA, delta_aBdB = delta
                    eq1 = 1 / (1 + rho_a_A_m * xa_A * delta_aAdA + rho_a_B_m * xa_B * delta_aBdA) - xd_A
                    eq2 = 1 / (1 + rho_d_A_m * xd_A * delta_aAdA + rho_d_B_m * xd_B * delta_aAdB) - xa_A
                    eq3 = 1 / (1 + rho_a_A_m * xa_A * delta_aAdB + rho_a_B_m * xa_B * delta_aBdB) - xd_B
                    eq4 = 1 / (1 + rho_d_A_m * xd_A * delta_aBdA + rho_d_B_m * xd_B * delta_aBdB) - xa_B
                    return [eq1, eq2, eq3, eq4]

                # Calculate the pure X_a and X_d
                rho_A_p = [rho_d_A_p, rho_a_A_p]
                rho_B_p = [rho_d_B_p, rho_a_B_p]
                XPure_guess = [0.5, 0.5]
                XMix_guess = [0.5, 0.5, 0.5, 0.5]
                X_d_A_p = np.zeros_like(invT)
                X_a_A_p = np.zeros_like(invT)
                X_d_B_p = np.zeros_like(invT)
                X_a_B_p = np.zeros_like(invT)
                X_d_A_m = np.zeros_like(invT)
                X_a_A_m = np.zeros_like(invT)
                X_d_B_m = np.zeros_like(invT)
                X_a_B_m = np.zeros_like(invT)
                if np.shape(invT)[0] > 1 and np.shape(x_A)[0] > 1:
                    # Each xA and xB and T are linked. Let's iterate on T
                    for index, value in enumerate(invT):
                        sol = optimize.root(fXpure, XPure_guess, (rho_A_p, Delta_aA_dA[index]))
                        [x_d_A_p, x_a_A_p] = sol.x
                        self.check_solution(sol, 'A Pure')
                        sol = optimize.root(fXpure, XPure_guess, (rho_B_p, Delta_aB_dB[index]))
                        self.check_solution(sol, 'B Pure')
                        [x_d_B_p, x_a_B_p] = sol.x
                        X_d_A_p[index] = x_d_A_p
                        X_a_A_p[index] = x_a_A_p
                        X_d_B_p[index] = x_d_B_p
                        X_a_B_p[index] = x_a_B_p

                        Delta_mix = [Delta_aA_dB[index], Delta_aB_dA[index], Delta_aA_dA[index], Delta_aB_dB[index]]
                        rho_mix = [rho_d_A_m[index], rho_a_A_m[index], rho_d_B_m[index], rho_a_B_m[index]]
                        sol = optimize.root(fXmix, XMix_guess, (rho_mix, Delta_mix))
                        [x_d_A_m, x_a_A_m, x_d_B_m, x_a_B_m] = sol.x
                        self.check_solution(sol, 'Mix')
                        X_d_A_m[index] = x_d_A_m
                        X_a_A_m[index] = x_a_A_m
                        X_d_B_m[index] = x_d_B_m
                        X_a_B_m[index] = x_a_B_m
                        
                elif np.shape(invT)[0] > 1 and np.shape(x_A)[0] == 1:
                    # Each xA and xB and T are linked. Let's iterate on T
                    XMix_guess = [0.5, 0.5, 0.5, 0.5]
                    for index, value in enumerate(invT):
                        sol = optimize.root(fXpure, XPure_guess, (rho_A_p, Delta_aA_dA[index]))
                        [x_d_A_p, x_a_A_p] = sol.x
                        self.check_solution(sol, 'A Pure')
                        sol = optimize.root(fXpure, XPure_guess, (rho_B_p, Delta_aB_dB[index]))
                        [x_d_B_p, x_a_B_p] = sol.x
                        self.check_solution(sol, 'B Pure')
                        X_d_A_p[index] = x_d_A_p
                        X_a_A_p[index] = x_a_A_p
                        X_d_B_p[index] = x_d_B_p
                        X_a_B_p[index] = x_a_B_p

                        Delta_mix = [Delta_aA_dB[index], Delta_aB_dA[index], Delta_aA_dA[index], Delta_aB_dB[index]]
                        rho_mix = [rho_d_A_m[0], rho_a_A_m[0], rho_d_B_m[0], rho_a_B_m[0]]
                        sol = optimize.root(fXmix, XMix_guess, (rho_mix, Delta_mix))
                        [x_d_A_m, x_a_A_m, x_d_B_m, x_a_B_m] = sol.x
                        self.check_solution(sol, 'Mix')
                        X_d_A_m[index] = x_d_A_m
                        X_a_A_m[index] = x_a_A_m
                        X_d_B_m[index] = x_d_B_m
                        X_a_B_m[index] = x_a_B_m
                elif np.shape(invT)[0] == 1 and np.shape(x_A)[0] > 1:
                    X_d_A_p = np.zeros_like(x_A)
                    X_a_A_p = np.zeros_like(x_A)
                    X_d_B_p = np.zeros_like(x_A)
                    X_a_B_p = np.zeros_like(x_A)
                    X_d_A_m = np.zeros_like(x_A)
                    X_a_A_m = np.zeros_like(x_A)
                    X_d_B_m = np.zeros_like(x_A)
                    X_a_B_m = np.zeros_like(x_A)
                    for index, value in enumerate(x_A):
                        sol = optimize.root(fXpure, XPure_guess, (rho_A_p, Delta_aA_dA[0]))
                        [x_d_A_p, x_a_A_p] = sol.x
                        self.check_solution(sol, 'A Pure')
                        sol = optimize.root(fXpure, XPure_guess, (rho_B_p, Delta_aB_dB[0]))
                        [x_d_B_p, x_a_B_p] = sol.x
                        self.check_solution(sol, 'B Pure')
                        X_d_A_p[index] = x_d_A_p
                        X_a_A_p[index] = x_a_A_p
                        X_d_B_p[index] = x_d_B_p
                        X_a_B_p[index] = x_a_B_p

                        Delta_mix = [Delta_aA_dB[0], Delta_aB_dA[0], Delta_aA_dA[0], Delta_aB_dB[0]]
                        rho_mix = [rho_d_A_m[index], rho_a_A_m[index], rho_d_B_m[index], rho_a_B_m[index]]
                        sol = optimize.root(fXmix, XMix_guess, (rho_mix, Delta_mix))
                        [x_d_A_m, x_a_A_m, x_d_B_m, x_a_B_m] = sol.x
                        self.check_solution(sol, 'Mix')
                        X_d_A_m[index] = x_d_A_m
                        X_a_A_m[index] = x_a_A_m
                        X_d_B_m[index] = x_d_B_m
                        X_a_B_m[index] = x_a_B_m
                else:
                    # Each xA and xB and T are linked. Let's iterate on T
                    sol = optimize.root(fXpure, XPure_guess, (rho_A_p, Delta_aA_dA[0]))
                    [x_d_A_p, x_a_A_p] = sol.x
                    self.check_solution(sol, 'A Pure')
                    sol = optimize.root(fXpure, XPure_guess, (rho_B_p, Delta_aB_dB[0]))
                    [x_d_B_p, x_a_B_p] = sol.x
                    self.check_solution(sol, 'B Pure')
                    X_d_A_p = x_d_A_p
                    X_a_A_p = x_a_A_p
                    X_d_B_p = x_d_B_p
                    X_a_B_p = x_a_B_p

                    # Calculate the mixture X_a and X_d
                    XMix_guess = [0.5, 0.5, 0.5, 0.5]
                    Delta_mix = [Delta_aA_dB[0], Delta_aB_dA[0], Delta_aA_dA[0], Delta_aB_dB[0]]
                    rho_mix = [rho_d_A_m[0], rho_a_A_m[0], rho_d_B_m[0], rho_a_B_m[0]]
                    sol = optimize.root(fXmix, XMix_guess, (rho_mix, Delta_mix))
                    [x_d_A_m, x_a_A_m, x_d_B_m, x_a_B_m] = sol.x
                    self.check_solution(sol, 'Mix')
                    X_d_A_m = x_d_A_m
                    X_a_A_m = x_a_A_m
                    X_d_B_m = x_d_B_m
                    X_a_B_m = x_a_B_m

                # Compute the terms multiplied to N_a_A and N_d_A only if they are not zero.
                N_a_A_m_term = np.where((self.N_a_A == 0), 0, self.N_a_A * (np.log(X_a_A_m) - X_a_A_m/2 + self.N_a_A / 2) )
                N_d_A_m_term = np.where((self.N_d_A == 0), 0, self.N_d_A * (np.log(X_d_A_m) - X_d_A_m/2 + self.N_d_A / 2) )
                N_a_B_m_term = np.where((self.N_a_B == 0), 0, self.N_a_B * (np.log(X_a_B_m) - X_a_B_m/2 + self.N_a_B / 2) )
                N_d_B_m_term = np.where((self.N_d_B == 0), 0, self.N_d_B * (np.log(X_d_B_m) - X_d_B_m/2 + self.N_d_B / 2) )
                N_a_A_p_term = np.where((self.N_a_A == 0), 0, self.N_a_A * (np.log(X_a_A_p) - X_a_A_p/2 + self.N_a_A / 2) )
                N_d_A_p_term = np.where((self.N_d_A == 0), 0, self.N_d_A * (np.log(X_d_A_p) - X_d_A_p/2 + self.N_d_A / 2) )
                N_a_B_p_term = np.where((self.N_a_B == 0), 0, self.N_a_B * (np.log(X_a_B_p) - X_a_B_p/2 + self.N_a_B / 2) )
                N_d_B_p_term = np.where((self.N_d_B == 0), 0, self.N_d_B * (np.log(X_d_B_p) - X_d_B_p/2 + self.N_d_B / 2) )

                g = x_A * (N_a_A_m_term + N_d_A_m_term) + x_B * (N_a_B_m_term + N_d_B_m_term) - x_A * (N_a_A_p_term + N_d_A_p_term) - x_B * (N_a_B_p_term + N_d_B_p_term)

                return g

            elif degree == 1:
                # Calculate the association Delta function constant
                Delta_H2O = 0.034 * (np.exp(1960 * invT) - 1) # array due to T

                # Association term, from Lin et al, d_H2O and a_H2O are 1
                Delta_aA_dA = a_A * d_A * Delta_H2O
                Delta_aB_dB = a_B * d_B * Delta_H2O
                Delta_aA_dB = a_A * d_B * Delta_H2O
                Delta_aB_dA = a_B * d_A * Delta_H2O

                # We need to solve a system of coupled equations. No easy explicit form for non-infinite dilution
                def fXpure(x, rho, delta):
                    xd, xa = x
                    rho_d_p, rho_a_p = rho
                    # System of coupled equations
                    eq1 = 1 / (1 + rho_a_p * xa * delta) - xd
                    eq2 = 1 / (1 + rho_d_p * xd * delta) - xa
                    return [eq1, eq2]
                
                def fXmix(x, rho, delta):
                    xd_A, xa_A, xd_B, xa_B = x
                    rho_d_A_m, rho_a_A_m, rho_d_B_m, rho_a_B_m = rho
                    delta_aAdB, delta_aBdA, delta_aAdA, delta_aBdB = delta
                    eq1 = 1 / (1 + rho_a_A_m * xa_A * delta_aAdA + rho_a_B_m * xa_B * delta_aBdA) - xd_A
                    eq2 = 1 / (1 + rho_d_A_m * xd_A * delta_aAdA + rho_d_B_m * xd_B * delta_aAdB) - xa_A
                    eq3 = 1 / (1 + rho_a_A_m * xa_A * delta_aAdB + rho_a_B_m * xa_B * delta_aBdB) - xd_B
                    eq4 = 1 / (1 + rho_d_A_m * xd_A * delta_aBdA + rho_d_B_m * xd_B * delta_aBdB) - xa_B
                    return [eq1, eq2, eq3, eq4]

                # Calculate the pure X_a and X_d
                rho_A_p = [rho_d_A_p, rho_a_A_p]
                rho_B_p = [rho_d_B_p, rho_a_B_p]
                
                # Initial guess must be in right ratio or else the solver will not converge
                if rho_d_A_p > rho_a_A_p:
                    # More donors than acceptors, acceptor will have less open sites
                    XAPure_guess = [0.6, 0.4]
                elif rho_d_A_p < rho_a_A_p:
                    XAPure_guess = [0.4, 0.6]
                else:
                    XAPure_guess = [0.5, 0.5]

                if rho_d_B_p > rho_a_B_p:
                    # More donors than acceptors, acceptor will have less open sites
                    XBPure_guess = [0.6, 0.4]
                elif rho_d_B_p < rho_a_B_p:
                    XBPure_guess = [0.4, 0.6]
                else:
                    XBPure_guess = [0.5, 0.5]
                
                XMix_guess = XAPure_guess + XBPure_guess

                X_d_A_p = np.zeros_like(invT)
                X_a_A_p = np.zeros_like(invT)
                X_d_B_p = np.zeros_like(invT)
                X_a_B_p = np.zeros_like(invT)
                X_d_A_m = np.zeros_like(invT)
                X_a_A_m = np.zeros_like(invT)
                X_d_B_m = np.zeros_like(invT)
                X_a_B_m = np.zeros_like(invT)

                if np.shape(invT)[0] > 1 and np.shape(x_A)[0] > 1:
                    # Each xA and xB and T are linked. Let's iterate on T
                    for index, value in enumerate(invT):
                        sol = optimize.root(fXpure, XAPure_guess, (rho_A_p, Delta_aA_dA[index]))
                        [x_d_A_p, x_a_A_p] = sol.x
                        self.check_solution(sol, 'A Pure')
                        sol = optimize.root(fXpure, XBPure_guess, (rho_B_p, Delta_aB_dB[index]))
                        self.check_solution(sol, 'B Pure')
                        [x_d_B_p, x_a_B_p] = sol.x
                        X_d_A_p[index] = x_d_A_p
                        X_a_A_p[index] = x_a_A_p
                        X_d_B_p[index] = x_d_B_p
                        X_a_B_p[index] = x_a_B_p

                        Delta_mix = [Delta_aA_dB[index], Delta_aB_dA[index], Delta_aA_dA[index], Delta_aB_dB[index]]
                        rho_mix = [rho_d_A_m[index], rho_a_A_m[index], rho_d_B_m[index], rho_a_B_m[index]]
                        sol = optimize.root(fXmix, XMix_guess, (rho_mix, Delta_mix))
                        [x_d_A_m, x_a_A_m, x_d_B_m, x_a_B_m] = sol.x
                        self.check_solution(sol, 'Mix')
                        X_d_A_m[index] = x_d_A_m
                        X_a_A_m[index] = x_a_A_m
                        X_d_B_m[index] = x_d_B_m
                        X_a_B_m[index] = x_a_B_m
                        
                elif np.shape(invT)[0] > 1 and np.shape(x_A)[0] == 1:
                    # Each xA and xB and T are linked. Let's iterate on T
                    for index, value in enumerate(invT):
                        sol = optimize.root(fXpure, XAPure_guess, (rho_A_p, Delta_aA_dA[index]))
                        [x_d_A_p, x_a_A_p] = sol.x
                        self.check_solution(sol, 'A Pure')
                        sol = optimize.root(fXpure, XBPure_guess, (rho_B_p, Delta_aB_dB[index]))
                        [x_d_B_p, x_a_B_p] = sol.x
                        self.check_solution(sol, 'B Pure')
                        X_d_A_p[index] = x_d_A_p
                        X_a_A_p[index] = x_a_A_p
                        X_d_B_p[index] = x_d_B_p
                        X_a_B_p[index] = x_a_B_p

                        Delta_mix = [Delta_aA_dB[index], Delta_aB_dA[index], Delta_aA_dA[index], Delta_aB_dB[index]]
                        rho_mix = [rho_d_A_m[0], rho_a_A_m[0], rho_d_B_m[0], rho_a_B_m[0]]
                        sol = optimize.root(fXmix, XMix_guess, (rho_mix, Delta_mix))
                        [x_d_A_m, x_a_A_m, x_d_B_m, x_a_B_m] = sol.x
                        self.check_solution(sol, 'Mix')
                        X_d_A_m[index] = x_d_A_m
                        X_a_A_m[index] = x_a_A_m
                        X_d_B_m[index] = x_d_B_m
                        X_a_B_m[index] = x_a_B_m

                elif np.shape(invT)[0] == 1 and np.shape(x_A)[0] > 1:
                    X_d_A_p = np.zeros_like(x_A)
                    X_a_A_p = np.zeros_like(x_A)
                    X_d_B_p = np.zeros_like(x_A)
                    X_a_B_p = np.zeros_like(x_A)
                    X_d_A_m = np.zeros_like(x_A)
                    X_a_A_m = np.zeros_like(x_A)
                    X_d_B_m = np.zeros_like(x_A)
                    X_a_B_m = np.zeros_like(x_A)

                    # Solve once, never changes with xA, only with 1 / T
                    sol = optimize.root(fXpure, XAPure_guess, (rho_A_p, Delta_aA_dA[0]))
                    [x_d_A_p, x_a_A_p] = sol.x
                    self.check_solution(sol, 'A Pure')
                    sol = optimize.root(fXpure, XBPure_guess, (rho_B_p, Delta_aB_dB[0]))
                    [x_d_B_p, x_a_B_p] = sol.x
                    self.check_solution(sol, 'B Pure')
                    XMix_guess = [x_d_A_p, x_a_A_p, x_d_B_p, x_a_B_p]
                    for index, value in enumerate(x_A):
                        # Assume that we are moving from low to high x_A. Then we can use previous solution to seed the initial guess of the next solution
                        X_d_A_p[index] = x_d_A_p
                        X_a_A_p[index] = x_a_A_p
                        X_d_B_p[index] = x_d_B_p
                        X_a_B_p[index] = x_a_B_p
                        
                        Delta_mix = [Delta_aA_dB[0], Delta_aB_dA[0], Delta_aA_dA[0], Delta_aB_dB[0]]
                        rho_mix = [rho_d_A_m[index], rho_a_A_m[index], rho_d_B_m[index], rho_a_B_m[index]]
                        sol = optimize.root(fXmix, XMix_guess, (rho_mix, Delta_mix))
                        [x_d_A_m, x_a_A_m, x_d_B_m, x_a_B_m] = sol.x
                        self.check_solution(sol, 'Mix')
                        X_d_A_m[index] = x_d_A_m
                        X_a_A_m[index] = x_a_A_m
                        X_d_B_m[index] = x_d_B_m
                        X_a_B_m[index] = x_a_B_m
                        XMix_guess = sol.x
                else:
                    # Each xA and xB and T are linked. Let's iterate on T
                    sol = optimize.root(fXpure, XAPure_guess, (rho_A_p, Delta_aA_dA[0]))
                    [x_d_A_p, x_a_A_p] = sol.x
                    self.check_solution(sol, 'A Pure')
                    sol = optimize.root(fXpure, XBPure_guess, (rho_B_p, Delta_aB_dB[0]))
                    [x_d_B_p, x_a_B_p] = sol.x
                    self.check_solution(sol, 'B Pure')
                    X_d_A_p = x_d_A_p
                    X_a_A_p = x_a_A_p
                    X_d_B_p = x_d_B_p
                    X_a_B_p = x_a_B_p

                    # Calculate the mixture X_a and X_d
                    XMix_guess = [0.5, 0.5, 0.5, 0.5]
                    Delta_mix = [Delta_aA_dB[0], Delta_aB_dA[0], Delta_aA_dA[0], Delta_aB_dB[0]]
                    rho_mix = [rho_d_A_m[0], rho_a_A_m[0], rho_d_B_m[0], rho_a_B_m[0]]
                    sol = optimize.root(fXmix, XMix_guess, (rho_mix, Delta_mix))
                    [x_d_A_m, x_a_A_m, x_d_B_m, x_a_B_m] = sol.x
                    self.check_solution(sol, 'Mix')
                    X_d_A_m = x_d_A_m
                    X_a_A_m = x_a_A_m
                    X_d_B_m = x_d_B_m
                    X_a_B_m = x_a_B_m

                # Compute the terms multiplied to N_a_A and N_d_A only if they are not zero.
                N_a_A_term = np.where((self.N_a_A == 0), 0, self.N_a_A * (np.log(X_a_A_m / X_a_A_p) + (X_a_A_p - 1)/2) )
                N_d_A_term = np.where((self.N_d_A == 0), 0, self.N_d_A * (np.log(X_d_A_m / X_d_A_p) + (X_d_A_p - 1)/2) )

                # Compute the terms multiplied to N_a_B and N_d_B only if they are not zero.
                N_a_B_term = np.where((self.N_a_B == 0), 0, self.N_a_B * (np.log(X_a_B_m / X_a_B_p) + (X_a_B_p - 1)/2) )
                N_d_B_term = np.where((self.N_d_B == 0), 0, self.N_d_B * (np.log(X_d_B_m / X_d_B_p) + (X_d_B_p - 1)/2) )

                ln_y_A = N_a_A_term + N_d_A_term + rA * (rho_a_A_m * (1 - X_a_A_m)/2 + rho_d_A_m * (1 - X_d_A_m)/2 + rho_a_B_m * (1 - X_a_B_m)/2 + rho_d_B_m * (1 - X_d_B_m)/2)

                ln_y_B = N_a_B_term + N_d_B_term + rB * (rho_a_A_m * (1 - X_a_A_m)/2 + rho_d_A_m * (1 - X_d_A_m)/2 + rho_a_B_m * (1 - X_a_B_m)/2 + rho_d_B_m * (1 - X_d_B_m)/2)
                
                return ln_y_A, ln_y_B
        
class TrainedModel():
    def __init__(self, model_spec, df, df_vol, Tmat, temp_exponent, df_antoine, Omat = None, solute_smiles = 'Solute SMILES', solvent_smiles = 'Solvent SMILES'):
        # df is the dataframe of result parameters
        # A, P, Alpha, and D are dataframes of each parameter, where the appropriate row can be accessed with the canonical smiles
        self.A = df.filter(like = 'ua_')
        self.P = df.filter(like = 'uaa_')
        self.Alpha = df.filter(like = 'alpha_')
        self.D = df.loc[:, df.columns.str.contains('acceptor') | df.columns.str.contains('donor')]
        self.Q = df.filter(like = 'q_')
        self.molecular_param = df_vol.set_index('Canonical SMILES') # Searchable dataframe of molecular parameters such as volume and surface area, make the smiles the index for easier searching
        self.Antoine_param = df_antoine
        self.temp_mat = Tmat # This is the trained temperature matrix as a numpy array
        self.dist_mat = Omat # This is the trained distance combining matrix as numpy array, only assigns if provided
        self.temp_exponent = temp_exponent
        self.solute_smiles = solute_smiles
        self.solvent_smiles = solvent_smiles

        self.v0 = 0.01517 # the -CH2- reference volume used by Abrams and Prausnitz (m3/kmol)
        self.s0 = 2.5 * 10**8 # the -CH2- reference area used by Abrams and Prausnitz (m2/kmol)
        self.z = 10 # coordination number used by Abrams and Prausnitz

        self.combinatorial_label, self.residual_label, self.association_label = model_spec

    def _filter_chemical_parameters(self, smiles):
        # select a chemical based on the smiles and return the parameters (molecular and regressed)
        A_select = self.A.loc[smiles]
        P_select = self.P.loc[smiles]
        Alpha_select = self.Alpha.loc[smiles]
        D_select = self.D.loc[smiles]
        Q_select = self.Q.loc[smiles]
        mol_param_select = self.molecular_param.loc[smiles]

        return A_select, P_select, Alpha_select, D_select, Q_select, mol_param_select
    
    def _activity(self, x, invT):
        # Combine the functional forms
        if self.association_label != 'none':
            model = tuple(a + b + c for a, b, c in zip(self.combinatorial(x, invT, 1), self.residual(x, invT, 1), self.association(x, invT, 1)))
        else:
            model = tuple(a + b for a, b in zip(self.combinatorial(x, invT, 1), self.residual(x, invT, 1)))
        
        # Should return two values, ln A and ln B
        return model
    
    def _numerical_second_deriv(self, x, invT):
        # For use when analytical second deriative not possible.
        ln_A, ln_B = self._activity(x, invT)
        gMix = x[0] * (ln_A + np.log(x[0])) + x[1] * (ln_B + np.log(x[1]))
        # Find nan values and replace with 0 # edges
        gMix = np.nan_to_num(gMix)
        dx = x[0][1] - x[0][0]

        d2G = np.zeros_like(gMix)
        # 3 pt forward
        d2G[0] = (-30 * gMix[0] + 16 * gMix[1] - gMix[2]) / (12*dx**2)
        d2G[1] = (-30 * gMix[1] + 16 * gMix[2] - gMix[3]) / (12*dx**2)

        # Central 5 pt stencil
        for i in range(2, np.size(gMix) - 2):
            d2G[i] = (-gMix[i-2] + 16*gMix[i-1] - 30*gMix[i] + 16*gMix[i+1] - gMix[i+2]) / (12*dx**2)
        
        # 3 pt backward
        d2G[-2] = (-30 * gMix[-2] + 16 * gMix[-3] - gMix[-4]) / (12*dx**2)
        d2G[-1] = (-30 * gMix[-1] + 16 * gMix[-2] - gMix[-3]) / (12*dx**2)

        return d2G

    def _numerical_third_deriv(self, x, invT):
        # For use with Wertheim where the analytical is not possible. use small sizes of x for best ersult
        # The free energy should also be calculated from activity coefficients in the following way
        # G/RT = xA * ln(gammaA) + xB * ln(gammaB)
        ln_A, ln_B = self._activity(x, invT)
        gMix = x[0] * (ln_A + np.log(x[0])) + x[1] * (ln_B + np.log(x[1]))
        # Find nan values and replace with 0 - these are the zero comp points
        gMix = np.nan_to_num(gMix)
        dx = x[0][1] - x[0][0]

        d3G = np.zeros_like(gMix)

        # 3 pt forward difference
        d3G[0] = (-5 * gMix[0] + 18 * gMix[1] - 24 * gMix[2] + 14 * gMix[3] - 3 * gMix[4]) / (2 * dx**3)
        d3G[1] = (-5 * gMix[1] + 18 * gMix[2] - 24 * gMix[3] + 14 * gMix[4] - 3 * gMix[5]) / (2 * dx**3)

        # 5 pt stencil
        for i in range (2, np.size(gMix) - 2):
            d3G[i] = (gMix[i - 2] - 2 * gMix[i - 1] + 2 * gMix[i + 1] - gMix[i + 2]) / (2 * dx**3)
        
        # 3 pt backward difference
        d3G[-2] = (5 * gMix[-2] - 18 * gMix[-3] + 24 * gMix[-4] - 14 * gMix[-5] + 3 * gMix[-6]) / (2 * dx**3)
        d3G[-1] = (5 * gMix[-1] - 18 * gMix[-2] + 24 * gMix[-3] - 14 * gMix[-4] + 3 * gMix[-5]) / (2 * dx**3)

        return d3G
    
    def _model_param(self, x, invT):
        model = self.residual(x, invT, 'param')
        
        return model

    def _antoine(self, row, T):
        # Not really Antoine's equation but rather DIPPR correlations.
        # Pass in the row of the chemical form the vapor pressure dataframe.
        # The desired columns are 'A', 'B', 'C', 'D', 'E', 'Equation'
        eqn_num = row['Equation']
        if row['Equation'] == 'antoine':
            Psat = 10 ** (row['A'] - row['B'] / (row['C'] + T))
        elif row['Equation'] == 101:
            # DIPPR equation 101 - results in Pa, convert to bar
            Psat = 10**-5 * np.exp(row['A'] + row['B'] / T + row['C'] * np.log(T) + row['D'] * T ** row['E'])
        elif row['Equation'] == 'extended_101':
            # Aspen Plus extended 101 equation - results in Pa, convert to bar
            Psat = 10**-5 * np.exp(row['A'] + row['B'] / (row['C'] + T) + row['D'] * T + row['E'] * np.log(T) + row['F'] * T ** row['G'])
        else:
            print(f'Unrecognized equation {eqn_num}')
        return Psat
    
    def create_model(self, *args):
        if len(args) == 2:
            print('Received two smiles string, generating binary solution model')
            smilesA = args[0]
            smilesB = args[1]

            # Create the appropriate model depending on specifications
            A_A, P_A, Alpha_A, D_A, Q_A, mol_param_A = self._filter_chemical_parameters(smilesA)
            A_B, P_B, Alpha_B, D_B, Q_B, mol_param_B = self._filter_chemical_parameters(smilesB)

            A = (A_A, A_B)
            P = (P_A, P_B)
            Alpha = (Alpha_A, Alpha_B)
            D = (D_A, D_B)
            Q = (Q_A, Q_B)
            mol_param = (mol_param_A, mol_param_B)

            # Initialize the models with the correct functional forms
            combinatorial = Combinatorial(2, self.combinatorial_label, mol_param)
            residual = Residual(2, self.residual_label, A, P, Q, Alpha, mol_param, self.temp_mat, self.dist_mat, self.temp_exponent)
            association = Association(2, self.association_label, D, mol_param)

            # Create the desired model
            self.combinatorial = combinatorial.model
            self.residual = residual.model
            self.association = association.model
        else:
            raise ValueError('Currently, only binary systems are supported.')

        # Store for the over-writing of the residual model
        self.mol_param = mol_param

        return self
    
    def free_energy(self, x, invT):
        # There is an analytical form of the free energy. This is useful for testing purposes.
        '''
        if self.association_label != 'none':
            gMix = self.combinatorial(x, invT, 0) + self.residual(x, invT, 0) + self.association(x, invT, 0)
        else:
            gMix = self.combinatorial(x, invT, 0) + self.residual(x, invT, 0)
        '''
        # The free energy can also be calculated from activity coefficients in the following way
        # G/RT = xA * ln(gammaA * xA) + xB * ln(gammaB * xB)
        ln_A, ln_B = self._activity(x, invT)
        gMix = x[0] * (ln_A + np.log(x[0])) + x[1] * (ln_B + np.log(x[1]))
    
        return gMix

    def solution(self, smilesA, smilesB):
        # Generate the solution of A and B
        # Get antoine parameters
        try:
            self.Antoine_A = self.Antoine_param.loc[smilesA]
            self.Antoine_B = self.Antoine_param.loc[smilesB]
        except:
            self.Antoine_A = np.nan
            self.Antoine_B = np.nan

        # Create the model
        self.create_model(smilesA, smilesB)

        self.smilesA = smilesA
        self.smilesB = smilesB

        self.num_components = 2

        return self
    
    def plot_idac(self, df, activity_label, T_low, T_high):
        # Plot the calculated activity coefficients (full including combinatorial) against the literature results (found in df).
        # Get the right pairs and temperatures
        A_inf_cond = (df[self.solute_smiles] == self.smilesA) & (df[self.solvent_smiles] == self.smilesB)
        B_inf_cond = (df[self.solvent_smiles] == self.smilesA) & (df[self.solute_smiles] == self.smilesB)
        
        # These are the data values of ln y A and B inf at different temperatures
        try:
            ln_y_A_inf = df[A_inf_cond][activity_label].to_numpy()
            ln_y_B_inf = df[B_inf_cond][activity_label].to_numpy()
            # Report inverse temperature
            A_inf_temp = 1 / df[A_inf_cond]['Temp (K)'].to_numpy()
            B_inf_temp = 1 / df[B_inf_cond]['Temp (K)'].to_numpy()
        except:
            print(f'No data found for {self.smilesA} and {self.smilesB}')
            ln_y_A_inf = []
            ln_y_B_inf = []
            A_inf_temp = []
            B_inf_temp = []

        # Set up the compositions to evaluate at
        A_inf = (np.array([0]), np.array([1]))
        B_inf = (np.array([1]), np.array([0]))

        # Evaluate over a range of temperatures. Span 273.15 K to 473.15 K (0 to 200 C)
        A_inf_temp_eval = np.linspace(1/T_high, 1/T_low, 100)
        B_inf_temp_eval = np.linspace(1/T_high, 1/T_low, 100)

        ln_y_A_hat_inf, _ = self._activity(A_inf, A_inf_temp_eval)
        _, ln_y_B_hat_inf = self._activity(B_inf, B_inf_temp_eval)
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(8, 8))

        # Plot predicted ln gamma A (red dashed line)
        ax.plot(
            A_inf_temp_eval, 
            ln_y_A_hat_inf,
            color='red',
            linestyle = '--',  # Dashed line
            linewidth=2,      # Line thickness
            label=rf"Prediction of {self.smilesA} $\ln \gamma^\infty$"
        )

        # Plot predicted ln gamma B (blue dashed line)
        ax.plot(
            B_inf_temp_eval, 
            ln_y_B_hat_inf, 
            color='blue', 
            linestyle = '--',  # Dashed line
            linewidth=2,      # Line thickness
            label=rf"Prediction of {self.smilesB} $\ln \gamma^\infty$"
        )

        # Plot true ln gamma A (red circles with border)
        ax.scatter(
            A_inf_temp, 
            ln_y_A_inf, 
            facecolors='none',  # Makes the inside of the circle transparent
            edgecolors='red',   # Red border
            linewidth=2,        # Border thickness
            marker='o',         # 'o' stands for circle marker
            label=rf"Data of {self.smilesA} $\ln \gamma^\infty$"
        )

        # Plot true ln gamma B (blue circles with border)
        ax.scatter(
            B_inf_temp, 
            ln_y_B_inf, 
            facecolors='none',  # Transparent circle
            edgecolors='blue',  # Blue border
            linewidth=2,        # Border thickness
            marker='o',         # 'o' for circle marker
            label=rf"Data of {self.smilesB} $\ln \gamma^\infty$"
        )

        # Set ax limit to either be -1 to 5 or min and max of data, whatever is tighter
        ax.set_ylim(-2, 10)
        ln_y = np.concatenate((ln_y_A_inf, ln_y_B_inf))
        if np.min(ln_y) > -2 or np.max(ln_y) < 10:
            ax.set_ylim(np.min(ln_y - 1), np.max(ln_y + 1))

        ax.legend()

        return fig
    
    def calc_param(self, x, T):
        # Calculate the NRTL/UNIQUAC/Wilson parameters given a model input
        invT = 1 / T
        param = self._model_param(x, invT)

        return param
    
    def calc_activity(self, x, T):
        # Calculate the activity coefficients of the model as a function of composition
        invT = 1 / T
        ln_y_A, ln_y_B = self._activity(x, invT)

        return ln_y_A, ln_y_B
    
    def calc_activity_parts(self, x, T):
        # Calculate each contribution to the activity coefficient
        invT = 1 / T
        if self.association_label != 'none':
            comb = self.combinatorial(x, invT, 1)
            res = self.residual(x, invT, 1)
            assoc = self.association(x, invT, 1)
            activity = {'combinatorial': comb, 'residual': res, 'association': assoc}
            return activity
        else:
            comb = self.combinatorial(x, invT, 1)
            res = self.residual(x, invT, 1)
            activity = {'combinatorial': comb, 'residual': res}
            return activity

    def calc_VLE(self, T, p, npts, *args):
        # Calculate the VLE curves of species A and B
        # This is designed to pass in some set of known VLE data (Aspen Plus generated or otherwise). The model will evaluate the VLE curves at the same points as the data.
        # Depending on whether T or p are specified (not none), calculate a P-x-y (former) or T-x-y (latter) plot
        aspen_data = args[0]
        if T is not None:
            # Calculate p-x-y plot
            invT = 1 / T # Inverse temperature
            if aspen_data is not None:
                # Find the same points as the aspen dataset to compare deviations
                xA = np.copy(aspen_data['x'])
            else:
                xA = np.linspace(0, 1, npts)
            xB = 1 - xA
            x = (xA, xB)
            ln_y_A, ln_y_B = self._activity(x, invT) # the model should have already been created
            gamma_A = np.exp(ln_y_A)
            gamma_B = np.exp(ln_y_B)
            psat_A = self._antoine(self.Antoine_A, T)
            psat_B = self._antoine(self.Antoine_B, T)

            P_bubble = psat_A * xA * gamma_A + psat_B * xB * gamma_B
            P_dew = 1 / (xA / (psat_A * gamma_A) + xB / (psat_B * gamma_B))

            return xA, P_bubble, P_dew
            
        elif p is not None:
            # Calculate the T-x-y plot
            # Conduct the optimization in inverse temperature (more stable to large changes in temp)
            def bp(T, p, antoine_param):
                root = self._antoine(antoine_param, T) - p
                return root
            if aspen_data is not None:
                Tbp_B_guess = aspen_data['T_bubble'][0]
                Tbp_A_guess = aspen_data['T_bubble'][-1]
            else:
                Tbp_B_guess = 298.15
                Tbp_A_guess = 298.15
            
            Tbp_A = optimize.root(bp, x0 = Tbp_A_guess, args = (p, self.Antoine_A), tol = 1e-12, method = 'lm')
            Tbp_B = optimize.root(bp, x0 = Tbp_B_guess, args = (p, self.Antoine_B), tol = 1e-12, method = 'lm')
            print(f'Tbp finder for A exited with {Tbp_A.message}')
            print(f'Tbp finder for B exited with {Tbp_B.message}')
            if not Tbp_A.success:
                # Try with higher initial guess
                print(f'Tbp finder for A exited with {Tbp_A.message}')
                Tbp_A = optimize.root(bp, x0 = 473.15, args = (p, self.Antoine_A), tol = 1e-12, method = 'lm')
            if not Tbp_B.success:
                # Try with higher initial guess
                print(f'Tbp finder for B exited with {Tbp_B.message}')
                Tbp_B = optimize.root(bp, x0 = 473.15, args = (p, self.Antoine_B), tol = 1e-12, method = 'lm')
            
            print(f'Boiling point of A is {Tbp_A.x} and boiling point of B is {Tbp_B.x}')

            # The guess is from xA = 0 to xA = 1. So should be from bp B to bp A
            if aspen_data is not None:
                # Find the same points as the aspen dataset to compare deviations
                xA = np.copy(aspen_data['x'])
                # Make Tguess the same as Aspen plus
                if ~np.isnan(aspen_data['T_bubble']).any():
                    invT_bubble_guess = 1 / aspen_data['T_bubble']
                else:
                    invT_bubble_guess = 1 / np.linspace(Tbp_B.x, Tbp_A.x, npts)
                if ~np.isnan(aspen_data['T_dew']).any():
                    invT_dew_guess = 1 / aspen_data['T_dew']
                else:
                    invT_dew_guess = 1 / np.linspace(Tbp_B.x, Tbp_A.x, npts)
            else:
                invT_bubble_guess = 1 / np.linspace(Tbp_B.x, Tbp_A.x, npts)
                invT_dew_guess = 1 / np.linspace(Tbp_B.x, Tbp_A.x, npts)
                xA = np.linspace(0, 1, npts)
            
            # We need to flatten the temp arrays
            invT_bubble_guess = invT_bubble_guess.flatten()
            invT_dew_guess = invT_dew_guess.flatten()

            xB = 1 - xA
            x = (xA, xB)

            def dew_pt(invT, x, p):
                xA, xB = x
                #invT = 1 / T
                T = 1 / invT
                ln_gamma_A, ln_gamma_B = self._activity(x, invT) # the model should have already been created
                gamma_A = np.exp(ln_gamma_A)
                gamma_B = np.exp(ln_gamma_B)
                psat_A = self._antoine(self.Antoine_A, T)
                psat_B = self._antoine(self.Antoine_B, T)
                res = 1 - p * (xA / (gamma_A * psat_A) + xB / (gamma_B * psat_B))
                return res
            
            def bubble_pt(invT, x, p):
                xA, xB = x
                #invT = 1 / T
                T = 1 / invT
                ln_gamma_A, ln_gamma_B = self._activity(x, invT) # the model should have already been created
                gamma_A = np.exp(ln_gamma_A)
                gamma_B = np.exp(ln_gamma_B)
                psat_A = self._antoine(self.Antoine_A, T)
                psat_B = self._antoine(self.Antoine_B, T)
                res = 1 - (gamma_A * xA * psat_A + gamma_B * xB * psat_B) / p
                return res
            
            # Get the mean absolute deviation of the initial guess
            print(f'Mean absolute deviation of bubble guess is : {np.mean(np.abs(bubble_pt(invT_bubble_guess, x, p)))}')
            print(f'Mean absolute deviation of dew guess is: {np.mean(np.abs(dew_pt(invT_dew_guess, x, p)))}')

            invT_bubble = optimize.root(bubble_pt, invT_bubble_guess, args = (x, p), method = 'lm')
            print(f'Bubble calculation exited with message: {invT_bubble.message}. Success status: {invT_bubble.success}, mean absolute deviation: {np.mean(np.abs(invT_bubble.fun))}')
            invT_dew = optimize.root(dew_pt, invT_dew_guess, args = (x, p), method = 'lm')
            print(f'Dew calculation exited with message: {invT_dew.message}. Success status: {invT_bubble.success}, mean absolute deviation: {np.mean(np.abs(invT_dew.fun))}')

            if not invT_bubble.success and invT_dew.success:
                # Redo the bubble calculation with dew point as guess
                invT_bubble = optimize.root(bubble_pt, invT_dew.x, args = (x, p), method = 'lm')
                print(f'Redo bubble calculation exited with message: {invT_bubble.message}')
            elif invT_bubble.success and not invT_dew.success:
                # Redo the dew calculation with bubble point as guess
                invT_dew = optimize.root(dew_pt, invT_bubble.x, args = (x, p), method = 'lm')
                print(f'Redo dew calculation exited with message: {invT_dew.message}')
            elif invT_bubble.success and invT_dew.success:
                print(f'Both bubble and dew point calculations converged successfully')
            else:
                print(f'Neither bubble or dew point calculations converged successfully')

            # Calculate the activity coefficients
            T_bubble = 1 / invT_bubble.x
            T_dew = 1 / invT_dew.x
            
            ln_y_A, ln_y_B = self._activity(x, 1 / T_dew)

            return xA, T_bubble, T_dew
    
    def calc_LLE(self, T_range, xpts):
        # Calculate the LLE by enforcing the two strict criteria: tangent plane and minimal energy
        # This means that for each component, mu_I = mu_II --> ln(activity) + ln(x_I) = ln(activity) + ln(x_II)
        # Also subject to minimization criteria of free energy of mixing: G_mix = x_I * G_mix_I + x_II * G_mix_II
        # Calculation of x_I via lever rule: x_I = (x_A - x_AII) / (x_AI - x_AII)
        # The lever rule is used to calculate the mole fractions of the two phases

        # At each temperature, calculate the numerical second and third derivatives of the free energy of mixing
        # If the energy diagram has a local maximum, then it is unstable. Use that as the guess for the energy minimization calculation
        # If no maximum exists, it does not mean the system is stable, there could be an inflection point. Use that as a guess for the energy minimization
        x_list = [np.linspace(0, 1, xpts), np.linspace(1, 0, xpts)]

        def LLE_energy(x, xA_inlet, T):
            # Calculate the free energy of mixing
            invT = 1 / T
            xA_I, xA_II = x
            xB_I = 1 - xA_I
            xB_II = 1 - xA_II
            x_I = (xA_inlet - xA_II) / (xA_I - xA_II)
            x_II = 1 - x_I
            G_mix_I = self.free_energy([np.array([xA_I]), np.array([xB_I])], invT)
            G_mix_II = self.free_energy([np.array([xA_II]), np.array([xB_II])], invT)
            
            G_mix = x_I * G_mix_I + x_II * G_mix_II

            return G_mix
        
        def tangent_plane(x, T):
            # Calculate mu of each species in each phase
            xA_I, xA_II = x
            xB_I = 1 - xA_I
            xB_II = 1 - xA_II
            ln_y_A_I, ln_y_B_I = self._activity([np.array([xA_I]), np.array([xB_I])], 1 / T)
            ln_y_A_II, ln_y_B_II = self._activity([np.array([xA_II]), np.array([xB_II])], 1 / T)

            y_A_I = np.exp(ln_y_A_I)
            y_B_I = np.exp(ln_y_B_I)
            y_A_II = np.exp(ln_y_A_II)
            y_B_II = np.exp(ln_y_B_II)

            mu_A_res = y_A_I * xA_I - y_A_II * xA_II
            mu_B_res = y_B_I * xB_I - y_B_II * xB_II

            return np.array([mu_A_res[0], mu_B_res[0]])

        x_I_list = []
        x_II_list = []
        T_lle = []
        
        for T in T_range:
            # Identify the most positive value in the free energy curve. if this value is greater than 0, then the system is unstable
            gMix = self.free_energy(x_list, np.array([1/T]))
            gMix = np.nan_to_num(gMix, nan = 0)

            x_initial_guess_idx = np.argmax(gMix) # Scalar value
            x_max = x_list[0][x_initial_guess_idx]
            x_initial_guess_idx = np.array([x_initial_guess_idx]) # Make array to have later code work

            # If maximum is less than zero, than not a local maximum, find the third derivative inflection points
            if x_max <= 1e-12:
                # No maximum in energy detected, check third derivative for inflection points
                # Third derivative of the free energy of mixing
                d3G = self._numerical_third_deriv(x_list, np.array([1/T]))

                # Detect zero crossings
                x_initial_guess_idx = np.where(np.diff(np.sign(d3G)) != 0)[0]

                # Remove if any of these points are 3 away from the end: these are numerical errors
                x_initial_guess_idx = x_initial_guess_idx[(x_initial_guess_idx > 3) & (x_initial_guess_idx < xpts - 3)]

            if x_initial_guess_idx.size == 0:
                print(f'No zero points detected for T = {T}')
                continue
            elif x_initial_guess_idx.size > 1 and len(x_I_list) > 0:
                x_initial_guess = (x_I_list[-1] + x_II_list[-1]) / 2
                print(f'Multiple inflection points determined at {x_initial_guess_idx} but previously converged. Using guess {x_initial_guess}')
            elif x_initial_guess_idx.size > 1:
                print(f'Multiple inflection points determined at {x_initial_guess_idx}. Using guess {x_list[0][x_initial_guess_idx[1]]}')
                x_initial_guess = x_list[0][x_initial_guess_idx[1]]
            else:
                x_initial_guess = x_list[0][x_initial_guess_idx[0]]
            # bounds for the optimization
            bounds = [(1e-12, x_initial_guess), (x_initial_guess, 1-1e-12)]
            x0 = np.array([x_initial_guess - 1e-3, x_initial_guess + 1e-3])

            # Optimizie using L-BFGS-B
            sol = optimize.minimize(LLE_energy, x0 = x0, args = (x_initial_guess, np.array([T])), method = 'L-BFGS-B', bounds = bounds, tol = 1e-12)
            dif_x = (sol.x[0] - sol.x[1])**2
            # sol.x should be close to the stable point. We can then find the root of the tangent plane equation to verify
            mu_res = tangent_plane(sol.x, np.array([T]))
            # If the errors are okay, then we can add the solution to the list
            print(f'Optimization result is {sol.x} with values {sol.fun}. Tangent plane result is {mu_res}')
            
            # Refine using equipotential criteria
            sol = optimize.root(tangent_plane, sol.x, args = (np.array([T])), method = 'lm', tol = 1e-12)
            dif_x = (sol.x[0] - sol.x[1])**2 # 1e-10 error corresponds to around 5 decimal place difference
            # If sucessful, append. Else, data not entered
            if sol.success and dif_x > 1e-10:
                x_I_list.append(sol.x[0])
                x_II_list.append(sol.x[1])
                T_lle.append(T)

        return x_I_list, x_II_list, T_lle
    