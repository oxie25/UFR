"""

Oliver Xie, Olsen Group at Massachusetts Institute of Technology, 2025
@author oxie25

"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# This file contains all the pytorch layers used for training on IDAC datasets
# No models take average across number of parameter dimensions.
# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
# Note: T is actually 1/T
torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True

def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)

# Linear combination with positive (>0 <1) weighting using softmax. Softmax behavior is such that the rows of the linear_comb_matrix sum to 1, and are probabilities. This fits our desired outcome, where for a given parameter, we must not 'over-assign' it but instead distribute its data across temperature relations
class TempSoftmaxWeight(nn.Module):
    def __init__(self):
        super(TempSoftmaxWeight, self).__init__()
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x, T_mat):
        linear_comb_matrix = self.softmax(T_mat) # For each row, apply softmax which distributes the parameter across the temperature dependencies in such a way that the total assignment is 1. This means that there cannot be cases where a parameter is 'used' in all temp dependencies equally
        transformed_output = torch.matmul(x, linear_comb_matrix) # We matrix multiply the input array (Ndata x Nparam) against the linear combination matrix
        return transformed_output

# For learning different mixtures of distance metric    
# Linear combination with positive (>0 <1) weighting using softmax. Softmax behavior is such that the rows of the linear_comb_matrix sum to 1, and are probabilities. This fits our desired outcome, where for a given parameter, we must not 'over-assign' it but instead distribute its data across temperature relations
class OperationSoftmaxWeight(nn.Module):
    def __init__(self):
        super(OperationSoftmaxWeight, self).__init__()
        self.softmax = nn.Softmax(dim = 1) # Apply in 1st dimension, we will unsqueeze/resqueeze dimensions appropriately

    def forward(self, x, O_mat):
        # x should have size Nd x Nparam x 2

        linear_comb_matrix = self.softmax(O_mat) # For each row, apply softmax which distributes the parameter across the temperature dependencies in such a way that the total assignment is 1. This means that there cannot be cases where a particular distancing operation is used 'more' than linearly added to 1 # Shape Nparam x 2
        linear_comb_matrix_expanded = linear_comb_matrix.unsqueeze(0) # Shape 1 x Nparam x 2
        
        transformed_output = torch.sum(x * linear_comb_matrix_expanded, dim=-1)  # Nd x Nparam

        return transformed_output

# Have the Wertheim interaction term be its own class that's trainable
class Wertheim(nn.Module):
    def __init__(self):
        super(Wertheim, self).__init__()
        self.softplus = nn.Softplus()

    def forward(self, Delta, i, j, invT, r, q, N, rho):
        # Flatten
        r = torch.flatten(r) # Needs to be flat
        invT = torch.flatten(invT) # Needs to be flat
        # Calculate the Wertheim association term
        rhoAap = rho[:, 0]
        rhoAdp = rho[:, 1]
        rhoBam = rho[:, 2]
        rhoBdm = rho[:, 3]
        
        NAa = N[:, 0]
        NAd = N[:, 1]

        # Calculate the interaction term
        dref = 0.034 * (torch.exp(1960 * invT) - 1) # not the version in the overleaf doc
        D = self.softplus(Delta)
        D_AaAd = D[i, 0] * D[i, 1] * dref  # delta Aa * delta Ad * dref
        D_BaBd = D[j, 0] * D[j, 1] * dref  # delta Ba * delta Bd * dref
        D_AaBd = D[i, 0] * D[j, 1] * dref  # delta Aa * delta Bd * dref
        D_AdBa = D[j, 0] * D[i, 1] * dref  # delta Ba * delta Ad * dref

        # The equations for XaBm and XdBm are:
        bb_a = D_BaBd * rhoBam
        bb_d = D_BaBd * rhoBdm
        
        # If bb_d and bb_a are 0, then XaBm and XdBm respectively are one
        # Create masks for bb_a == 0 and bb_d == 0
        mask_bb_a_zero = rhoBam == 0
        mask_bb_d_zero = rhoBdm == 0
                
        # Exact calculation - only one root is positive
        adj_bb_a = bb_a.masked_fill(mask_bb_a_zero, 1) #  We need to do this or else it will propagate incorrectly
        adj_bb_d = bb_d.masked_fill(mask_bb_d_zero, 1) #  We need to do this or else it will propagate incorrectly
        XaBm_temp = torch.sqrt(adj_bb_a**2 - 2 * adj_bb_a * (adj_bb_d - 1) + (adj_bb_d + 1)**2) / (2*adj_bb_a) + 0.5 - rhoBdm/(2*rhoBam) - 1 / (2*adj_bb_a)
        XdBm_temp = torch.sqrt(adj_bb_d**2 - 2 * adj_bb_d * (adj_bb_a - 1) + (adj_bb_a + 1)**2) / (2*adj_bb_d) + 0.5 - rhoBam/(2*rhoBdm) - 1 / (2*adj_bb_d)
        
        XaBm = torch.where(mask_bb_d_zero, 1, torch.where(mask_bb_a_zero, 1 / adj_bb_d, XaBm_temp))
        XdBm = torch.where(mask_bb_a_zero, 1, torch.where(mask_bb_d_zero, 1 / adj_bb_a, XdBm_temp))

        # The equations for XaAp and XdAp are:
        aa_a = D_AaAd * rhoAap
        aa_d = D_AaAd * rhoAdp

        # Create masks for aa_a == 0 and aa_d == 0
        mask_aa_a_zero = rhoAap == 0
        mask_aa_d_zero = rhoAdp == 0

        # Override the XaAp value
        adj_aa_a = aa_a.masked_fill(mask_aa_a_zero, 1) #  We need to do this or else it will propagate incorrectly
        adj_aa_d = aa_d.masked_fill(mask_aa_d_zero, 1) #  We need to do this or else it will propagate incorrectly
        XaAp_temp = torch.sqrt(adj_aa_a**2 - 2 * adj_aa_a * (adj_aa_d - 1) + (adj_aa_d + 1)**2) / (2*adj_aa_a) + 0.5 - rhoAdp/(2*rhoAap) - 1 / (2*adj_aa_a)
        XdAp_temp = torch.sqrt(adj_aa_d**2 - 2 * adj_aa_d * (adj_aa_a - 1) + (adj_aa_a + 1)**2) / (2*adj_aa_d) + 0.5 - rhoAap/(2*rhoAdp) - 1 / (2*adj_aa_d)

        XaAp = torch.where(mask_aa_d_zero, 1, torch.where(mask_aa_a_zero, 1 / adj_aa_d, XaAp_temp))
        XdAp = torch.where(mask_aa_a_zero, 1, torch.where(mask_aa_d_zero, 1 / adj_aa_a, XdAp_temp))

        # Calculate the remaining XaAm and XdAm terms
        XaAm = 1 / (1 + D_AaBd * rhoBdm * XdBm)
        XdAm = 1 / (1 + D_AdBa * rhoBam * XaBm)
        
        termAa = torch.where(NAa==0, 0, NAa * (torch.log(XaAm/XaAp) + (XaAp - 1)/2))
        termAd = torch.where(NAd==0, 0, NAd * (torch.log(XdAm/XdAp) + (XdAp - 1)/2))
        termB = r * (rhoBam * (1-XaBm)/2 + rhoBdm * (1-XdBm)/2)

        combined = termAa + termAd + termB # 1D

        return combined

class UNIQUAC(nn.Module):
    def __init__(self, A_initial, T_initial, O_initial, D_initial, association_layer, temp_exponents, device):
        super(UNIQUAC, self).__init__()
        self.A = nn.Parameter(torch.tensor(A_initial, dtype=torch.float32).to(device))  # Initialize A
        self.T = nn.Parameter(torch.tensor(T_initial, dtype=torch.float32).to(device))  # Initialize T here
        self.O = nn.Parameter(torch.tensor(O_initial, dtype=torch.float32).to(device))  # Initialize O here

        # For combining distance
        self.operationLayer = OperationSoftmaxWeight()
        # For combining temperature
        self.tempLayer = TempSoftmaxWeight()
        
        if association_layer == 'wertheim':
            self.associationLayer = Wertheim()
            self.D = nn.Parameter(torch.tensor(D_initial, dtype=torch.float32).to(device)) # Initialize D only if needed
        else:
            self.associationLayer = None
            print('No association term used')

        # Create an exponent tensor of shape (1, Ntemp) containing [0, 1, 2, ..., Ntemp-1]
        if T_initial.shape[1] == 1: # If ony one temp dim, then create only a single column which gets 1/T
            self.exponents = torch.ones(T_initial.shape[1]).unsqueeze(0).to(device)  # Shape (1, Ntemp)
        else:
            self.exponents = torch.tensor(temp_exponents).unsqueeze(0).to(device)  # Shape (1, Ntemp)

        # Generate the correct taylor series coefficients for higher degree temperature dependence
        if np.min(temp_exponents) < 0 and np.max(temp_exponents) == 1:
            # Negative temperature powers means that we are doing a Taylor series expansion in T. Calculate and multiply the correct coefficients
            taylor_coeffs = np.array([factorial(x) for x in np.absolute(temp_exponents - 1)])
            self.taylor_coeffs = torch.tensor(1 / taylor_coeffs, dtype=torch.float32).unsqueeze(0).to(device)  # Shape (1, Ntemp)
            print(f'Using Taylor series expansion with coefficients: {self.taylor_coeffs}')
        elif np.min(temp_exponents) == 0 and np.max(temp_exponents) > 0:
            # Positive temperature powers means that we are doing a Taylor series expansion in 1/T. Calculate and multiply the correct coefficients
            taylor_coeffs = np.array([factorial(x) for x in np.absolute(temp_exponents)])
            self.taylor_coeffs = torch.tensor(1 / taylor_coeffs, dtype = torch.float32).unsqueeze(0).to(device)
            print(f'Using Taylor series expansion with coefficients: {self.taylor_coeffs}')
        else:
            self.taylor_coeffs = None

    def forward(self, i, j, invT, r, qA, qB, N, rho):
        # A is the matrix Nchem x Nparam
        A_prod = torch.clamp(self.A[i, :], min = 0) * torch.clamp(self.A[j, :], min = 0) # Return an elementwise multiply of the clamp of each parameter where each row is a data entry and each column is the parameter
        A_dif = (self.A[i, :] - self.A[j, :])**2 # Return an elementwise squared difference where each row is a data entry and each column is the parameter
        A_gather = torch.stack((A_prod, A_dif), dim=2)

        A_i_prod_self = torch.clamp(self.A[i, :], min = 0) * torch.clamp(self.A[i, :], min = 0) # Return an elementwise multiply of the clamp of each parameter where each row is a data entry and each column is the parameter
        A_i_dif_self = (self.A[i, :] - self.A[i, :])**2 # Return an elementwise squared difference where each row is a data entry and each column is the parameter

        A_j_prod_self = torch.clamp(self.A[j, :], min = 0) * torch.clamp(self.A[j, :], min = 0) # Return an elementwise multiply of the clamp of each parameter where each row is a data entry and each column is the parameter
        A_j_dif_self = (self.A[j, :] - self.A[j, :])**2 # Return an elementwise squared difference where each row is a data entry and each column is the parameter

        A_i_gather_self = torch.stack((A_i_prod_self, A_i_dif_self), dim=2)
        A_j_gather_self = torch.stack((A_j_prod_self, A_j_dif_self), dim=2)

        # Choose the right ways of combining
        A_combined = self.operationLayer(A_gather, self.O)
        A_combined_temp = self.tempLayer(A_combined, self.T)

        A_i_combined = self.operationLayer(A_i_gather_self, self.O)
        p_i = self.tempLayer(A_i_combined, self.T)
        A_j_combined = self.operationLayer(A_j_gather_self, self.O)
        p_j = self.tempLayer(A_j_combined, self.T)

        # Use broadcasting to raise each element of T to the corresponding exponent
        Tmat = torch.pow(invT, self.exponents)  # Shape (Ndata, Ntemp)
        if self.taylor_coeffs is not None:
            Tmat = Tmat * self.taylor_coeffs

        u_ij = torch.sum( ( (A_combined_temp - p_j) * Tmat), dim=1, keepdim=True)
        u_ji = torch.sum( ( (A_combined_temp - p_i) * Tmat), dim=1, keepdim=True)
        
        result = qA * (1 + u_ji - torch.exp(-u_ij))

        # Use the Wertheim layer if the layer exists
        if self.associationLayer is not None:
            asc_term = self.associationLayer(self.D, i, j, invT, r, qA, N, rho)
            asc_term = asc_term[:, None] # Add another dimension
            result = asc_term + result
        
        return result
    
class mod_UNIQUAC(nn.Module):
    def __init__(self, A_initial, Q_initial, T_initial, O_initial, D_initial, association_layer, temp_exponents, device):
        super(mod_UNIQUAC, self).__init__()
        self.A = nn.Parameter(torch.tensor(A_initial, dtype=torch.float32).to(device))  # Initialize A
        self.Q = nn.Parameter(torch.tensor(Q_initial, dtype=torch.float32).to(device))  # Initialize P - this is the interaction correction factor
        self.T = nn.Parameter(torch.tensor(T_initial, dtype=torch.float32).to(device))  # Initialize T here
        self.O = nn.Parameter(torch.tensor(O_initial, dtype=torch.float32).to(device))  # Initialize O here

        # For combining distancing layers and learning them correctly
        self.operationLayer = OperationSoftmaxWeight()
        # For combining temperature
        self.tempLayer = TempSoftmaxWeight()

        self.relu = nn.ReLU()
        
        if association_layer == 'wertheim':
            self.associationLayer = Wertheim()
            self.D = nn.Parameter(torch.tensor(D_initial, dtype=torch.float32).to(device)) # Initialize D only if needed
        else:
            self.associationLayer = None
            print('No association term used')

        # Create an exponent tensor of shape (1, Ntemp) containing [0, 1, 2, ..., Ntemp-1]
        if T_initial.shape[1] == 1: # If ony one temp dim, then create only a single column which gets 1/T
            self.exponents = torch.ones(T_initial.shape[1]).unsqueeze(0).to(device)  # Shape (1, Ntemp)
        else:
            self.exponents = torch.tensor(temp_exponents).unsqueeze(0).to(device)  # Shape (1, Ntemp)

        if np.min(temp_exponents) < 0 and np.max(temp_exponents) == 1:
            # Negative temperature powers means that we are doing a Taylor series expansion in T. Calculate and multiply the correct coefficients
            taylor_coeffs = np.array([factorial(x) for x in np.absolute(temp_exponents - 1)])
            self.taylor_coeffs = torch.tensor(1 / taylor_coeffs, dtype=torch.float32).unsqueeze(0).to(device)  # Shape (1, Ntemp)
            print(f'Using Taylor series expansion with coefficients: {self.taylor_coeffs}')
        elif np.min(temp_exponents) == 0 and np.max(temp_exponents) > 0:
            # Positive temperature powers means that we are doing a Taylor series expansion in 1/T. Calculate and multiply the correct coefficients
            taylor_coeffs = np.array([factorial(x) for x in np.absolute(temp_exponents)])
            self.taylor_coeffs = torch.tensor(1 / taylor_coeffs, dtype = torch.float32).unsqueeze(0).to(device)
            print(f'Using Taylor series expansion with coefficients: {self.taylor_coeffs}')
        else:
            self.taylor_coeffs = None

    def forward(self, i, j, invT, r, qA, qB, N, rho):
        # A is the matrix Nchem x Nparam
        A_prod = torch.clamp(self.A[i, :], min = 0) * torch.clamp(self.A[j, :], min = 0) # Return an elementwise multiply of the clamp of each parameter where each row is a data entry and each column is the parameter
        A_dif = (self.A[i, :] - self.A[j, :])**2 # Return an elementwise squared difference where each row is a data entry and each column is the parameter
        A_gather = torch.stack((A_prod, A_dif), dim=2)

        A_i_prod_self = torch.clamp(self.A[i, :], min = 0) * torch.clamp(self.A[i, :], min = 0) # Return an elementwise multiply of the clamp of each parameter where each row is a data entry and each column is the parameter
        A_i_dif_self = (self.A[i, :] - self.A[i, :])**2 # Return an elementwise squared difference where each row is a data entry and each column is the parameter

        A_j_prod_self = torch.clamp(self.A[j, :], min = 0) * torch.clamp(self.A[j, :], min = 0) # Return an elementwise multiply of the clamp of each parameter where each row is a data entry and each column is the parameter
        A_j_dif_self = (self.A[j, :] - self.A[j, :])**2 # Return an elementwise squared difference where each row is a data entry and each column is the parameter

        A_i_gather_self = torch.stack((A_i_prod_self, A_i_dif_self), dim=2)
        A_j_gather_self = torch.stack((A_j_prod_self, A_j_dif_self), dim=2)

        # Choose the right ways of combining
        A_combined = self.operationLayer(A_gather, self.O)
        A_combined_temp = self.tempLayer(A_combined, self.T)

        A_i_combined = self.operationLayer(A_i_gather_self, self.O)
        p_i = self.tempLayer(A_i_combined, self.T)
        A_j_combined = self.operationLayer(A_j_gather_self, self.O)
        p_j = self.tempLayer(A_j_combined, self.T)

        # Use broadcasting to raise each element of T to the corresponding exponent
        Tmat = torch.pow(invT, self.exponents)  # Shape (Ndata, Ntemp)
        if self.taylor_coeffs is not None:
            Tmat = Tmat * self.taylor_coeffs

        u_ij = torch.sum( ( (A_combined_temp - p_j) * Tmat), dim=1, keepdim=True)
        u_ji = torch.sum( ( (A_combined_temp - p_i) * Tmat), dim=1, keepdim=True)

        Qi = self.relu(self.Q[i, :])

        result = Qi * (1 + u_ji - torch.exp(-u_ij))

        # Use the Wertheim layer if the layer exists
        if self.associationLayer is not None:
            asc_term = self.associationLayer(self.D, i, j, invT, r, qA, N, rho)
            asc_term = asc_term[:, None] # Add another dimension
            result = asc_term + result
        
        return result
    
class Wilson(nn.Module):
    def __init__(self, A_initial, T_initial, O_initial, D_initial, association_layer, temp_exponents, device):
        super(Wilson, self).__init__()
        self.A = nn.Parameter(torch.tensor(A_initial, dtype=torch.float32).to(device))  # Initialize A
        self.T = nn.Parameter(torch.tensor(T_initial, dtype=torch.float32).to(device))  # Initialize T here
        self.O = nn.Parameter(torch.tensor(O_initial, dtype=torch.float32).to(device))  # Initialize O here

        # For combining distancing layers and learning them correctly
        self.operationLayer = OperationSoftmaxWeight()
        
        # Choose the layer based on the specified type
        self.tempLayer = TempSoftmaxWeight()
        
        if association_layer == 'wertheim':
            self.associationLayer = Wertheim()
            self.D = nn.Parameter(torch.tensor(D_initial, dtype=torch.float32).to(device)) # Initialize D only if needed
        else:
            self.associationLayer = None
            print('No association term used')

        # Create an exponent tensor of shape (1, Ntemp) containing [0, 1, 2, ..., Ntemp-1]
        if T_initial.shape[1] == 1: # If ony one temp dim, then create only a single column which gets 1/T
            self.exponents = torch.ones(T_initial.shape[1]).unsqueeze(0).to(device)  # Shape (1, Ntemp)
        else:
            self.exponents = torch.tensor(temp_exponents).unsqueeze(0).to(device)  # Shape (1, Ntemp)

        if np.min(temp_exponents) < 0 and np.max(temp_exponents) == 1:
            # Negative temperature powers means that we are doing a Taylor series expansion in T. Calculate and multiply the correct coefficients
            taylor_coeffs = np.array([factorial(x) for x in np.absolute(temp_exponents - 1)])
            self.taylor_coeffs = torch.tensor(1 / taylor_coeffs, dtype=torch.float32).unsqueeze(0).to(device)  # Shape (1, Ntemp)
            print(f'Using Taylor series expansion with coefficients: {self.taylor_coeffs}')
        elif np.min(temp_exponents) == 0 and np.max(temp_exponents) > 0:
            # Positive temperature powers means that we are doing a Taylor series expansion in 1/T. Calculate and multiply the correct coefficients
            taylor_coeffs = np.array([factorial(x) for x in np.absolute(temp_exponents)])
            self.taylor_coeffs = torch.tensor(1 / taylor_coeffs, dtype = torch.float32).unsqueeze(0).to(device)
            print(f'Using Taylor series expansion with coefficients: {self.taylor_coeffs}')
        else:
            self.taylor_coeffs = None

    def forward(self, i, j, invT, r, qA, qB, N, rho):
        # A is the matrix Nchem x Nparam
        A_prod = torch.clamp(self.A[i, :], min = 0) * torch.clamp(self.A[j, :], min = 0) # Return an elementwise multiply of the clamp of each parameter where each row is a data entry and each column is the parameter
        A_dif = (self.A[i, :] - self.A[j, :])**2 # Return an elementwise squared difference where each row is a data entry and each column is the parameter
        A_gather = torch.stack((A_prod, A_dif), dim=2)

        A_i_prod_self = torch.clamp(self.A[i, :], min = 0) * torch.clamp(self.A[i, :], min = 0) # Return an elementwise multiply of the clamp of each parameter where each row is a data entry and each column is the parameter
        A_i_dif_self = (self.A[i, :] - self.A[i, :])**2 # Return an elementwise squared difference where each row is a data entry and each column is the parameter

        A_j_prod_self = torch.clamp(self.A[j, :], min = 0) * torch.clamp(self.A[j, :], min = 0) # Return an elementwise multiply of the clamp of each parameter where each row is a data entry and each column is the parameter
        A_j_dif_self = (self.A[j, :] - self.A[j, :])**2 # Return an elementwise squared difference where each row is a data entry and each column is the parameter

        A_i_gather_self = torch.stack((A_i_prod_self, A_i_dif_self), dim=2)
        A_j_gather_self = torch.stack((A_j_prod_self, A_j_dif_self), dim=2)

        # Choose the right ways of combining
        A_combined = self.operationLayer(A_gather, self.O)

        A_combined_temp = self.tempLayer(A_combined, self.T)

        A_i_combined = self.operationLayer(A_i_gather_self, self.O)
        p_i = self.tempLayer(A_i_combined, self.T)
        A_j_combined = self.operationLayer(A_j_gather_self, self.O)
        p_j = self.tempLayer(A_j_combined, self.T)

        # Use broadcasting to raise each element of T to the corresponding exponent
        Tmat = torch.pow(invT, self.exponents)  # Shape (Ndata, Ntemp)
        if self.taylor_coeffs is not None:
            Tmat = Tmat * self.taylor_coeffs

        u_ij = torch.sum( ( (A_combined_temp - p_j) * Tmat), dim=1, keepdim=True)
        u_ji = torch.sum( ( (A_combined_temp - p_i) * Tmat), dim=1, keepdim=True)
        
        result = r * (1 + u_ji - torch.exp(-u_ij))

        # Use the Wertheim layer if the layer exists
        if self.associationLayer is not None:
            asc_term = self.associationLayer(self.D, i, j, invT, r, qA, N, rho)
            asc_term = asc_term[:, None] # Add another dimension
            result = asc_term + result
        
        return result
    
class NRTL(nn.Module):
    def __init__(self, Alpha_initial, A_initial, T_initial, O_initial, D_initial, association_layer, temp_exponents, device):
        super(NRTL, self).__init__()
        self.Alpha = nn.Parameter(torch.tensor(Alpha_initial, dtype=torch.float32).to(device))  # Initialize A
        self.A = nn.Parameter(torch.tensor(A_initial, dtype=torch.float32).to(device))  # Initialize A
        self.T = nn.Parameter(torch.tensor(T_initial, dtype=torch.float32).to(device))  # Initialize T here
        self.O = nn.Parameter(torch.tensor(O_initial, dtype=torch.float32).to(device))  # Initialize O here

        # For combining distancing layers and learning them correctly
        self.operationLayer = OperationSoftmaxWeight()
        self.tempLayer = TempSoftmaxWeight()
        
        if association_layer == 'wertheim':
            self.associationLayer = Wertheim()
            self.D = nn.Parameter(torch.tensor(D_initial, dtype=torch.float32).to(device)) # Initialize D only if needed
        else:
            self.associationLayer = None
            print('No association term used')

        # Create an exponent tensor of shape (1, Ntemp) containing [0, 1, 2, ..., Ntemp-1]
        if T_initial.shape[1] == 1: # If ony one temp dim, then create only a single column which gets 1/T
            self.exponents = torch.ones(T_initial.shape[1]).unsqueeze(0).to(device)  # Shape (1, Ntemp)
        else:
            self.exponents = torch.tensor(temp_exponents).unsqueeze(0).to(device)  # Shape (1, Ntemp)

        if np.min(temp_exponents) < 0 and np.max(temp_exponents) == 1:
            # Negative temperature powers means that we are doing a Taylor series expansion in T. Calculate and multiply the correct coefficients
            taylor_coeffs = np.array([factorial(x) for x in np.absolute(temp_exponents - 1)])
            self.taylor_coeffs = torch.tensor(1 / taylor_coeffs, dtype=torch.float32).unsqueeze(0).to(device)  # Shape (1, Ntemp)
            print(f'Using Taylor series expansion with coefficients: {self.taylor_coeffs}')
        elif np.min(temp_exponents) == 0 and np.max(temp_exponents) > 0:
            # Positive temperature powers means that we are doing a Taylor series expansion in 1/T. Calculate and multiply the correct coefficients
            taylor_coeffs = np.array([factorial(x) for x in np.absolute(temp_exponents)])
            self.taylor_coeffs = torch.tensor(1 / taylor_coeffs, dtype = torch.float32).unsqueeze(0).to(device)
            print(f'Using Taylor series expansion with coefficients: {self.taylor_coeffs}')
        else:
            self.taylor_coeffs = None
            

    def forward(self, i, j, invT, r, qA, qB, N, rho):
        # A is the matrix Nchem x Nparam
        A_prod = torch.clamp(self.A[i, :], min = 0) * torch.clamp(self.A[j, :], min = 0) # Return an elementwise multiply of the clamp of each parameter where each row is a data entry and each column is the parameter
        A_dif = (self.A[i, :] - self.A[j, :])**2 # Return an elementwise squared difference where each row is a data entry and each column is the parameter
        A_gather = torch.stack((A_prod, A_dif), dim=2)

        A_i_prod_self = torch.clamp(self.A[i, :], min = 0) * torch.clamp(self.A[i, :], min = 0) # Return an elementwise multiply of the clamp of each parameter where each row is a data entry and each column is the parameter
        A_i_dif_self = (self.A[i, :] - self.A[i, :])**2 # Return an elementwise squared difference where each row is a data entry and each column is the parameter

        A_j_prod_self = torch.clamp(self.A[j, :], min = 0) * torch.clamp(self.A[j, :], min = 0) # Return an elementwise multiply of the clamp of each parameter where each row is a data entry and each column is the parameter
        A_j_dif_self = (self.A[j, :] - self.A[j, :])**2 # Return an elementwise squared difference where each row is a data entry and each column is the parameter

        A_i_gather_self = torch.stack((A_i_prod_self, A_i_dif_self), dim=2)
        A_j_gather_self = torch.stack((A_j_prod_self, A_j_dif_self), dim=2)

        # Choose the right ways of combining
        A_combined = self.operationLayer(A_gather, self.O)

        A_combined_temp = self.tempLayer(A_combined, self.T)

        A_i_combined = self.operationLayer(A_i_gather_self, self.O)
        p_i = self.tempLayer(A_i_combined, self.T)
        A_j_combined = self.operationLayer(A_j_gather_self, self.O)
        p_j = self.tempLayer(A_j_combined, self.T)

        # Alpha parameters are kept between 0 and 1. In the Renon and Prausnitz formulation, alpha is the non-randomness paramter and should be low if systems are similar in chemical identity should be low. Differencing offers the best way of this
        alpha_i = torch.clamp(self.Alpha[i, :], min = 0, max = 1)
        alpha_j = torch.clamp(self.Alpha[j, :], min = 0, max = 1)
        alpha_ij = torch.sum((alpha_i - alpha_j)**2, dim = 1, keepdim=True) # Return the sum of squared elementwise squared difference where each row is a data entry and each column is the parameter

        # Compute the NRTL function
        # Use broadcasting to raise each element of T to the corresponding exponent
        Tmat = torch.pow(invT, self.exponents)  # Shape (Ndata, Ntemp)
        if self.taylor_coeffs is not None:
            Tmat = Tmat * self.taylor_coeffs

        u_ij = torch.sum( ( (A_combined_temp - p_j) * Tmat), dim=1, keepdim=True)
        u_ji = torch.sum( ( (A_combined_temp - p_i) * Tmat), dim=1, keepdim=True)
        
        result = (u_ji + (u_ij) * torch.exp(-alpha_ij * u_ij)) # All should have dim of Ndata x 1

        # Use the Wertheim layer if the layer exists
        if self.associationLayer is not None:
            asc_term = self.associationLayer(self.D, i, j, invT, r, qA, N, rho)
            asc_term = asc_term[:, None] # Add another dimension
            result = asc_term + result
        
        return result

# Early stopping where if after 100 steps (or patience), there an increase in the loss value
class EarlyStopper:
    def __init__(self, patience=100, threshold=0):
        self.patience = patience  # Number of iterations to check loss change
        self.threshold = threshold  # Threshold for 0% change
        self.counter = 0  # To count the number of iterations without significant improvement
        self.best_loss = None  # To store the best loss encountered
        self.early_stop = False

    def __call__(self, current_loss):
        if self.best_loss is None:
            self.best_loss = current_loss  # Set the first loss as the best loss
        
        # Calculate percentage change in loss, if increasing, then it automatically fails
        loss_change = (self.best_loss - current_loss)

        if loss_change < self.threshold:  # If change is increasing
            self.counter += 1
        else:
            self.counter = 0  # Reset the counter if loss improves

        # Update best loss
        if current_loss < self.best_loss:
            self.best_loss = current_loss

        # Stop training if no improvement after patience
        if self.counter >= self.patience:
            self.early_stop = True

# Define custom loss functions
class MAPELoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, outputs, targets):
        return (torch.absolute((targets - outputs) / targets)).mean()
