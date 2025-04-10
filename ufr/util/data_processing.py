"""

Oliver Xie, Olsen Group at Massachusetts Institute of Technology, 2025
@author oxie25

"""

import numpy as np
import pandas as pd
import warnings
# Disable SettingWithCopyWarning
pd.options.mode.chained_assignment = None  # default='warn'
# Disable prototype warnings
warnings.filterwarnings(action='ignore', category=UserWarning)
# Disable future deprecation warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def remove_temperature_outliers(df, min_temps=4, min_points=5, std_dev = 3, solute_smiles = 'Solute SMILES', solvent_smiles = 'Solvent SMILES'):
    # Create copy of dataframe to avoid modifying original
    df_clean = df.copy()
    
    # Get unique solute-solvent pairs
    pairs = df_clean.groupby([solute_smiles, solvent_smiles])
    
    # Store indices to drop
    indices_to_drop = []
    
    # Process each pair
    for (solute, solvent), group in pairs:
        # Check if group meets minimum requirements
        n_temps = group['Temp (K)'].nunique()
        n_points = len(group)
        
        if n_temps >= min_temps and n_points >= min_points:
            # Calculate 1/T
            inv_temp = 1 / group['Temp (K)']
            ln_activity = group['ln gamma']
            
            # If not aqueous, use linear fit and 2 standard deviations as cutoff criteria. If aqueous, use quadratic fit and 2 standard deviations as cutoff criteria
            if solvent != 'O' and solute != 'O':
                # Fit linear regression
                coeffs = np.polyfit(inv_temp, ln_activity, 1)
                y_pred = coeffs[0] * inv_temp + coeffs[1]
                
                # Calculate residuals
                residuals = ln_activity - y_pred
                
                # Calculate standard deviation of residuals
                std_residuals = np.sqrt(np.sum(residuals ** 2) / (len(residuals) - 2))
                
                # Find outliers (points more than x std deviations away)
                outliers = abs(residuals) > std_dev * std_residuals
                
                # Add outlier indices to list
                indices_to_drop.extend(group[outliers].index)
            elif solvent == 'O' or solute == 'O':
                # Fit quadratic regression
                coeffs = np.polyfit(inv_temp, ln_activity, 2)
                y_pred = coeffs[0] * inv_temp ** 2 + coeffs[1] * inv_temp + coeffs[2]
                
                # Calculate residuals
                residuals = ln_activity - y_pred
                
                # Calculate standard deviation of residuals
                std_residuals = np.sqrt(np.sum(residuals ** 2) / (len(residuals) - 3))
                
                # Find outliers (points more than x std deviations away)
                outliers = abs(residuals) > std_dev * std_residuals
                
                # Add outlier indices to list
                indices_to_drop.extend(group[outliers].index)
    
    # Drop all identified outliers
    df_clean = df_clean.drop(indices_to_drop)
    
    print(f"Removed {len(indices_to_drop)} outliers from {len(df)} points")

    # Create a dataframe of the dropped rows
    dropped_indices = set(indices_to_drop)
    df_dropped = df[df.index.isin(dropped_indices)]

    print(f"Dropped dataframe has {len(df_dropped)} rows")
    
    return df_clean, df_dropped

# Do manual cleaning first. Load the file of data cleaning rules
def apply_cleaning(df, df_cleaning_rules, solute_smiles = 'Solute SMILES', solvent_smiles = 'Solvent SMILES'):
    # Create copy of dataframe to avoid modifying original
    df_clean = df.copy()

    # For every row in the cleaning rules, apply the rule to the solute-solvent pair
    for index, row in df_cleaning_rules.iterrows():
        # Get the solute and solvent
        solute = row['Solute SMILES']
        solvent = row['Solvent SMILES']

        # Iteratively add conditions
        conditions = []
        # Get the correct solute solvent pair
        conditions.append(df_clean[solute_smiles] == solute)
        conditions.append(df_clean[solvent_smiles] == solvent)

        if pd.notna(row['Removal Source']):
            conditions.append(df_clean['Source'] == row['Removal Source'])
        
        if pd.notna(row['Removal T min']):
            conditions.append(df_clean['Temp (K)'] >= row['Removal T min'])
        
        if pd.notna(row['Removal T max']):
            conditions.append(df_clean['Temp (K)'] <= row['Removal T max'])

        if row['Removal Activity Rule'] == 'Greater':
            conditions.append(df_clean['ln gamma'] > row['Removal Activity Value'])
        elif row['Removal Activity Rule'] == 'Less':
            conditions.append(df_clean['ln gamma'] < row['Removal Activity Value'])
        elif row['Removal Activity Rule'] == 'Rescale':
            # We need to perform operation on the ln gamma. Resscale the BARE gamma value
            scale_value = row['Removal Activity Value']
            # Combine the conditions
            combined_conditions = conditions[0]
            for condition in conditions[1:]:
                combined_conditions = combined_conditions & condition
            # Apply the filter
            df_clean.loc[combined_conditions, 'ln gamma'] = np.log(np.exp(df_clean.loc[combined_conditions, 'ln gamma']) * scale_value)
            continue
        elif row['Removal Activity Rule'] == 'Drop':
            # Drop the entire set of data
            combined_conditions = conditions[0]
            for condition in conditions[1:]:
                combined_conditions = combined_conditions & condition
            df_clean = df_clean[~combined_conditions]
            continue
        elif pd.isna(row['Removal Activity Rule']):
            continue
        else:
            print(f'Error, unknown removal rule: {row["Removal Activity Rule"]} for solute {solute} and solvent {solvent}')
            continue

        # Combine the conditions
        combined_conditions = conditions[0]
        for condition in conditions[1:]:
            combined_conditions = combined_conditions & condition
        
        # Apply the filter
        df_clean = df_clean[~combined_conditions]
    
    return df_clean

def drop_duplicates(df, solute_smiles = 'Solute SMILES', solvent_smiles = 'Solvent SMILES'):
    # Check if there are any duplicate entries. Defined same solute, solvent and temperature with same activity. This might be a bit lax (sometimes activities differ a bit) but should be good enough

    # Drop duplicates of df where solute, solvent, temperature and activity are the same
    df_return = df.drop_duplicates(subset=[solute_smiles, solvent_smiles, 'Temp (K)', 'ln gamma'], keep='first')

    return df_return

def invtemp_gradient_calc(df, y_label, min_temps = 6, min_points = 10, min_delta_T = 0, std_residuals_tol = 0.1, rel_std_residuals_tol = 0.1, solute_smiles = 'Solute SMILES', solvent_smiles = 'Solvent SMILES'):
    # Initialize a column for the slopes
    df['dlny_dinvT'] = np.nan
    df['residual_std_dev'] = np.nan
    df['relative_residual_std_dev'] = np.nan
    df['ae_fit'] = np.nan

    # Get unique solute-solvent pairs
    pairs = df.groupby([solute_smiles, solvent_smiles])
    counter = 0
    calc_idac = 0

    num_unique_groups = len(pairs)

    # Process each pair
    for (solute, solvent), group in pairs:
        # Check if group meets minimum requirements
        n_temps = group['Temp (K)'].nunique()
        delta_T = group['Temp (K)'].max() - group['Temp (K)'].min()
        n_points = len(group)
        
        if n_temps >= min_temps and n_points >= min_points and delta_T >= min_delta_T:
            # Calculate 1/T
            inv_temp = 1 / group['Temp (K)'].to_numpy()
            ln_activity = group[y_label].to_numpy()

            # Never used when min_temps > 1 and min_points > 1, but put here for when finding best fit
            if n_temps == 1 and n_points == 1:
                y_derivative = 0
                df.loc[group.index, ['dlny_dinvT', 'residual_std_dev', 'relative_residual_std_dev', 'mae_fit']] = [0, 0, 0, 0]

            elif n_temps == 1 and n_points > 1:
                # Get the mean of these points
                mean_ln_gamma = np.mean(ln_activity)
                # Find the absolute error for each point
                ae = np.abs(ln_activity - mean_ln_gamma)
                df.loc[group.index, ['dlny_dinvT', 'residual_std_dev', 'relative_residual_std_dev', 'ae_fit']] = np.column_stack((np.repeat(0, len(ae)), np.repeat(0, len(ae)), np.repeat(0, len(ae)), ae))
            
            else:
                # If not aqueous, use linear fit and 2 standard deviations as cutoff criteria. If aqueous, use quadratic fit and 2 standard deviations as cutoff criteria
                # Sobolev is very prone to error, so keep the min_temp and min_points requirement high
                if solvent != 'O' and solute != 'O':
                    # Fit linear regression
                    coeffs = np.polyfit(inv_temp, ln_activity, 1)
                    y_pred = coeffs[0] * inv_temp + coeffs[1]
                    y_derivative = coeffs[0] * np.ones_like(inv_temp)
                    
                    # Calculate residuals
                    residuals = ln_activity - y_pred
                    
                    # Calculate standard deviation of residuals
                    std_residuals = np.sqrt(np.sum(residuals ** 2) / (len(residuals) - 2))
                    rel_std_residuals = std_residuals / np.absolute(np.mean(ln_activity))
                    
                elif solvent == 'O' or solute == 'O':
                    # Fit quadratic regression
                    coeffs = np.polyfit(inv_temp, ln_activity, 2)
                    y_pred = coeffs[0] * inv_temp ** 2 + coeffs[1] * inv_temp + coeffs[2]
                    y_derivative = 2 * coeffs[0] * inv_temp + coeffs[1]
                    
                    # Calculate residuals
                    residuals = ln_activity - y_pred
                    
                    # Calculate standard deviation of residuals
                    std_residuals = np.sqrt(np.sum(residuals ** 2) / (len(residuals) - 3))
                    rel_std_residuals = std_residuals / np.absolute(np.mean(ln_activity))

                # Calculate the mean absolute error of the pairing
                ae = np.abs(residuals)

                # Assign the calculated slope to the entire group
                # Assign y_derivative at each temperature point, not just the average
                #t_idx = 0
                #for idx in group.index:
                #    df.loc[idx, ['dlny_dinvT', 'residual_std_dev', 'mae_fit']] = [y_derivative[t_idx], std_residuals, mae]
                #    t_idx += 1
                if std_residuals < std_residuals_tol and rel_std_residuals < rel_std_residuals_tol:
                    df.loc[group.index, ['dlny_dinvT', 'residual_std_dev', 'relative_residual_std_dev', 'ae_fit']] = np.column_stack((y_derivative, np.repeat(std_residuals, len(y_derivative)), np.repeat(rel_std_residuals, len(y_derivative)), ae))
                    #df.loc[(df[i_label] == solute) & (df[j_label] == solvent), ['dlny_dinvT', 'residual_std_dev', 'mae_fit']] = [y_derivative, std_residuals, mae]
                    calc_idac += 1
                else:
                    continue

        # Print message every 500th iteration
        if (counter + 1) % 500 == 0:
            print(f'Completed {counter+1} pairs out of {num_unique_groups}')

        counter += 1

    print(f'Calculated {calc_idac} slopes out of {num_unique_groups} pairs')

    return df

# Use the ELBRO-FV correction on the free volume and combinatorial entropy term
def daubert_danner(T, A, B, C, D, eqn_num):
    # Calculate the molar volume im m3/kmol at the prescribed temperature T in Kelvin
    # Determine which equation to use
    if eqn_num == 105:
        factor = 1 + (1 - (T / C))**D
        rho = A / B**factor # in units of kmol/m3
    elif eqn_num == 100:
        rho = A + B*T + C*T**2 + D*T**3 # E * T**4
    elif eqn_num == 116:
        tau = (1 - (T / 647.096)) # water
        rho = A + B*tau**0.35 + C * tau**(2/3) + D*tau
    else:
        print(f"Unknown equation number, reported value is: {eqn_num}")
    v = (1 / rho)

    return v


def molecular_property_addition(df_inf, df_prop, mode, volume_smiles = 'Canonical SMILES', solvent_smiles = 'Solvent SMILES', solute_smiles = 'Solute SMILES'):
    print(f'There are {df_prop.shape[0]} different molecules in the dataset')

    # Count the number of times each pairing of solvent, solute occurs. This is counting how many temperature data points there are and serves as the weight
    df_inf['Solute_Solvent_Count'] = df_inf.groupby([solute_smiles, solvent_smiles]).transform('size')

    # Calculate residual ln gamma
    # van der Waals molar volume in m3/kmol
    df_inf['Solvent_VDW_Volumes'] = df_inf[solvent_smiles].map(df_prop.set_index(volume_smiles)['van der waals volume (m3/kmol)'])
    df_inf['Solute_VDW_Volumes'] = df_inf[solute_smiles].map(df_prop.set_index(volume_smiles)['van der waals volume (m3/kmol)'])

    # Surface Area in m2/kmol
    df_inf['Solvent_VDW_Area'] = df_inf[solvent_smiles].map(df_prop.set_index(volume_smiles)['van der waals area (m2/kmol)'])
    df_inf['Solute_VDW_Area'] = df_inf[solute_smiles].map(df_prop.set_index(volume_smiles)['van der waals area (m2/kmol)'])

    # Molecular weights kg/kmol
    df_inf['Solvent_MW'] = df_inf[solvent_smiles].map(df_prop.set_index(volume_smiles)['molecular weight (kg/kmol)'])
    df_inf['Solute_MW'] = df_inf[solute_smiles].map(df_prop.set_index(volume_smiles)['molecular weight (kg/kmol)'])

    # H-bond donor and acceptor sites
    df_inf['Solvent_H_Donor_Sites'] = df_inf[solvent_smiles].map(df_prop.set_index(volume_smiles)['H donor sites'])
    df_inf['Solvent_H_Acceptor_Sites'] = df_inf[solvent_smiles].map(df_prop.set_index(volume_smiles)['H acceptor sites'])
    df_inf['Solute_H_Donor_Sites'] = df_inf[solute_smiles].map(df_prop.set_index(volume_smiles)['H donor sites'])
    df_inf['Solute_H_Acceptor_Sites'] = df_inf[solute_smiles].map(df_prop.set_index(volume_smiles)['H acceptor sites'])

    # Activity coefficients
    # Drop columns that don't have VDW volumes ONLY. We did RDKit calc on VDW volumes so this is ok for FH and mod_FH methods only
    df_inf_filtered = df_inf.dropna(subset=['Solvent_VDW_Volumes', 'Solute_VDW_Volumes', 'Solvent_VDW_Area', 'Solute_VDW_Area'], how = 'any')

    ########################################################################
    # Calculate combinatorial properties that only rely on VDW volumes
    # Use the Flory-Huggins correction on the combinatorial entropy term
    # Note, in the Abrams and Prausnitz formulation of UNIQUAC, these are taken to be the VDW volumes. We use these here as well
    # ln(gamma_res) = ln(gamma) - (ln(v_solute/v_solvent) + 1 - v_solute/v_solvent)
    comb = ( np.log(df_inf_filtered['Solute_VDW_Volumes']/df_inf_filtered['Solvent_VDW_Volumes']) + 1 - df_inf_filtered['Solute_VDW_Volumes']/df_inf_filtered['Solvent_VDW_Volumes'] )
    ln_gamma_res = df_inf_filtered['ln gamma'] - comb
    # Correct for solvent size?
    df_inf_filtered['ln_gamma_res_FH'] = ln_gamma_res
    df_inf_filtered['comb_FH'] = comb

    # Modified Flory-Huggins to the 2/3
    comb = ( np.log((df_inf_filtered['Solute_VDW_Volumes']/df_inf_filtered['Solvent_VDW_Volumes'])**(2/3)) + 1 - (df_inf_filtered['Solute_VDW_Volumes']/df_inf_filtered['Solvent_VDW_Volumes'])**(2/3) )
    ln_gamma_res = df_inf_filtered['ln gamma'] - comb
    # Correct for solvent size?
    df_inf_filtered['ln_gamma_res_mod_FH'] = ln_gamma_res
    df_inf_filtered['comb_mod_FH'] = comb

    # Staverman-Guggenheim correction on the combinatorial entropy
    #ln(gamma_res) = ln(gamma) - (ln(v_solute/v_solvent) - + 1 - v_solute/v_solvent) + z/2 * qA * (ln(rA/rB / qA/qB) + 1 - (rA/rB)/(qA/qB))
    v0 = 0.01517 # the -CH2- reference volume used by Abrams and Prausnitz (m3/kmol)
    s0 = 2.5 * 10**8 # the -CH2- reference area used by Abrams and Prausnitz (m2/kmol)
    z = 10 # coordination number used by Abrams and Prausnitz
    qA = df_inf_filtered['Solute_VDW_Area'] / s0
    qB = df_inf_filtered['Solvent_VDW_Area'] / s0
    rA = df_inf_filtered['Solute_VDW_Volumes'] / v0
    rB = df_inf_filtered['Solvent_VDW_Volumes'] / v0
    comb = ( np.log(df_inf_filtered['Solute_VDW_Volumes']/df_inf_filtered['Solvent_VDW_Volumes']) + 1 - df_inf_filtered['Solute_VDW_Volumes']/df_inf_filtered['Solvent_VDW_Volumes'] + z/2 * qA * ( np.log((rA/rB) / (qA/qB)) + 1 - (rA/rB)/(qA/qB) ) )
    ln_gamma_res_SG = df_inf_filtered['ln gamma'] - comb
    df_inf_filtered['ln_gamma_res_SG'] = ln_gamma_res_SG
    df_inf_filtered['comb_SG'] = comb

    if mode == 'FV':
        # Liquid molar volume in m3/kmol
        df_inf_filtered['Solvent_Liquid_Volumes'] = df_inf_filtered[solvent_smiles].map(df_prop.set_index(volume_smiles)['liquid molar volume (m3/kmol)'])
        df_inf_filtered['Solute_Liquid_Volumes'] = df_inf_filtered[solute_smiles].map(df_prop.set_index(volume_smiles)['liquid molar volume (m3/kmol)'])

        # Daubert and Danner coefficients. Eqn is v = A/B^(1+(1 - T/C)^D) in kg/m^3 (liquid density) with T in K
        df_inf_filtered['Solvent_A'] = df_inf_filtered[solvent_smiles].map(df_prop.set_index(volume_smiles)['A'])
        df_inf_filtered['Solute_A'] = df_inf_filtered[solute_smiles].map(df_prop.set_index(volume_smiles)['A'])

        df_inf_filtered['Solvent_B'] = df_inf_filtered[solvent_smiles].map(df_prop.set_index(volume_smiles)['B'])
        df_inf_filtered['Solute_B'] = df_inf_filtered[solute_smiles].map(df_prop.set_index(volume_smiles)['B'])

        df_inf_filtered['Solvent_C'] = df_inf_filtered[solvent_smiles].map(df_prop.set_index(volume_smiles)['C'])
        df_inf_filtered['Solute_C'] = df_inf_filtered[solute_smiles].map(df_prop.set_index(volume_smiles)['C'])

        df_inf_filtered['Solvent_D'] = df_inf_filtered[solvent_smiles].map(df_prop.set_index(volume_smiles)['D'])
        df_inf_filtered['Solute_D'] = df_inf_filtered[solute_smiles].map(df_prop.set_index(volume_smiles)['D'])

        df_inf_filtered['Solvent_Eqn'] = df_inf_filtered[solvent_smiles].map(df_prop.set_index(volume_smiles)['Equation'])
        df_inf_filtered['Solute_Eqn'] = df_inf_filtered[solute_smiles].map(df_prop.set_index(volume_smiles)['Equation'])

        # Drop all without temperature correlation
        df_inf_filtered.dropna(subset=['Solvent_A', 'Solute_A'], how = 'any', inplace = True)

        T = 298.15
        df_inf_filtered['Solvent_Volume_Calc'] = df_inf_filtered.apply(lambda row: daubert_danner(row['Temp (K)'], row['Solvent_A'], row['Solvent_B'], row['Solvent_C'], row['Solvent_D'], row['Solvent_Eqn']), axis = 1)
        df_inf_filtered['Solute_Volume_Calc'] = df_inf_filtered.apply(lambda row: daubert_danner(row['Temp (K)'], row['Solute_A'], row['Solute_B'], row['Solute_C'], row['Solute_D'], row['Solute_Eqn']), axis = 1)

        # Free Volume Calc
        df_inf_filtered['Solvent_FV'] = df_inf_filtered['Solvent_Volume_Calc'] - df_inf_filtered['Solvent_VDW_Volumes']
        df_inf_filtered['Solute_FV'] = df_inf_filtered['Solute_Volume_Calc'] - df_inf_filtered['Solute_VDW_Volumes']

        ln_gamma_res_FV = df_inf_filtered['ln gamma'] - ( np.log(df_inf_filtered['Solute_FV']/df_inf_filtered['Solvent_FV']) + 1 - df_inf_filtered['Solute_FV']/df_inf_filtered['Solvent_FV'])
        df_inf_filtered['ln_gamma_res_FV'] = ln_gamma_res_FV

        # Some of the data temp is beyond regression range, check here for imaginaries
        imaginary = df_inf_filtered['ln_gamma_res_FV'].apply(lambda x: np.iscomplex(x) and x.imag != 0)
        df_inf_filtered = df_inf_filtered[~imaginary]

        # Modified Elbro FV
        ln_gamma_res_FV = df_inf_filtered['ln gamma'] - ( np.log((df_inf_filtered['Solute_FV']/df_inf_filtered['Solvent_FV'])**(2/3)) + 1 - (df_inf_filtered['Solute_FV']/df_inf_filtered['Solvent_FV'])**(2/3))
        df_inf_filtered['ln_gamma_res_mod_FV'] = ln_gamma_res_FV

        # Some of the data temp is beyond regression range, check here for imaginaries
        imaginary = df_inf_filtered['ln_gamma_res_mod_FV'].apply(lambda x: np.iscomplex(x) and x.imag != 0)
        df_inf_fv_filtered = df_inf_filtered[~imaginary]

        # GK-FV correction on the combinatorial entropy
        #ln(gamma_res) = ln(gamma) - (ln(fv_solute/fv_solvent) - + 1 - v_solute/v_solvent) + z/2 * qA * (ln(rA/rB / qA/qB) + 1 - (rA/rB)/(qA/qB))
        qA = df_inf_filtered['Solute_VDW_Area'] / s0
        qB = df_inf_filtered['Solvent_VDW_Area'] / s0
        rA = df_inf_filtered['Solute_VDW_Volumes'] / v0
        rB = df_inf_filtered['Solvent_VDW_Volumes'] / v0
        ln_gamma_res_GKFV = df_inf_filtered['ln gamma'] - ( np.log(df_inf_filtered['Solute_FV']/df_inf_filtered['Solvent_FV']) + 1 - df_inf_filtered['Solute_FV']/df_inf_filtered['Solvent_FV'] + z/2 * qA * ( np.log((rA/rB) / (qA/qB)) + 1 - (rA/rB)/(qA/qB) ) )
        df_inf_filtered['ln_gamma_res_GK-FV'] = ln_gamma_res_GKFV

        # Some of the data temp is beyond regression range, check here for imaginaries
        imaginary = df_inf_filtered['ln_gamma_res_GK-FV'].apply(lambda x: np.iscomplex(x) and x.imag != 0)
        df_inf_filtered = df_inf_filtered[~imaginary]

        # Convert all complex back into real
        # Function to convert complex numbers to their real parts
        def convert_complex_to_real(x):
            if isinstance(x, complex):
                return np.real(x)
            return x

        # Apply the function to each element of the DataFrame using apply twice
        df_inf_filtered = df_inf_filtered.apply(lambda col: col.apply(convert_complex_to_real))

    return df_inf_filtered, v0, s0