"""

Oliver Xie, Olsen Group at Massachusetts Institute of Technology, 2025
@author oxie25

"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
# Disable prototype warnings
warnings.filterwarnings(action='ignore', category=UserWarning)
# Disable future deprecation warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from ..model.idac_model import UNIQUAC, Wilson, NRTL, mod_UNIQUAC, EarlyStopper, MAPELoss

def calculate_node_degree_chemical(df, nodes, col1, col2):
    # Count unique solute-solvent pairs
    unique_pairs = df[[col1, col2]].drop_duplicates()
    counts_col1 = unique_pairs[col1].value_counts().reindex(nodes, fill_value=0)
    counts_col2 = unique_pairs[col2].value_counts().reindex(nodes, fill_value=0)

    # Sum counts from both columns
    degree = counts_col1 + counts_col2
    return degree

def calculate_node_degree_temperature(df, nodes, col1, col2, col3):
    # Count unique solute-solvent pairs across different temperatures
    unique_pairs = df[[col1, col2, col3]].drop_duplicates()
    counts_col1 = unique_pairs[col1].value_counts().reindex(nodes, fill_value=0)
    counts_col2 = unique_pairs[col2].value_counts().reindex(nodes, fill_value=0)

    # Sum counts from all columns
    degree = counts_col1 + counts_col2
    return degree

def model_selector(model_label, A_initial, Alpha_initial, Q_initial, T_initial, O_initial, D_initial, association_layer, temp_exponents, device):
    # Select model here
    if model_label == 'UNIQUAC':
        model = UNIQUAC(A_initial, T_initial, O_initial, D_initial, association_layer, temp_exponents, device)
    elif model_label == 'Wilson':
        model = Wilson(A_initial, T_initial, O_initial, D_initial, association_layer, temp_exponents, device)
    elif model_label == 'NRTL':
        model = NRTL(Alpha_initial, A_initial, T_initial, O_initial, D_initial, association_layer, temp_exponents, device)
    elif model_label == 'mod_UNIQUAC':
        model = mod_UNIQUAC(A_initial, Q_initial, T_initial, O_initial, D_initial, association_layer, temp_exponents, device)
    else:
        print(f'Model label of {model_label} is invalid')
    
    return model

def up_down_lr_schedule(epoch, low_factor, total_steps, up_steps, hold_steps):
    if epoch < up_steps:
        # Increase from 0.1 to 1 over `up_steps`
        return low_factor + (epoch / up_steps) * (1.0 - low_factor)
    elif epoch < up_steps + hold_steps:
        # Hold at 1 for `hold_steps`
        return 1.0
    else:
        # Decrease back to 0.1 over the remaining steps
        return 1.0 - ((epoch - up_steps - hold_steps) / (total_steps - up_steps - hold_steps)) * (1.0 - low_factor)

def launch_model(dim, trials, df_inf, df_prop, model_layer_options, model_opt_options, model_run_options, device):
    # Set index of df_prop to be Canonical SMILES
    df_prop.set_index('Canonical SMILES', inplace=True)

    # Unpack model options
    ln_y_data = model_layer_options['ln_y_data']
    combinatorial_layer = model_layer_options['combinatorial_layer']
    residual_layer = model_layer_options['residual_layer']
    association_layer = model_layer_options['association_layer']
    temp_exponents = model_layer_options['temp_exponents']
    v0 = model_layer_options['reference_volume']
    s0 = model_layer_options['reference_area']

    # Unpack optimization options
    sobolev = model_opt_options['sobolev']
    lr = model_opt_options['lr']
    total_epochs = model_opt_options['total_epochs']
    up_epochs = model_opt_options['up_epochs']
    hold_epochs = model_opt_options['hold_epochs']

    # Unpack run options
    truncation = model_run_options['truncation']
    pre_train_epoch = model_run_options['pre_train_epoch']
    solute_smiles, solvent_smiles = model_run_options['smile_labels']
    save_name = model_run_options['save_name']

    # Set up the model dimensions
    alpha_dim = 0
    delta_dim = 0
    q_dim = 0
    temp_dim = np.size(temp_exponents)
    
    if residual_layer == 'NRTL':
        alpha_dim = 1
    elif residual_layer == 'mod_UNIQUAC':
        q_dim = 1
    
    if association_layer == 'wertheim':
        delta_dim = 2

    u_dim = dim - alpha_dim - delta_dim - q_dim
    dim_active = u_dim
    param_dim_list = [u_dim]
    col_names = ['ua']
    
    # Set up the column names for df_results. Order must be u, alpha, q, delta
    if residual_layer == 'NRTL':
        print(f'Residual {residual_layer}, setting alpha as variable')
        param_dim_list.extend([dim_active + alpha_dim])
        col_names.extend(['alpha'])
        dim_active = param_dim_list[-1] # Total number of parameters
    elif residual_layer == 'mod_UNIQUAC':
        print(f'Residual {residual_layer}, setting Q as variable')
        param_dim_list.extend([dim_active + q_dim])
        col_names.extend(['q'])
        dim_active = param_dim_list[-1] # Total number of parameters
    
    if association_layer == 'wertheim':
        print(f'Association {association_layer}, setting Delta as variable')
        param_dim_list.extend([dim_active + 1, dim_active + 2]) # The spacing of the dimensions
        col_names.extend(['acceptor', 'donor'])
        dim_active = param_dim_list[-1] # Total number of parameters

    # Set up the optimization
    criterion = nn.L1Loss() # Use MAE as loss criteria
    grad_criterion = MAPELoss() # Use MAPE as gradient loss criteria for Sobolev

    print(f'Save filename is {save_name} with {dim_active} dimensions')

    # Perform data truncation
    node_degree_threshold = dim * 2 # Two times the dimension of the embedding
    max_iterations = 1000
    iteration_count = 0

    full_node_list = list(pd.unique(df_inf[[solute_smiles, solvent_smiles]].values.ravel('K')))
    nodes_original = np.shape(full_node_list)

    if truncation == 'chemical_connections':
        while iteration_count < max_iterations:
            # Get a list of all unique nodes (Solute + Solvent combined)
            full_node_list = list(pd.unique(df_inf[[solute_smiles, solvent_smiles]].values.ravel('K')))

            # Calculate the node degree, considering only unique solute-solvent pairs
            degree = calculate_node_degree_chemical(df_inf, full_node_list, solute_smiles, solvent_smiles)
            
            nodes_to_remove = degree[degree < node_degree_threshold].index.tolist()

            if not nodes_to_remove:
                break

            # Filter out rows containing nodes to remove
            df_inf = df_inf[~df_inf[solute_smiles].isin(nodes_to_remove) & ~df_inf[solvent_smiles].isin(nodes_to_remove)]

            iteration_count += 1

        final_count = calculate_node_degree_chemical(df_inf, full_node_list, solute_smiles, solvent_smiles)
        print(f'Nodes suppressed due to under-definition. Only directional edges counted as unique. Out of original {nodes_original[0]} nodes, we have {np.shape(final_count)[0]} nodes')

        full_node_list = list(pd.unique(df_inf[[solute_smiles, solvent_smiles]].values.ravel('K')))
        node_idx_dict = dict(zip(np.array(full_node_list) , np.array(range(np.size(full_node_list)))))

        # Add two columns, one for the index of solute and one for the index of solvent in the parameter array
        df_inf['Solute_Idx'] = df_inf[solute_smiles].map(node_idx_dict)
        df_inf['Solvent_Idx'] = df_inf[solvent_smiles].map(node_idx_dict)

        edges = df_inf.shape
        nodes = np.shape(full_node_list)
        n_node = np.size(full_node_list) # full graph node size

    elif truncation == 'temp_connections':
        while iteration_count < max_iterations:
            # Get a list of all unique nodes (Solute + Solvent combined)
            full_node_list = list(pd.unique(df_inf[[solute_smiles, solvent_smiles]].values.ravel('K')))
            
            # Calculate the node degree, considering only unique solute-solvent pairs but treating different temperatures as unique
            degree = calculate_node_degree_temperature(df_inf, full_node_list, solute_smiles, solvent_smiles, 'Temp (K)')
            
            # Identify nodes to remove (those below the degree threshold)
            nodes_to_remove = degree[degree < node_degree_threshold].index.tolist()

            if not nodes_to_remove:
                break

            # Filter out rows containing nodes to remove from either solute or solvent
            df_inf = df_inf[~df_inf[solute_smiles].isin(nodes_to_remove) & ~df_inf[solvent_smiles].isin(nodes_to_remove)]

            iteration_count += 1
        
        # Calculate final node degree after all iterations
        final_count = calculate_node_degree_chemical(df_inf, full_node_list, solute_smiles, solvent_smiles)
        print(f'Nodes suppressed due to under-definition. Each temperature connection is counted as a unique entry. Out of original {nodes_original[0]} nodes, we have {np.shape(final_count)[0]} nodes')

        # Generate a list of unique nodes
        full_node_list = list(pd.unique(df_inf[[solute_smiles, solvent_smiles]].values.ravel('K')))
        
        # Create a mapping of node to index
        node_idx_dict = dict(zip(np.array(full_node_list), np.array(range(np.size(full_node_list)))))

        # Add index columns for solute and solvent
        df_inf['Solute_Idx'] = df_inf[solute_smiles].map(node_idx_dict)
        df_inf['Solvent_Idx'] = df_inf[solvent_smiles].map(node_idx_dict)

        edges = df_inf.shape
        nodes = np.shape(full_node_list)
        n_node = np.size(full_node_list)  # full graph node size

    elif truncation == 'keep_all':
        # Generate a list of unique nodes
        full_node_list = list(pd.unique(df_inf[[solute_smiles, solvent_smiles]].values.ravel('K')))
        final_count = calculate_node_degree_chemical(df_inf, full_node_list, solute_smiles, solvent_smiles)
        print(f'All nodes kept')

        # Create a mapping of node to index
        node_idx_dict = dict(zip(np.array(full_node_list), np.array(range(np.size(full_node_list)))))

        edges = df_inf.shape
        nodes = np.shape(full_node_list)
        n_node = np.size(full_node_list)  # full graph node size

    else:
        print('Data reduction method invalid')

    print(f'Regressing on {edges[0]} data points and {nodes[0]} unique chemicals')

    # Set up data as tensors for regression
    i_data = df_inf['Solute_Idx'].to_numpy()
    j_data = df_inf['Solvent_Idx'].to_numpy()
    invT_data = 1 / df_inf['Temp (K)'].to_numpy()
    y_data = df_inf[ln_y_data].to_numpy() # This is the IDAC data after removing the combinatorial layer contribution
    rA_data = df_inf['Solute_VDW_Volumes'].to_numpy() / v0
    rB_data = df_inf['Solvent_VDW_Volumes'].to_numpy() / v0
    qA_data = df_inf['Solute_VDW_Area'].to_numpy() / s0
    qB_data = df_inf['Solvent_VDW_Area'].to_numpy() / s0
    N_d_A = df_inf['Solute_H_Donor_Sites'].astype(float).to_numpy()
    N_a_A = df_inf['Solute_H_Acceptor_Sites'].astype(float).to_numpy()
    N_d_B = df_inf['Solvent_H_Donor_Sites'].astype(float).to_numpy()
    N_a_B = df_inf['Solvent_H_Acceptor_Sites'].astype(float).to_numpy()
    rhoBam = N_a_B / rB_data
    rhoBdm = N_d_B / rB_data
    rhoAap = N_a_A / rA_data
    rhoAdp = N_d_A / rA_data
    N_data = np.column_stack((N_a_A, N_d_A))
    rho_data = np.column_stack((rhoAap, rhoAdp, rhoBam, rhoBdm))

    # Extract the normalized surface area of each molecule. This serves as an initial guess for the mod-UNIQUAC model Q
    q = df_prop.loc[full_node_list, 'van der waals area (m2/kmol)'].to_numpy() / s0

    # Push data to GPU
    early_stopping = EarlyStopper(patience = 1000) # Set up early stopper
    i = torch.tensor(i_data).to(device)
    j = torch.tensor(j_data).to(device)
    invT = torch.tensor(invT_data[:, None]).to(device) # Must be Ndata x 1
    y = torch.tensor(y_data[:, None]).to(device)
    rA = torch.tensor(rA_data[:, None]).to(device)
    qA = torch.tensor(qA_data[:, None]).to(device)
    qB = torch.tensor(qB_data[:, None]).to(device)
    N = torch.tensor(N_data, dtype = torch.float32).to(device)
    rho = torch.tensor(rho_data, dtype = torch.float32).to(device)

    # Get the temperature gradient data only if needed
    if sobolev > 0:
        print(f'Using sobolev penalty with weight : {sobolev}')
        dy_dinvT_data = df_inf['dlny_dinvT'].to_numpy()
        dy_dinvT = torch.tensor(dy_dinvT_data).to(device)

    # Launch model for different trials
    for trial_idx in range(trials):
        np.random.seed(trial_idx) # Set rng seed

        loss_values = []

        # Set up initial guesses
        Alpha_shape = (n_node, alpha_dim) # For NRTL
        A_shape = (n_node, u_dim) # For all models
        Q_shape = (n_node, 1) # Single dimension, for mod-UNIQUAC
        D_shape = (n_node, 2) # For Wertheim
        T_shape = (u_dim, temp_dim) # Temperature layer learned matrix
        O_shape = (u_dim, 2) # Combining layer learned matrix

        # This is the section for setting up initial guesses
        A_initial = 0.1 * 20 * np.random.standard_normal(size = A_shape) + 20 # This makes all values large, and makes it 10% of the guess size
        Alpha_initial = np.random.random_sample(size = Alpha_shape) # From [0, 1)
        Q_initial = np.zeros(Q_shape) # This must be q to start
        Q_initial[:, 0] = q.copy()

        # T is pass through activation function. Softmax is log orders of magnitude.
        if temp_dim == 1:
            # Only 1 / T, doesn't matter what T is
            T_initial = np.ones(T_shape)
        else:
            # Set it so that each column is centered 2 orders of magnitude apart (so if 3 terms, then we get -2, 0, 2)
            T_initial = 0.1 * np.random.standard_normal(size = T_shape) + 1 # Centered around 1
            for col, exponent in enumerate(temp_exponents):
                T_initial[:, col] = np.log(T_initial[:, col] * 10 ** (2 * exponent))

        D_initial = 0.1 * np.random.standard_normal(size = D_shape) + np.log(np.exp(1) - 1) * np.ones(D_shape) # Narrow random distribution around 1 (after softmax)
        O_initial = np.random.random_sample(size = O_shape) # From [0, 1)

        # Select model here and push to GPU
        model = model_selector(residual_layer, A_initial, Alpha_initial, Q_initial, T_initial, O_initial, D_initial, association_layer, temp_exponents, device).to(device)
        
        # If Wertheim, use pretraining of the model parameters
        if association_layer == 'wertheim':
            # Pretrain the residual and association separately
            # Train the association layer first
            model.D = nn.Parameter(torch.tensor(np.zeros_like(D_initial), dtype = torch.float32).to(device)) # Turn 'off' the Wertheim by setting it to zero (smaller value)
            optimizer_wertheim = optim.Adam(model.parameters(), lr = 0.01)
            model.train()
            for epoch in range(pre_train_epoch):
                optimizer_wertheim.zero_grad()

                outputs = model(i, j, invT, rA, qA, qB, N, rho)

                loss = criterion(torch.squeeze(outputs), torch.squeeze(y))

                loss.backward()
                
                for param in model.parameters():
                    if param is model.D:
                        param.grad = None
                    # Don't let the Q gradient be updated either
                    if residual_layer == 'mod_UNIQUAC':
                        if param is model.Q:
                            param.grad = None

                optimizer_wertheim.step()

                if (epoch + 1) % 500 == 0:
                    print(f'Pre-training of Other parameters : Trial {trial_idx}, Epoch [{epoch+1}/{pre_train_epoch}], Loss: {loss.item():.4f}')

            # Now train the residual layer
            # Bring back the Wertheim layer
            model.D = nn.Parameter(torch.tensor(D_initial, dtype = torch.float32).to(device))
            optimizer_wertheim = optim.Adam(model.parameters(), lr = 0.01)
            model.train()
            for epoch in range(pre_train_epoch):
                optimizer_wertheim.zero_grad()

                outputs = model(i, j, invT, rA, qA, qB, N, rho)

                loss = criterion(torch.squeeze(outputs), torch.squeeze(y))

                loss.backward()
                
                for param in model.parameters():
                    if param is not model.D:
                        param.grad = None

                optimizer_wertheim.step()

                if (epoch + 1) % 500 == 0:
                    print(f'Pre-training of Delta : Trial {trial_idx}, Epoch [{epoch+1}/{pre_train_epoch}], Loss: {loss.item():.4f}')

        # Main training
        # Define the optimizer
        optimizer = optim.Adam(model.parameters(), lr = lr)#, weight_decay=reg)

        # Change learning rate
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda epoch: up_down_lr_schedule(epoch, low_factor = 0.1, total_steps = total_epochs, up_steps = up_epochs, hold_steps = hold_epochs))

        model.train()

        # Two options depending if sobolev is required or not
        if sobolev == 0:
            # No sobolev loss
            for epoch in range(total_epochs):
                optimizer.zero_grad()

                # Forward pass
                outputs = model(i, j, invT, rA, qA, qB, N, rho)

                # Compute the loss between predicted and target values without Sobolev regularization
                loss = criterion(torch.squeeze(outputs), torch.squeeze(y))

                # Backward pass, comput gradients
                loss.backward()
                    
                # Update parameters            
                optimizer.step()

                # Change learning rate
                scheduler.step()

                # Append loss to the array
                loss_values.append(loss.item())
                
                early_stopping(loss.item())

                # Print message every 500th iteration
                if (epoch + 1) % 500 == 0:
                    print(f'Trial {trial_idx}, Epoch [{epoch+1}/{total_epochs}], Loss: {loss.item():.4f}')

                # Early stopping check
                if early_stopping.early_stop:
                    print(f"Early stopping occurred at Epoch [{epoch + 1}/{total_epochs}], Loss: {loss.item():.4f}")
                    break
                
        else:
            # Sobolev loss
            for epoch in range(total_epochs):
                optimizer.zero_grad()
                
                # Require gradient of invT
                invT.requires_grad = True

                # Forward pass
                outputs = model(i, j, invT, rA, qA, qB, N, rho)
                output_dinvT = torch.autograd.grad(outputs=outputs, inputs= invT, grad_outputs=torch.ones_like(outputs), create_graph=True)[0]

                # Compute the loss between predicted and target values
                loss_lny = criterion(torch.squeeze(outputs), torch.squeeze(y))

                # Only pass in output_dT and dy_dT if dy_dT is not nan
                mask = ~torch.isnan(dy_dinvT)
                loss_sobolev = grad_criterion(torch.squeeze(output_dinvT[mask]), torch.squeeze(dy_dinvT[mask]))

                loss = loss_lny + sobolev * loss_sobolev

                # Backward pass, compute gradients
                loss.backward(retain_graph=True)

                # Update parameters
                optimizer.step()

                # Change learning rate
                scheduler.step()

                # Append loss to the array
                loss_values.append(loss.item())

                # Print message every 500th iteration
                if (epoch + 1) % 500 == 0:
                    print(f'Trial {trial_idx}, Epoch [{epoch+1}/{total_epochs}], Loss: {loss.item():.4f}, ln y Loss: {loss_lny.item():.4f}, Sobolev Loss: {loss_sobolev.item():.4f}')

                # Do not check early stopping, because the Sobolev loss term tends to hop around a lot near the beginning of training
                    
        # Training complete. Set the model in eval mode and save the model parameters
        model.eval()
        save_model(trial_idx, model, loss_values, save_name, model_layer_options, node_idx_dict, col_names, param_dim_list)

def save_model(trial, model, loss_values, save_name, model_layer_options, node_idx_dict, col_names, param_dim_list):
    # Unpack model options
    ln_y_data = model_layer_options['ln_y_data']
    combinatorial_layer = model_layer_options['combinatorial_layer']
    residual_layer = model_layer_options['residual_layer']
    association_layer = model_layer_options['association_layer']
    temp_exponents = model_layer_options['temp_exponents']

    # Transfer tensors to array
    # All models have A, O, T
    A = model.A.detach().cpu().numpy()
    T = model.T.detach().cpu().numpy()
    O = model.O.detach().cpu().numpy()
    Alpha = np.array([])
    Q = np.array([])
    D = np.array([])

    # Extract alpha for NRTL, Q for mod-UNIQUAC
    if residual_layer == 'NRTL':
        Alpha = model.Alpha.detach().cpu().numpy()
    elif residual_layer == 'mod_UNIQUAC':
        Q = model.Q.detach().cpu().numpy()
    
    # Extract Delta for Wertheim
    if association_layer == 'wertheim':
        D = model.D.detach().cpu().numpy()

    # Save the model parameters using state_dict, it is a .pt file
    model_filename = f'{save_name}_{trial}.pt'
    torch.save(model.state_dict(), model_filename)

    # Save the model parameters as h5 file format. We need to merge the array first
    param_array = [arr for arr in (A, Alpha, Q, D) if arr.size > 0]
    merged_array = np.hstack(param_array)

    # Array index
    index_series = pd.Series(node_idx_dict)
    # Initialize the new column names list
    new_column_names = []
    # Generate the new column names
    start = 0
    for name, end in zip(col_names, param_dim_list):
        for idx in range(start, end):
            new_column_names.append(f"{name}_{idx - start + 1}")
        start = end
    # Add remaining columns if there are any - error catch
    for idx in range(start, merged_array.shape[1]):
        new_column_names.append(f"unspecified_{idx - start + 1}")

    df_results = pd.DataFrame(merged_array, index = index_series.index, columns = new_column_names)
    df_distance = pd.DataFrame(O)
    df_temp = pd.DataFrame(T, columns = temp_exponents)
    df_loss = pd.DataFrame(loss_values)

    # Save all DataFrames as an HDF5 file
    h5_filename = f'{save_name}_{trial}.h5'
    with pd.HDFStore(h5_filename) as store:
        store.put('df_chemical_parameters', df_results)
        store.put('df_temp_parameters', df_temp)
        store.put('df_loss', df_loss)
        store.put('df_distance', df_distance)

    print(f'Model saved as {model_filename} and {h5_filename}')
        
