import numpy as np
import pandas as pd
from bct.utils import cuberoot, BCTParamError, dummyvar, binarize, get_rng
import bct
import netneurotools
import os
import matplotlib.pyplot as plt
from sklearn.utils.validation import check_random_state
import matplotlib.patches as patches
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score


def find_consensus_2(assignments, null_func=np.mean, return_agreement=True,
                   gamma=1.0, seed=None): # Added gamma parameter with default value 1.0
    """
    Find consensus clustering labels from cluster solutions in `assignments`.

    Parameters
    ----------
    assignments : (N, M) array_like
        Array of `M` clustering solutions for `N` samples (e.g., subjects,
        nodes, etc). Values of array should be integer-based cluster assignment
        labels
    null_func : callable, optional
        Function used to generate null model when performing consensus-based
        clustering. Must accept a 2D array as input and return a single value.
        Default: :func:`numpy.mean`
    return_agreement : bool, optional
        Whether to return the thresholded N x N agreement matrix used in
        generating the final consensus clustering solution. Default: False
    gamma : float, optional
        Resolution parameter for consensus clustering. Default is 1.0
    seed : {int, np.random.RandomState instance, None}, optional
        Seed for random number generation. Used when permuting cluster
        assignments during generation of null model. Default: None

    Returns
    -------
    consensus : (N,) numpy.ndarray
        Consensus cluster labels
    agreement : (N, N) numpy.ndarray
        Agreement matrix

    References
    ----------
    Bassett, D. S., Porter, M. A., Wymbs, N. F., Grafton, S. T., Carlson,
    J. M., & Mucha, P. J. (2013). Robust detection of dynamic community
    structure in networks. Chaos: An Interdisciplinary Journal of Nonlinear
    Science, 23(1), 013142.
    """
    rs = check_random_state(seed)
    samp, comm = assignments.shape

    # create agreement matrix from input community assignments and convert to
    # probability matrix by dividing by `comm`
    agreement = bct.clustering.agreement(assignments, buffsz=samp) / comm

    # generate null agreement matrix and use to create threshold
    null_assign = np.column_stack([rs.permutation(i) for i in assignments.T])
    null_agree = bct.clustering.agreement(null_assign, buffsz=samp) / comm
    threshold = null_func(null_agree)

    # run consensus clustering on agreement matrix after thresholding
    consensus = consensus_und_2(agreement, threshold, 300, gamma=gamma, seed=seed) # Pass gamma here

    if return_agreement:
        return consensus.astype(int), agreement * (agreement > threshold)

    return consensus.astype(int)
def consensus_und_2(D, tau, reps=300, gamma=1.0, seed=None): # Added gamma parameter with default value 1.0
    '''
    This algorithm seeks a consensus partition of the
    agreement matrix D. The algorithm used here is almost identical to the
    one introduced in Lancichinetti & Fortunato (2012): The agreement
    matrix D is thresholded at a level TAU to remove an weak elements. The
    resulting matrix is then partitions REPS number of times using the
    Louvain algorithm (in principle, any clustering algorithm that can
    handle weighted matrixes is a suitable alternative to the Louvain
    algorithm and can be substituted in its place). This clustering
    produces a set of partitions from which a new agreement is built. If
    the partitions have not converged to a single representative partition,
    the above process repeats itself, starting with the newly built
    agreement matrix.

    NOTE: In this implementation, the elements of the agreement matrix must
    be converted into probabilities.

    NOTE: This implementation is slightly different from the original
    algorithm proposed by Lanchichinetti & Fortunato. In its original
    version, if the thresholding produces singleton communities, those
    nodes are reconnected to the network. Here, we leave any singleton
    communities disconnected.

    Parameters
    ----------
    D : NxN np.ndarray
        agreement matrix with entries between 0 and 1 denoting the probability
        of finding node i in the same cluster as node j
    tau : float
        threshold which controls the resolution of the reclustering
    reps : int
        number of times the clustering algorithm is reapplied. default value
        is 1000.
    gamma : float, optional
        Resolution parameter for Louvain algorithm. Default is 1.0
    seed : hashable, optional
        If None (default), use the np.random's global random state to generate random numbers.
        Otherwise, use a new np.random.RandomState instance seeded with the given value.

    Returns
    -------
    ciu : Nx1 np.ndarray
        consensus partition
    '''
    rng = get_rng(seed)
    def unique_partitions(cis):
        # relabels the partitions to recognize different numbers on same
        # topology

        n, r = np.shape(cis)  # ci represents one vector for each rep
        ci_tmp = np.zeros(n)

        for i in range(r):
            for j, u in enumerate(sorted(
                    np.unique(cis[:, i], return_index=True)[1])):
                ci_tmp[np.where(cis[:, i] == cis[u, i])] = j
            cis[:, i] = ci_tmp
            # so far no partitions have been deleted from ci

        # now squash any of the partitions that are completely identical
        # do not delete them from ci which needs to stay same size, so make
        # copy
        ciu = []
        cis = cis.copy()
        c = np.arange(r)
        # count=0
        while (c != 0).sum() > 0:
            ciu.append(cis[:, 0])
            dup = np.where(np.sum(np.abs(cis.T - cis[:, 0]), axis=1) == 0)
            cis = np.delete(cis, dup, axis=1)
            c = np.delete(c, dup)
            # count+=1
            # print count,c,dup
            # if count>10:
            #	class QualitativeError(): pass
            #	raise QualitativeError()
        return np.transpose(ciu)

    n = len(D)
    flag = True
    while flag:
        flag = False
        dt = D * (D >= tau)
        np.fill_diagonal(dt, 0)

        if np.size(np.where(dt == 0)) == 0:
            ciu = np.arange(1, n + 1)
        else:
            cis = np.zeros((n, reps))
            for i in np.arange(reps):
                cis[:, i], _ = bct.modularity.modularity_louvain_und_sign(dt, gamma=gamma, seed=rng) # Use dynamic gamma here
            ciu = unique_partitions(cis)
            nu = np.size(ciu, axis=1)
            if nu > 1:
                flag = True
                D = bct.clustering.agreement(cis) / reps

    return np.squeeze(ciu + 1)

def _grid_communities(communities):
    """
    Generate boundaries of `communities`.

    Parameters
    ----------
    communities : array_like
        Community assignment vector

    Returns
    -------
    bounds : list
        Boundaries of communities
    """
    communities = np.asarray(communities)
    if 0 in communities:
        communities = communities + 1

    comm = communities[np.argsort(communities)]
    bounds = []
    for i in np.unique(comm):
        ind = np.where(comm == i)
        if len(ind) > 0:
            bounds.append(np.min(ind))

    bounds.append(len(communities))

    return bounds

def sort_communities(consensus, communities):
    """
    Sort `communities` in `consensus` according to strength.

    Parameters
    ----------
    consensus : array_like
        Correlation matrix
    communities : array_like
        Community assignments for `consensus`

    Returns
    -------
    inds : np.ndarray
        Index array for sorting `consensus`
    """
    communities = np.asarray(communities)
    if 0 in communities:
        communities = communities + 1

    bounds = _grid_communities(communities)
    inds = np.argsort(communities)

    for n, f in enumerate(bounds[:-1]):
        i = inds[f:bounds[n + 1]]
        cco = i[consensus[np.ix_(i, i)].mean(axis=1).argsort()[::-1]]
        inds[f:bounds[n + 1]] = cco

    return inds
def plot_mod_heatmap(data, communities, *, inds=None, edgecolor='black',
                     ax=None, figsize=(6.4, 4.8), xlabels=None, ylabels=None,
                     xlabelrotation=90, ylabelrotation=0, cbar=True,
                     square=True, xticklabels=None, yticklabels=None,
                     mask_diagonal=True, **kwargs):
    """
    Plot `data` as heatmap with borders drawn around `communities`.

    Parameters
    ----------
    data : (N, N) array_like
        Correlation matrix
    communities : (N,) array_like
        Community assignments for `data`
    inds : (N,) array_like, optional
        Index array for sorting `data` within `communities`. If None, these
        will be generated from `data`. Default: None
    edgecolor : str, optional
        Color for lines demarcating community boundaries. Default: 'black'
    ax : matplotlib.axes.Axes, optional
        Axis on which to plot the heatmap. If none provided, a new figure and
        axis will be created. Default: None
    figsize : tuple, optional
        Size of figure to create if `ax` is not provided. Default: (20, 20)
    {x,y}labels : list, optional
        List of labels on {x,y}-axis for each community in `communities`. The
        number of labels should match the number of unique communities.
        Default: None
    {x,y}labelrotation : float, optional
        Angle of the rotation of the labels. Available only if `{x,y}labels`
        provided. Default : xlabelrotation: 90, ylabelrotation: 0
    square : bool, optional
        Setting the matrix with equal aspect. Default: True
    {x,y}ticklabels : list, optional
        Incompatible with `{x,y}labels`. List of labels for each entry (not
        community) in `data`. Default: None
    cbar : bool, optional
        Whether to plot colorbar. Default: True
    mask_diagonal : bool, optional
        Whether to mask the diagonal in the plotted heatmap. Default: True
    kwargs : key-value mapping
        Keyword arguments for `plt.pcolormesh()`

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axis object containing plot
    """
    for t, label in zip([xticklabels, yticklabels], [xlabels, ylabels]):
        if t is not None and label is not None:
            raise ValueError('Cannot set both {x,y}labels and {x,y}ticklabels')

    # get indices for sorting consensus
    if inds is None:
        inds = sort_communities(data, communities)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    # plot data re-ordered based on community and node strength
    if mask_diagonal:
        plot_data = np.ma.masked_where(np.eye(len(data)),
                                       data[np.ix_(inds, inds)])
    else:
        plot_data = data[np.ix_(inds, inds)]

    coll = ax.pcolormesh(plot_data, edgecolor='none', **kwargs)
    ax.set(xlim=(0, plot_data.shape[1]), ylim=(0, plot_data.shape[0]))

    # set equal aspect
    if square:
        ax.set_aspect('equal')

    for side in ['top', 'right', 'left', 'bottom']:
        ax.spines[side].set_visible(False)

    # invert the y-axis so it looks 
    ax.invert_yaxis()

    # plot the colorbar
    if cbar:
        cb = ax.figure.colorbar(coll)
        if kwargs.get('rasterized', False):
            cb.solids.set_rasterized(True)

    # draw borders around communities
    bounds = _grid_communities(communities)
    bounds[0] += 0.2
    bounds[-1] -= 0.2
    for n, edge in enumerate(np.diff(bounds)):
        ax.add_patch(patches.Rectangle((bounds[n], bounds[n]),
                                       edge, edge, fill=False, linewidth=1,
                                       edgecolor=edgecolor))

    if xlabels is not None or ylabels is not None:
        # find the tick locations
        initloc = _grid_communities(communities)
        tickloc = []
        for loc in range(len(initloc) - 1):
            tickloc.append(np.mean((initloc[loc], initloc[loc + 1])))

        if xlabels is not None:
            # make sure number of labels match the number of ticks
            if len(tickloc) != len(xlabels):
                raise ValueError('Number of labels do not match the number of '
                                 'unique communities.')
            else:
                ax.set_xticks(tickloc)
                ax.set_xticklabels(labels=xlabels, rotation=xlabelrotation)
                ax.tick_params(left=False, bottom=False)
        if ylabels is not None:
            # make sure number of labels match the number of ticks
            if len(tickloc) != len(ylabels):
                raise ValueError('Number of labels do not match the number of '
                                 'unique communities.')
            else:
                ax.set_yticks(tickloc)
                ax.set_yticklabels(labels=ylabels, rotation=ylabelrotation)
                ax.tick_params(left=False, bottom=False)

    if xticklabels is not None:
        labels_ind = [xticklabels[i] for i in inds]
        ax.set_xticks(np.arange(len(labels_ind)) + 0.5)
        ax.set_xticklabels(labels_ind, rotation=90)
    if yticklabels is not None:
        labels_ind = [yticklabels[i] for i in inds]
        ax.set_yticks(np.arange(len(labels_ind)) + 0.5)
        ax.set_yticklabels(labels_ind)

    return ax

data_dir = '/mnt/munin/Morey/Lab/ahmed/cerebellum/data/non_residualized/SCH/control'
output_dir = '/mnt/munin/Morey/Lab/ahmed/cerebellum/data/Outputs/testing/louvain_non_resid/gamma_1_to_2'

# Pre-determine array sizes
n_subjects = 560
n_runs = 300

def process_subject(subject_file, gamma, n_runs=n_runs, use_mst=True):
    # Load and prepare data
    data = np.load(os.path.join(data_dir, subject_file))
    data_clean = np.nan_to_num(data, nan=0)
    data_symmetric = (data_clean + data_clean.T) / 2
    np.fill_diagonal(data_symmetric, 1)
    nonegative = data_symmetric.copy()
    nonegative[nonegative < 0] = 0

    if use_mst:
        # Apply MST to the nonnegative matrix
        mst_matrix = minimum_spanning_tree(nonegative*-1)*-1
        mst_matrix = mst_matrix.toarray()  # Convert sparse matrix to dense array
        # Ensure symmetry
        mst_matrix = (mst_matrix + mst_matrix.T) / 2
        # Use MST matrix for community detection
        cis = []
        modularities = []
        for _ in range(n_runs):
            ci, q = bct.community_louvain(mst_matrix, gamma=gamma, seed=_) 
            cis.append(ci)
            modularities.append(q)

    else:
        # Use original nonnegative matrix
        cis = []
        modularities = []
        for _ in range(n_runs):
            ci, q = bct.community_louvain(nonegative, gamma=gamma, seed=_) 
            cis.append(ci)
            modularities.append(q)

    # If you want to keep the agreement matrix
    consensus_result = find_consensus_2(np.column_stack(cis), gamma=gamma, seed=_, return_agreement=True)
    consensus_labels, agreement_matrix = consensus_result  # Unpack the tuple
            
    return consensus_labels, nonegative, cis, modularities


# Process subjects using list comprehension
subject_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npy')])[:n_subjects]
#conn_matrix = results[0][1]  # Get connectivity matrix from first subject
# Gamma range
gamma_range = np.array([i/10 for i in range(10, 21)])
#gamma_range = np.array([i/10 for i in range(10, 12)]) # gamma = 1.0 & 1.1
n_gamma = len(gamma_range)

# Initialize lists to store metrics
avg_consensus_modularities = []
std_consensus_modularities = []
avg_nmi = []
avg_ari = []
avg_ami = []
final_agreement_matrix_modularities = [] 

# Data storage
all_metrics = []


for gamma in gamma_range:
    print(f"Processing gamma = {gamma}")
    all_subject_cis = []
    all_subject_modularities = []
    all_subject_consensus_partitions = []
    subject_consensus_partition_modularities = [] # Store modularity of each subject's consensus partition
    results = [] # To store results for each subject to access conn_matrix later

    for subject_file in subject_files:
        consensus_partition, conn_matrix, cis, modularities = process_subject(subject_file, gamma, use_mst=True)
        results.append((consensus_partition, conn_matrix, cis, modularities)) # Store results
        all_subject_cis.append(cis)
        all_subject_modularities.append(modularities)
        all_subject_consensus_partitions.append(consensus_partition)
        # Add a check to debug size mismatch
        print(f"conn_matrix shape: {conn_matrix.shape}")
        print(f"consensus_partition length: {len(consensus_partition)}")

        # Only calculate if sizes match
        if len(consensus_partition) == conn_matrix.shape[0]:
            subject_consensus_modularity = bct.community_louvain(conn_matrix, gamma=gamma, ci=consensus_partition)[1]
            subject_consensus_partition_modularities.append(subject_consensus_modularity)
        else:
            print(f"Size mismatch for subject {subject_file}! Skipping modularity calculation.")
            # Append a placeholder or NaN value
            subject_consensus_partition_modularities.append(float('nan'))
        # Calculate modularity of subject's consensus partition
        # Use the conn_matrix from the current subject, not from a previous run
        subject_consensus_modularity = bct.community_louvain(conn_matrix, gamma=gamma, ci=consensus_partition)[1]
        subject_consensus_partition_modularities.append(subject_consensus_modularity)


    # Calculate average and std of consensus modularities
    avg_consensus_modularities.append(np.mean(subject_consensus_partition_modularities))
    std_consensus_modularities.append(np.std(subject_consensus_partition_modularities))


    # Calculate NMI, ARI, AMI (same as before, between runs within each subject)
    nmi_values = []
    ari_values = []
    ami_values = []
    for subject_cis in all_subject_cis:
        for i in range(n_runs):
            for j in range(i + 1, n_runs):
                nmi_values.append(normalized_mutual_info_score(subject_cis[i], subject_cis[j]))
                ari_values.append(adjusted_rand_score(subject_cis[i], subject_cis[j]))
                ami_values.append(adjusted_mutual_info_score(subject_cis[i], subject_cis[j]))
    avg_nmi.append(np.mean(nmi_values))
    avg_ari.append(np.mean(ari_values))
    avg_ami.append(np.mean(ami_values))

    # Final consensus and Agreement Matrix using find_consensus_2
    final_consensus, final_agreement_matrix = find_consensus_2(np.column_stack(all_subject_consensus_partitions), gamma=gamma) # Pass gamma here
    final_agreement_modularity = bct.community_louvain(final_agreement_matrix, gamma=gamma)[1] # Modularity of final agreement matrix
    final_agreement_matrix_modularities.append(final_agreement_modularity) # Append final agreement matrix modularity
    
    # Save the final agreement matrix
    filename = os.path.join(output_dir, f"agreement_matrix_gamma_{gamma:.4f}.npy")
    np.save(filename, final_agreement_matrix)
    print(f"Saved agreement matrix for gamma = {gamma} to {filename}")

    # Store metrics for saving
    all_metrics.append({
        'gamma': gamma,
        'avg_consensus_modularity': avg_consensus_modularities[-1],
        'std_consensus_modularity': std_consensus_modularities[-1],
        'avg_nmi': avg_nmi[-1],
        'avg_ari': avg_ari[-1],
        'avg_ami': avg_ami[-1],
        'final_agreement_matrix_modularity': final_agreement_matrix_modularities[-1]
    })

# Process each gamma value
for gamma in gamma_range:
    print(f"\nProcessing gamma = {gamma}")
    
    # Create a gamma-specific output directory
    gamma_output_dir = os.path.join(output_dir, f"gamma_{gamma:.4f}")
    if not os.path.exists(gamma_output_dir):
        os.makedirs(gamma_output_dir)
    
    # Get consensus partition for current gamma
    final_consensus, final_agreement_matrix = find_consensus_2(np.column_stack(all_subject_consensus_partitions), gamma=gamma)
    
    # Analyze community sizes
    unique_communities, community_sizes = np.unique(final_consensus, return_counts=True)
    large_communities = community_sizes[community_sizes > 5]
    
    print(f"Number of communities with >5 members: {len(large_communities)}")
    print("\nCommunity sizes (>5 members):")
    for comm, size in zip(unique_communities[community_sizes > 5], large_communities):
        print(f"Community {comm}: {size} members")
    
    # Save to log file
    with open(os.path.join(gamma_output_dir, f'community_sizes_gamma_{gamma:.4f}.txt'), 'w') as f:
        f.write(f"Number of communities with >5 members: {len(large_communities)}\n\n")
        f.write("Community sizes (>5 members):\n")
        for comm, size in zip(unique_communities[community_sizes > 5], large_communities):
            f.write(f"Community {comm}: {size} members\n")
    
    # Read the node labels
    node_labels = pd.read_csv('/mnt/munin/Morey/Lab/ahmed/cerebellum/data/SCH_region_order.csv', header=None)
    node_labels = node_labels[0].tolist()  
    
    # Create a DataFrame with node assignments
    partition_df = pd.DataFrame({
        'Node': node_labels,
        'Community': final_consensus,
        'Gamma': gamma  # Include gamma value in the dataframe
    })
    
    # Save to CSV
    partition_df.to_csv(os.path.join(gamma_output_dir, f'node_community_assignments_gamma_{gamma:.4f}.csv'), index=False)
    
    # Save community memberships
    with open(os.path.join(gamma_output_dir, f'community_memberships_gamma_{gamma:.4f}.txt'), 'w') as f:
        f.write(f"Community Memberships (gamma = {gamma}):\n\n")
        for comm in unique_communities:
            nodes_in_comm = partition_df[partition_df['Community'] == comm]['Node'].tolist()
            f.write(f"Community {comm} ({len(nodes_in_comm)} members):\n")
            f.write("\n".join(f"  {node}" for node in nodes_in_comm))
            f.write("\n\n")
    
    # Save agreement matrix
    np.save(os.path.join(gamma_output_dir, f'agreement_matrix_gamma_{gamma:.4f}.npy'), final_agreement_matrix)

# Combined Modularity Plot
fig_mod, ax_mod = plt.subplots(figsize=(10, 6))
ax_mod.errorbar(gamma_range, avg_consensus_modularities, yerr=std_consensus_modularities, fmt='-o', label='Average Consensus Modularity')
ax_mod.plot(gamma_range, final_agreement_matrix_modularities, '-o', label='Final Agreement Matrix Modularity')
ax_mod.set_xlabel('Gamma')
ax_mod.set_ylabel('Modularity')
ax_mod.set_title('Modularity vs Gamma')
ax_mod.grid(True)
ax_mod.legend()
fig_mod.savefig(os.path.join(output_dir, 'clustering_metrics_gamma_scan_combined_modularity.jpeg'), bbox_inches='tight')
plt.close(fig_mod)

# Combined Similarity Indices Plot
fig_sim, ax_sim = plt.subplots(figsize=(10, 6))
ax_sim.plot(gamma_range, avg_nmi, '-o', label='NMI', color='blue')
ax_sim.plot(gamma_range, avg_ari, '-o', label='ARI', color='green')
ax_sim.plot(gamma_range, avg_ami, '-o', label='AMI', color='red')
ax_sim.set_xlabel('Gamma')
ax_sim.set_ylabel('Similarity Index Value')
ax_sim.set_title('Similarity Indices vs Gamma')
ax_sim.grid(True)
ax_sim.legend()
fig_sim.savefig(os.path.join(output_dir, 'clustering_metrics_gamma_scan_combined_similarity.jpeg'), bbox_inches='tight')
plt.close(fig_sim)


# Save plots separately (individual metrics plots)
plot_filename_prefix = os.path.join(output_dir, 'clustering_metrics_gamma_scan')

# Subplot 1: Average Consensus Modularity
fig1, ax1 = plt.subplots(figsize=(10, 5)) 
ax1.errorbar(gamma_range, avg_consensus_modularities, yerr=std_consensus_modularities, fmt='-o')
ax1.set_xlabel('Gamma')
ax1.set_ylabel('Average Consensus Modularity')
ax1.set_title('Average Consensus Modularity vs Gamma')
ax1.grid(True)
fig1.savefig(f'{plot_filename_prefix}_subplot_1.jpeg', bbox_inches='tight')
plt.close(fig1) # Close figure to free memory

# Subplot 2: Average NMI
fig2, ax2 = plt.subplots(figsize=(10, 5)) 
ax2.plot(gamma_range, avg_nmi, '-o')
ax2.set_xlabel('Gamma')
ax2.set_ylabel('Average NMI')
ax2.set_title('Average NMI vs Gamma')
ax2.grid(True)
fig2.savefig(f'{plot_filename_prefix}_subplot_2.jpeg', bbox_inches='tight')
plt.close(fig2)

# Subplot 3: Average ARI
fig3, ax3 = plt.subplots(figsize=(10, 5)) 
ax3.plot(gamma_range, avg_ari, '-o')
ax3.set_xlabel('Gamma')
ax3.set_ylabel('Average ARI')
ax3.set_title('Average ARI vs Gamma')
ax3.grid(True)
fig3.savefig(f'{plot_filename_prefix}_subplot_3.jpeg', bbox_inches='tight')
plt.close(fig3)

# Subplot 4: Average AMI
fig4, ax4 = plt.subplots(figsize=(10, 5)) 
ax4.plot(gamma_range, avg_ami, '-o')
ax4.set_xlabel('Gamma')
ax4.set_ylabel('Average AMI')
ax4.set_title('Average AMI vs Gamma')
ax4.grid(True)
fig4.savefig(f'{plot_filename_prefix}_subplot_4.jpeg', bbox_inches='tight')
plt.close(fig4)

# Subplot 5: Final Agreement Matrix Modularity
fig5, ax5 = plt.subplots(figsize=(10, 5)) 
ax5.plot(gamma_range, final_agreement_matrix_modularities, '-o', label='Final Agreement Matrix Modularity') 
ax5.set_xlabel('Gamma')
ax5.set_ylabel('Modularity')
ax5.set_title('Final Agreement Matrix Modularity vs Gamma') 
ax5.grid(True)
ax5.legend()
fig5.savefig(f'{plot_filename_prefix}_subplot_5.jpeg', bbox_inches='tight')
plt.close(fig5)


print(f"Plots saved to {output_dir}")

# Save metrics to CSV (rest of the code remains the same)
metrics_df = pd.DataFrame(all_metrics)
metrics_file = os.path.join(output_dir, 'clustering_metrics_gamma_scan.csv')
metrics_df.to_csv(metrics_file, index=False)
print(f"Metrics saved to {metrics_file}")

plt.show() # Keep plt.show() to display the last individual plot 
        
# # First plot: Subject consensus
# fig1, ax1 = plt.subplots(figsize=(6.4, 2))
# im1 = ax1.imshow(all_subject_consensus, cmap='Set1', aspect='auto')
# ax1.set(ylabel='Subjects', xlabel='ROIs', xticklabels=[], yticklabels=[])

# # Save first plot
# output_path1 = os.path.join(output_dir, 'subject_consensus.png')
# plt.savefig(output_path1, dpi=600, bbox_inches='tight')

# # Second plot: Final consensus
# fig2, ax2 = plt.subplots(figsize=(6.4, 2))
# im2 = plot_mod_heatmap(conn_matrix, final_consensus, cmap='viridis', ax=ax2)

# # Save second plot
# output_path2 = os.path.join(output_dir, 'final_consensus.png')
# plt.savefig(output_path2, dpi=600, bbox_inches='tight')
