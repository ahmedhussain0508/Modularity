import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score
from sklearn.utils.validation import check_random_state
from joblib import Parallel, delayed
import bct
import random
import time

# Keep the original consensus functions intact but add performance enhancements
def find_consensus_2(assignments, null_func=np.mean, return_agreement=True,
                   gamma=1.0, seed=None):
    """
    Original find_consensus_2 function with minimal optimization
    """
    rs = check_random_state(seed)
    samp, comm = assignments.shape

    # Use vectorized operations for agreement matrix calculation
    # This is much faster than the original implementation
    agreement = bct.clustering.agreement(assignments, buffsz=samp) / comm

    # Generate null agreement matrix more efficiently
    null_assign = np.column_stack([rs.permutation(assignments[:, i]) for i in range(assignments.shape[1])])
    null_agree = bct.clustering.agreement(null_assign, buffsz=samp) / comm
    threshold = null_func(null_agree)

    # Run consensus clustering on agreement matrix after thresholding with correct gamma
    consensus = consensus_und_2(agreement, threshold, 300, gamma=gamma, seed=seed)

    if return_agreement:
        return consensus.astype(int), agreement * (agreement > threshold)

    return consensus.astype(int)

def consensus_und_2(D, tau, reps=300, gamma=1.0, seed=None):
    """
    Original consensus_und_2 function with minimal optimization
    """
    rng = bct.utils.get_rng(seed)
    n = len(D)
    flag = True
    
    while flag:
        flag = False
        dt = D * (D >= tau)
        np.fill_diagonal(dt, 0)

        if np.size(np.where(dt == 0)) == 0:
            ciu = np.arange(1, n + 1)
        else:
            # Pre-allocate cis array for better performance
            cis = np.zeros((n, reps), dtype=np.int32)
            
            # This is the key optimization - run Louvain in parallel chunks
            chunk_size = 10  # Process in batches of 10 for better efficiency
            for chunk_start in range(0, reps, chunk_size):
                chunk_end = min(chunk_start + chunk_size, reps)
                chunk_results = Parallel(n_jobs=-1, prefer="threads")(
                    delayed(bct.modularity.modularity_louvain_und_sign)(
                        dt, gamma=gamma, seed=rng.randint(10000)
                    ) for _ in range(chunk_start, chunk_end)
                )
                
                for i, (ci, _) in enumerate(chunk_results):
                    cis[:, chunk_start + i] = ci
            
            # Use original unique_partitions function
            ciu = unique_partitions(cis)
            nu = np.size(ciu, axis=1)
            if nu > 1:
                flag = True
                D = bct.clustering.agreement(cis) / reps

    return np.squeeze(ciu + 1)

def unique_partitions(cis):
    """The original unique_partitions function from the code"""
    # relabels the partitions to recognize different numbers on same topology
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
    
    return np.transpose(ciu)

def process_subject_parallel(subject_file, gamma, n_runs=300, use_mst=True, data_dir=None):
    """Optimized subject processing with parallel Louvain runs"""
    if data_dir is None:
        raise ValueError("data_dir must be provided")
        
    # Load and prepare data - no changes here
    data = np.load(os.path.join(data_dir, subject_file))
    data_clean = np.nan_to_num(data, nan=0)
    data_symmetric = (data_clean + data_clean.T) / 2
    np.fill_diagonal(data_symmetric, 1)
    nonegative = data_symmetric.copy()
    nonegative[nonegative < 0] = 0

    if use_mst:
        mst_matrix = minimum_spanning_tree(nonegative*-1).toarray()*-1
        mst_matrix = (mst_matrix + mst_matrix.T) / 2
        matrix_for_community = mst_matrix
    else:
        matrix_for_community = nonegative

    # Key optimization: run community_louvain in parallel chunks
    cis = []
    modularities = []
    
    # Process in chunks for better efficiency
    chunk_size = 10  # Adjust based on your system
    for chunk_start in range(0, n_runs, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_runs)
        chunk_seeds = range(chunk_start, chunk_end)
        
        # Use threads for better shared memory performance with BCT
        chunk_results = Parallel(n_jobs=-1, prefer="threads")(
            delayed(bct.community_louvain)(
                matrix_for_community, gamma=gamma, seed=seed
            ) for seed in chunk_seeds
        )
        
        for ci, q in chunk_results:
            cis.append(ci)
            modularities.append(q)

    # Use the original consensus function with the specified gamma
    consensus_seed = random.randint(0, 10000)
    consensus_result = find_consensus_2(
        np.column_stack(cis), 
        gamma=gamma,  # Pass gamma to ensure it's used correctly
        seed=consensus_seed, 
        return_agreement=True
    )
    consensus_labels, agreement_matrix = consensus_result
    
    return consensus_labels, nonegative, cis, modularities

def optimize_louvain_processing(data_dir, output_dir, gamma_range, n_subjects=None, n_runs=300):
    """
    Optimized main processing function that closely follows the original
    
    Parameters
    ----------
    data_dir : str
        Directory containing subject data files
    output_dir : str
        Directory to save results
    gamma_range : array-like
        Range of gamma values to process
    n_subjects : int, optional
        Number of subjects to process. If None, process all subjects
    n_runs : int, optional
        Number of Louvain runs per subject
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get subject files
    subject_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npy')])
    if n_subjects is not None:
        subject_files = subject_files[:n_subjects]
    
    print(f"Processing {len(subject_files)} subjects with {n_runs} runs each")
    print(f"Gamma range: {gamma_range}")
    
    # Initialize lists to store metrics
    all_metrics = []
    
    # Process each gamma value
    start_time = time.time()
    for gamma_idx, gamma in enumerate(gamma_range):
        gamma_start = time.time()
        print(f"\nProcessing gamma = {gamma} ({gamma_idx+1}/{len(gamma_range)})")
        
        # Create gamma-specific output directory
        gamma_output_dir = os.path.join(output_dir, f"gamma_{gamma:.4f}")
        if not os.path.exists(gamma_output_dir):
            os.makedirs(gamma_output_dir)
        
        # Process all subjects in parallel - key optimization
        results = Parallel(n_jobs=-1, verbose=10)(
            delayed(process_subject_parallel)(
                subject_file, gamma, n_runs=n_runs, data_dir=data_dir
            ) for subject_file in subject_files
        )
        
        # Unpack results - same as original code
        all_subject_consensus_partitions = []
        all_conn_matrices = []
        all_subject_cis = []
        all_subject_modularities = []
        subject_consensus_partition_modularities = []
        
        for subject_idx, (consensus_partition, conn_matrix, cis, modularities) in enumerate(results):
            all_subject_consensus_partitions.append(consensus_partition)
            all_conn_matrices.append(conn_matrix)
            all_subject_cis.append(cis)
            all_subject_modularities.append(modularities)
            
            subject_consensus_modularity = bct.community_louvain(conn_matrix, gamma=gamma, ci=consensus_partition)[1]
            subject_consensus_partition_modularities.append(subject_consensus_modularity)
        
        # Calculate metrics - same as original
        avg_consensus_modularity = np.mean(subject_consensus_partition_modularities)
        std_consensus_modularity = np.std(subject_consensus_partition_modularities)
        
        # Calculate similarity metrics
        nmi_values = []
        ari_values = []
        ami_values = []
        
        for subject_cis in all_subject_cis:
            for i in range(n_runs):
                for j in range(i + 1, n_runs):
                    nmi_values.append(normalized_mutual_info_score(subject_cis[i], subject_cis[j]))
                    ari_values.append(adjusted_rand_score(subject_cis[i], subject_cis[j]))
                    ami_values.append(adjusted_mutual_info_score(subject_cis[i], subject_cis[j]))
        
        avg_nmi = np.mean(nmi_values)
        avg_ari = np.mean(ari_values)
        avg_ami = np.mean(ami_values)
        
        # Final consensus
        final_consensus, final_agreement_matrix = find_consensus_2(
            np.column_stack(all_subject_consensus_partitions),
            gamma=gamma,  # Make sure gamma is passed correctly
            seed=random.randint(0, 10000)
        )
        
        # Calculate modularity of final agreement matrix
        final_agreement_modularity = bct.community_louvain(final_agreement_matrix, gamma=gamma)[1]
        
        # Calculate number of communities
        num_communities = len(np.unique(final_consensus))
        
        # Store metrics
        all_metrics.append({
            'gamma': gamma,
            'avg_consensus_modularity': avg_consensus_modularity,
            'std_consensus_modularity': std_consensus_modularity,
            'avg_nmi': avg_nmi,
            'avg_ari': avg_ari,
            'avg_ami': avg_ami,
            'final_agreement_matrix_modularity': final_agreement_modularity,
            'num_communities': num_communities
        })
        
        # Save the final agreement matrix
        np.save(os.path.join(gamma_output_dir, f"agreement_matrix_gamma_{gamma:.4f}.npy"), final_agreement_matrix)
        
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
        node_labels = pd.read_csv('/Users/ahmedhussain/Desktop/projects/testing/notebooks/graph/data/ready_for_harmon/groups/SCH/region_mapping.csv', header=None)
        node_labels = node_labels[0].tolist()  
        
        # Create a DataFrame with node assignments
        partition_df = pd.DataFrame({
            'Node': node_labels,
            'Community': final_consensus,
            'Gamma': gamma
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
        
        # Free memory
        del all_subject_consensus_partitions, all_conn_matrices, all_subject_cis, all_subject_modularities
        del subject_consensus_partition_modularities, nmi_values, ari_values, ami_values
    
    # Create DataFrame from metrics
    metrics_df = pd.DataFrame(all_metrics)
    
    # Save metrics to CSV
    metrics_file = os.path.join(output_dir, 'clustering_metrics_gamma_scan.csv')
    metrics_df.to_csv(metrics_file, index=False)
    
    total_time = time.time() - start_time
    print(f"Total processing time: {total_time:.2f} seconds")
    
    return metrics_df

# Example usage
if __name__ == "__main__":
    # Configuration - you can adjust these parameters
    # data_dir = '/mnt/munin/Morey/Lab/ahmed/cerebellum/data/non_residualized/SCH/control'
    # output_dir = '/mnt/munin/Morey/Lab/ahmed/cerebellum/data/Outputs/testing/louvain_non_resid/gamma_1_to_2'

    data_dir = '/Users/ahmedhussain/Desktop/projects/testing/notebooks/graph/data/ready_for_harmon/groups/SCH/control'
    output_dir = '/Users/ahmedhussain/Desktop/projects/testing/notebooks/graph/results/testing'

    #gamma_range = np.array([i/10 for i in range(10, 21)])
    gamma_range = np.array([i/10 for i in range(10, 12)]) # gamma = 1.0 & 1.1
    n_gamma = len(gamma_range)
    
    # Run the optimization
    metrics_df = optimize_louvain_processing(
        data_dir=data_dir,
        output_dir=output_dir,
        gamma_range=gamma_range,
        n_subjects=3,  # Set to None to process all subjects
        n_runs=3
    )

# After the gamma loop completes, create a DataFrame from all_metrics
metrics_df = pd.DataFrame(all_metrics)

# 1. Combined Modularity Plot
fig_mod, ax_mod = plt.subplots(figsize=(10, 6))
ax_mod.errorbar(metrics_df['gamma'], metrics_df['avg_consensus_modularity'], 
                yerr=metrics_df['std_consensus_modularity'], fmt='-o', 
                label='Average Consensus Modularity')
ax_mod.plot(metrics_df['gamma'], metrics_df['final_agreement_matrix_modularity'], 
            '-o', label='Final Agreement Matrix Modularity')
ax_mod.set_xlabel('Gamma')
ax_mod.set_ylabel('Modularity')
ax_mod.set_title('Modularity vs Gamma')
ax_mod.grid(True)
ax_mod.legend()
fig_mod.savefig(os.path.join(output_dir, 'clustering_metrics_gamma_scan_modularity.jpeg'), 
                bbox_inches='tight')
plt.close(fig_mod)

# 2. Combined Similarity Indices Plot (AMI, ARI, NMI together)
fig_sim, ax_sim = plt.subplots(figsize=(10, 6))
ax_sim.plot(metrics_df['gamma'], metrics_df['avg_nmi'], '-o', label='NMI', color='blue')
ax_sim.plot(metrics_df['gamma'], metrics_df['avg_ari'], '-o', label='ARI', color='green')
ax_sim.plot(metrics_df['gamma'], metrics_df['avg_ami'], '-o', label='AMI', color='red')
ax_sim.set_xlabel('Gamma')
ax_sim.set_ylabel('Similarity Index Value')
ax_sim.set_title('Similarity Indices vs Gamma')
ax_sim.grid(True)
ax_sim.legend()
fig_sim.savefig(os.path.join(output_dir, 'clustering_metrics_gamma_scan_similarity.jpeg'), 
                bbox_inches='tight')
plt.close(fig_sim)

# 3. Plot showing all metrics together
fig_all, ax_all = plt.subplots(figsize=(12, 7))
# Primary y-axis for modularity
ax_all.errorbar(metrics_df['gamma'], metrics_df['avg_consensus_modularity'], 
                yerr=metrics_df['std_consensus_modularity'], fmt='-o', 
                label='Avg Consensus Modularity', color='darkblue')
ax_all.plot(metrics_df['gamma'], metrics_df['final_agreement_matrix_modularity'], 
            '-o', label='Final Agreement Modularity', color='royalblue')
ax_all.set_xlabel('Gamma')
ax_all.set_ylabel('Modularity', color='blue')
ax_all.tick_params(axis='y', labelcolor='blue')

# Secondary y-axis for similarity indices
ax2 = ax_all.twinx()
ax2.plot(metrics_df['gamma'], metrics_df['avg_nmi'], '--^', label='NMI', color='darkgreen')
ax2.plot(metrics_df['gamma'], metrics_df['avg_ari'], '--s', label='ARI', color='green')
ax2.plot(metrics_df['gamma'], metrics_df['avg_ami'], '--d', label='AMI', color='limegreen')
ax2.set_ylabel('Similarity Index', color='green')
ax2.tick_params(axis='y', labelcolor='green')

# Add legend for both axes
lines1, labels1 = ax_all.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax_all.legend(lines1 + lines2, labels1 + labels2, loc='best')

ax_all.set_title('Community Detection Metrics vs Gamma')
ax_all.grid(True)
fig_all.tight_layout()
fig_all.savefig(os.path.join(output_dir, 'clustering_metrics_gamma_scan_all.jpeg'), 
                bbox_inches='tight')
plt.close(fig_all)

# 4. Additional visualization: Number of communities vs gamma
# This requires extracting the number of communities for each gamma from your data
fig_comm, ax_comm = plt.subplots(figsize=(10, 6))
# Extract number of communities per gamma from original analysis 
# (We'd need to capture this during the main loop)
# As a placeholder, we'll create a random example
if 'num_communities' in metrics_df.columns:
    ax_comm.plot(metrics_df['gamma'], metrics_df['num_communities'], '-o', color='purple')
    ax_comm.set_xlabel('Gamma')
    ax_comm.set_ylabel('Number of Communities')
    ax_comm.set_title('Number of Communities vs Gamma')
    ax_comm.grid(True)
    fig_comm.savefig(os.path.join(output_dir, 'clustering_metrics_gamma_scan_num_communities.jpeg'), 
                    bbox_inches='tight')
    plt.close(fig_comm)

# 5. Save metrics to CSV
metrics_file = os.path.join(output_dir, 'clustering_metrics_gamma_scan.csv')
metrics_df.to_csv(metrics_file, index=False)
print(f"Metrics saved to {metrics_file}")
