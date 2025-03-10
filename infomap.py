import os
import numpy as np
from sklearn.metrics import normalized_mutual_info_score
from scipy.stats import ttest_ind
import bct
from igraph import Graph, ADJ_UNDIRECTED
from itertools import combinations
import json
from scipy.sparse.csgraph import minimum_spanning_tree


# Directories
control_dir = "/mnt/munin/Morey/Lab/ahmed/cerebellum/data/non_residualized/SCH/control"
ptsd_dir = "/mnt/munin/Morey/Lab/ahmed/cerebellum/data/non_residualized/SCH/ptsd"
output_dir = "/mnt/munin/Morey/Lab/ahmed/cerebellum/data/Outputs/testing/infomap/2_18_2025"

def matrix_to_igraph(matrix,cost,binary=False,check_tri=True,method='linear',normalize=False,mst=False,test_matrix=True):
	"""
	Convert a matrix to an igraph object
	matrix: a numpy matrix
	cost: the proportion of edges. e.g., a cost of 0.1 has 10 percent
	of all possible edges in the graph
	binary: False, convert weighted values to 1
	check_tri: True, ensure that the matrix contains upper and low triangles.
	if it does not, the cost calculation changes.
	interpolation: midpoint, the interpolation method to pass to np.percentile
	normalize: False, make all edges sum to 1. Convienient for comparisons across subjects,
	as this ensures the same sum of weights and number of edges are equal across subjects
	mst: False, calculate the maximum spanning tree, which is the strongest set of edges that
	keep the graph connected. This is convienient for ensuring no nodes become disconnected.
	"""
	matrix = np.array(matrix)
	matrix = threshold(matrix,cost,binary,check_tri,method,normalize,mst)
	g = Graph.Weighted_Adjacency(matrix.tolist(),mode=ADJ_UNDIRECTED,attr="weight")
	print('Matrix converted to graph with density of:' + str(g.density()))
	if abs(np.diff([cost,g.density()])[0]) > .005:
		print('Density not %s! Did you want: ' %(cost)+ str(g.density()) + ' ?')
	return g


def threshold(matrix, cost, binary=False, check_tri=True, method='linear', normalize=False, mst=True, test_matrix=True):
    """
    Threshold a numpy matrix to obtain a certain "cost".
    matrix: a numpy matrix
    cost: the proportion of edges. e.g., a cost of 0.1 has 10 percent of all possible edges in the graph
    binary: False, convert weighted values to 1
    check_tri: True, ensure that the matrix contains upper and low triangles.
    if it does not, the cost calculation changes.
    interpolation: midpoint, the interpolation method to pass to np.percentile
    normalize: False, make all edges sum to 1. Convenient for comparisons across subjects,
    as this ensures the same sum of weights and number of edges are equal across subjects
    mst: False, calculate the maximum spanning tree, which is the strongest set of edges that
    keep the graph connected. This is convenient for ensuring no nodes become disconnected.
    """
    matrix = np.array(matrix)
    matrix[np.isnan(matrix)] = 0.0

    # Calculate the number of edges to retain based on density
    n = matrix.shape[0]
    max_edges = (n * (n - 1)) / 2  # Maximum possible edges in an undirected graph
    if check_tri and (np.sum(np.triu(matrix)) == 0.0 or np.sum(np.tril(matrix)) == 0.0):
        cost = cost / 2.0  # Adjust cost for triangular matrices
    num_edges = int(cost * max_edges)

    if num_edges > 0:
        if mst:
            if test_matrix:
                t_m = matrix.copy()  # Copy for assertion check
            # Ensure matrix is symmetric for MST computation
            assert (np.tril(matrix, -1) == np.triu(matrix, 1).transpose()).all()
            # Work on a copy to avoid modifying the original matrix
            matrix_lower = np.tril(matrix.copy(), -1)
            mst = minimum_spanning_tree(matrix_lower * -1) * -1
            mst = mst.toarray()
            mst = mst.transpose() + mst
            # Restore original matrix for assertion check
            if test_matrix:
                assert (matrix == t_m).all(), "Matrix was modified unexpectedly during MST computation"
        
        # Sort edges by weight and select top N edges
        triu_indices = np.triu_indices(n, k=1)
        edge_weights = matrix[triu_indices]
        edge_indices = np.argsort(edge_weights)[::-1]  # Sort in descending order
        selected_edges = edge_indices[:num_edges]  # Select top N edges

        # Create thresholded matrix
        thresholded_matrix = np.zeros_like(matrix)
        if mst:
            thresholded_matrix += mst  # Include MST edges
        for idx in selected_edges:
            i, j = triu_indices[0][idx], triu_indices[1][idx]
            thresholded_matrix[i, j] = matrix[i, j]
            thresholded_matrix[j, i] = matrix[j, i]  # Symmetric
    else:
        thresholded_matrix = np.zeros_like(matrix)

    if binary:
        thresholded_matrix[thresholded_matrix > 0] = 1
    if normalize and np.sum(thresholded_matrix) > 0:
        thresholded_matrix = thresholded_matrix / np.sum(thresholded_matrix)

    return thresholded_matrix


def ave_consensus_costs_parition(matrix, min_cost, max_cost):
	'''Run a partition for every cost threshold using infomap, turn parition into identiy matrix, average
	identiy matrix across costs to generate consensus matrix, run infomap on consens matrix to obtain final
partition'''

	consensus_matricies = np.zeros((len(np.arange(min_cost, max_cost+0.01, 0.01)), matrix.shape[0], matrix.shape[1]))

	for i, cost in enumerate(np.arange(min_cost, max_cost+0.01, 0.01)):

		graph = matrix_to_igraph(matrix.copy(),cost=cost)
		infomap_paritition = graph.community_infomap(edge_weights='weight')
		consensus_matricies[i,:,:] = community_matrix(infomap_paritition.membership)

	ave_consensus = np.mean(consensus_matricies, axis=0)
	graph = matrix_to_igraph(ave_consensus,cost=1.)
	final_infomap_partition = graph.community_infomap(edge_weights='weight')

	return final_infomap_partition.membership

def community_matrix(membership):
	'''To generate a identiy matrix where nodes that belong to the same community/patition has
	edges set as "1" between them, otherwise 0 '''

	membership = np.array(membership).reshape(-1)

	final_matrix = np.zeros((len(membership),len(membership)))
	final_matrix[:] = np.nan
	connected_nodes = []
	for i in np.unique(membership):
		for n in np.array(np.where(membership==i))[0]:
			connected_nodes.append(int(n))

	within_community_edges = []
	between_community_edges = []
	connected_nodes = np.array(connected_nodes)
	for edge in combinations(connected_nodes,2):
		if membership[edge[0]] == membership[edge[1]]:
			within_community_edges.append(edge)
		else:
			between_community_edges.append(edge)

	# set edge as 1 if same community
	for edge in within_community_edges:
		final_matrix[edge[0],edge[1]] = 1
		final_matrix[edge[1],edge[0]] = 1
	for edge in between_community_edges:
		final_matrix[edge[0],edge[1]] = 0
		final_matrix[edge[1],edge[0]] = 0

	return final_matrix

def power_recursive_partition(matrix, min_cost, max_cost, min_community_size=5):
	''' this is the interpretation of what Power did in his 2011 Neuron paper, start with a high cost treshold, get infomap parition, then step down, but keep the
	parition that did not change across thresholds'''

	final_edge_matrix = matrix.copy()
	final_identity_matrix = np.zeros(matrix.shape)

	cost = max_cost

	while True:
		graph = matrix_to_igraph(matrix.copy(),cost=cost)
		partition = graph.community_infomap(edge_weights='weight')
		connected_nodes = []

		for node in range(partition.graph.vcount()):
			if partition.sizes()[partition.membership[node]] > min_community_size:
				connected_nodes.append(node)

		within_community_edges = []
		between_community_edges = []
		for edge in combinations(connected_nodes,2):
			if partition.membership[edge[0]] == partition.membership[edge[1]]:
				within_community_edges.append(edge)
			else:
				between_community_edges.append(edge)
		for edge in within_community_edges:
			final_identity_matrix[edge[0],edge[1]] = 1
			final_identity_matrix[edge[1],edge[0]] = 1
		for edge in between_community_edges:
			final_identity_matrix[edge[0],edge[1]] = 0
			final_identity_matrix[edge[1],edge[0]] = 0
		if cost < min_cost:
			break
		if cost <= .05:
			cost = cost - 0.001
			continue
		if cost <= .15:
			cost = cost - 0.01
			continue

	graph = matrix_to_igraph(final_identity_matrix,cost=1.)
	final_infomap_partition = np.array(graph.community_infomap(edge_weights='weight').membership)
	return final_infomap_partition

def load_matrices(directory, max_files=None):
    matrices = []
    filenames = [] # To store filenames
    npy_files = [f for f in os.listdir(directory) if f.endswith(".npy")]
    
    # Limit the number of files if max_files is specified
    if max_files is not None:
        npy_files = npy_files[:max_files]
        
    for file in npy_files:
        matrix = np.load(os.path.join(directory, file))
        matrix[np.isnan(matrix)] = 0.0  # Replace NaNs with 0
        matrix[matrix < 0] = 0.0       # Set negative values to 0
        matrices.append(matrix)
        filenames.append(file[:-4]) # Store filename without .npy extension
    return matrices, filenames # Return both matrices and filenames


# Save Results
def save_results(output_dir, filename, data):
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    if isinstance(data, np.ndarray):
        np.save(filepath, data)
    elif isinstance(data, dict) or isinstance(data, list):
        with open(filepath, 'w') as f:
            json.dump(data, f)
    else:
        with open(filepath, 'w') as f:
            f.write(str(data))

def test_pipline(control_dir, base_output_dir="/mnt/munin/Morey/Lab/ahmed/cerebellum/data/Outputs/testing/infomap/2_18_2025", max_files=None): 
    """A run through test using a directory of matrices as input, processes each subject individually
    Args:
        control_dir (str): Directory containing the input matrices
        base_output_dir (str): Directory to save output results
        max_files (int, optional): Maximum number of files to process. If None, processes all files.
    """
    # load matrices from directory and get filenames
    matrices, filenames = load_matrices(control_dir, max_files)
    if not matrices:
        print(f"No matrices found in directory: {control_dir}")
        return

    print(f"Found {len(matrices)} matrices to process")
    
    max_cost = .15
    min_cost = .01

    for i, matrix in enumerate(matrices): 
        filename = filenames[i]
        print(f"\nProcessing subject: {filename}")
        print(f"Matrix shape: {matrix.shape}")

        # ave consensus across costs
        partition_ave_consensus = ave_consensus_costs_parition(matrix, min_cost, max_cost)
        partition_ave_consensus = np.array(partition_ave_consensus) + 1
        print(f"Partition shape: {partition_ave_consensus.shape}")

        # Calculate number of cost steps
        cost_steps = len(np.arange(min_cost, max_cost+0.001, 0.001))
        print(f"Number of cost steps: {cost_steps}")

        # import thresholded matrix to BCT, import partition, run WMD/PC
        PCs = np.zeros((cost_steps, matrix.shape[0]))
        WMDs = np.zeros((cost_steps, matrix.shape[0]))

        for j, cost in enumerate(np.arange(min_cost, max_cost+0.001, 0.001)):
            tmp_matrix = threshold(matrix.copy(), cost)
            #PC
            PCs[j,:] = bct.participation_coef(tmp_matrix, partition_ave_consensus)
            #WMD
            WMDs[j,:] = bct.module_degree_zscore(matrix, partition_ave_consensus)

        print(f"PC matrix shape: {PCs.shape}")
        print(f"WMD matrix shape: {WMDs.shape}")

        # Save results for average consensus partition
        ave_consensus_dir = os.path.join(base_output_dir, "results_ave_consensus")
        print(f"\nSaving average consensus results to: {ave_consensus_dir}")
        
        try:
            save_results(ave_consensus_dir, f"{filename}_partition_ave_consensus.npy", partition_ave_consensus)
            save_results(ave_consensus_dir, f"{filename}_PCs_ave_consensus.npy", PCs)
            save_results(ave_consensus_dir, f"{filename}_WMDs_ave_consensus.npy", WMDs)
            print("Successfully saved average consensus results")
        except Exception as e:
            print(f"Error saving average consensus results: {str(e)}")

        # alternatively, merge consensus using the power method
        recursive_partition = power_recursive_partition(matrix, min_cost, max_cost)
        recursive_partition = recursive_partition + 1
        print(f"Recursive partition shape: {recursive_partition.shape}")

        # Save results for recursive partition
        recursive_dir = os.path.join(base_output_dir, "results_recursive_partition")
        print(f"\nSaving recursive partition results to: {recursive_dir}")
        
        try:
            save_results(recursive_dir, f"{filename}_recursive_partition.npy", recursive_partition)
            print("Successfully saved recursive partition results")
        except Exception as e:
            print(f"Error saving recursive partition results: {str(e)}")

    print("\nProcessing complete for all subjects.")


if __name__ == '__main__':
    control_dir = "/mnt/munin/Morey/Lab/ahmed/cerebellum/data/non_residualized/SCH/control"
    output_dir = "/mnt/munin/Morey/Lab/ahmed/cerebellum/data/Outputs/testing/infomap/2_18_2025"
    
    # Process only the first 10 files
    test_pipline(control_dir, output_dir, max_files=10)


# control_matrices = load_matrices(control_dir)
# ptsd_matrices = load_matrices(ptsd_dir)

# # Parameters
# min_cost = 0.01
# max_cost = 0.15

# # Apply Partitioning Methods
# control_partitions_consensus = [ave_consensus_costs_parition(matrix, min_cost, max_cost) for matrix in control_matrices]
# ptsd_partitions_consensus = [ave_consensus_costs_parition(matrix, min_cost, max_cost) for matrix in ptsd_matrices]

# # Save Partitions
# save_results(output_dir, "control_partitions_consensus.npy", np.array(control_partitions_consensus))
# save_results(output_dir, "ptsd_partitions_consensus.npy", np.array(ptsd_partitions_consensus))


# # For Consensus Method
# consensus_observed_nmis = []
# for control_part in control_partitions_consensus:
#     for ptsd_part in ptsd_partitions_consensus:
#         nmi = normalized_mutual_info_score(control_part, ptsd_part)
#         consensus_observed_nmis.append(nmi)
# consensus_observed_mean = np.mean(consensus_observed_nmis)

# # Permutation Testing for Consensus Method
# n_permutations = 10000
# all_consensus_partitions = control_partitions_consensus + ptsd_partitions_consensus
# n_control = len(control_partitions_consensus)
# consensus_permuted_means = []

# for _ in range(n_permutations):
#     np.random.shuffle(all_consensus_partitions)
#     perm_group1 = all_consensus_partitions[:n_control]
#     perm_group2 = all_consensus_partitions[n_control:]
    
#     perm_nmis = []
#     for part1 in perm_group1:
#         for part2 in perm_group2:
#             nmi = normalized_mutual_info_score(part1, part2)
#             perm_nmis.append(nmi)
#     consensus_permuted_means.append(np.mean(perm_nmis))

# consensus_p_value = np.mean(np.array(consensus_permuted_means) >= consensus_observed_mean)

