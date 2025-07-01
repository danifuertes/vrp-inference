from utils import evaluate


def main(model_path, dataset_path, problem, temperature=1.0, prize_type='const', use_cuda=True):
    
    # Collect results
    results = evaluate(model_path, dataset_path, num_nodes, temperature, prize_type, use_cuda)
    
    # Print results
    print(f"\nProblem: {problem.upper()}")
    print(f"Cost: {results[0]:.4f}")
    print(f"Sequence: {results[1]}")
    print(f"Time: {results[2]:.4f} seconds")
    
if __name__ == "__main__":
    
    # Parameters
    problem = 'tsp'         # Options: 'tsp', 'op', 'cvrp'
    num_nodes = 20          # Options: 20, 50, 100
    temperature = 1.0
    use_cuda = True
    prize_type = 'const'    # Options (only valid for OP): 'const', 'unif', 'dist'
    
    # Paths
    if problem == 'op':
        model_path = f"pretrained/{problem}_{prize_type}_{num_nodes}"
    else:
        model_path = f"pretrained/{problem}_{num_nodes}"
    dataset_path = f"samples/{problem}_{num_nodes}.json"
    
    # Run model
    main(model_path, dataset_path, problem, temperature, prize_type, use_cuda)
    
    # TODO:
    # - Add plots
    # - Add json generator
    
