from pabutools.election import parse_pabulib
from pabutools.rules import method_of_equal_shares, exhaustion_by_budget_increase
from pabutools.election import Instance, ApprovalProfile, Cost_Sat
import pandas as pd
import os
from pathlib import Path
import sys

def run_mes_with_exhaustion(pabulib_file: str, budget: int = 0) -> pd.DataFrame:
    """
    Run the Method of Equal Shares with budget exhaustion.
    
    Args:
        pabulib_file (str): Path to Pabulib file
        budget (int): Optional custom budget (defaults to instance budget if 0)
        
    Returns:
        pd.DataFrame: Results dataframe containing the allocation details
    """
    # Parse the instance and profile
    instance, profile = parse_pabulib(pabulib_file)
    
    # Set custom budget if provided
    if budget > 0:
        initial_budget = int(budget)
        instance.budget_limit = int(budget)
    else:
        initial_budget = int(instance.budget_limit)

    # Ensure budget is an integer
    instance.budget_limit = int(instance.budget_limit)

    # Run MES with budget exhaustion
    # stopper = True
    increase_counter = 0
    best_efficiency = 0
    non_mono_flag = 0
    non_mono_is_true = 0
    print("result")
    while True:
        result = method_of_equal_shares(
            instance=instance,
            profile=profile,
            sat_class = Cost_Sat,
        )
        
        total_cost = sum(p.cost for p in result)
        if len(result) == len(instance): #all projects selected
            break
        efficiency = total_cost / initial_budget
        if efficiency < 1 and efficiency > best_efficiency:
            best_efficiency = efficiency
            if non_mono_flag:
                non_mono_is_true = 1
        if efficiency > 1:
            non_mono_flag = 1
        increase_counter+=1
        print(f"Increase counter {increase_counter}")
        print(f"efficiency {efficiency}")
        instance.budget_limit = instance.budget_limit+1

    #     # Calculate efficiency metrics
    # total_cost = sum(p.cost for p in result)
    # efficiency = total_cost / initial_budget

    # # Get budget increases from the details
    # if hasattr(result, "details") and result.details is not None:
    #     budget_increases = [
    #         int((instance.budget_limit - initial_budget) / len(profile))
    #         for iteration in result.details.iterations
    #     ]
    #     print("ResuLTS")
    #     print(result.details.iterations)
    #     budget_increase_count = len(budget_increases)
    #     max_increase = max(budget_increases) if budget_increases else 0
    #     min_increase = min(budget_increases) if budget_increases else 0
    #     avg_increase = sum(budget_increases) / len(budget_increases) if budget_increases else 0
    # else:
    #     budget_increase_count = increase_counter
    max_increase = 0
    min_increase = 0
    avg_increase = 0
    budget_increase_count = increase_counter
    # Create results DataFrame
    data = {
        'selected_projects': [list(result)],
        'efficiency': [efficiency],
        'budget_increase_count': [budget_increase_count],
        'max_budget_increase': [max_increase],
        'min_budget_increase': [min_increase],
        'avg_budget_increase': [avg_increase],
        'non_mono': [non_mono_is_true],
    }
    
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_mes_exhaustion.py <file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        sys.exit(1)
    
    # Get absolute paths for input and output
    input_path = Path(file_path).resolve()
    
    # Create results directory
    if 'DATA' in os.environ:
        base_dir = Path(os.environ['DATA'])
    else:
        base_dir = input_path.parent
        
    # /data/coml-humanchess/univ5678/results/waterflow_equal_shares/cost/mes_waterflow_exhaustive_results_cost
    results_dir = base_dir / "results/waterflow_equal_shares/cost/mes_waterflow_non_exhaustive_results_cost"
    
    try:
        # Create results directory with parents and proper permissions
        results_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(results_dir, 0o755)
        
        print(f"Processing file: {input_path}")
        print(f"Results will be saved to: {results_dir / f'{input_path.stem}.csv'}")
        
        output_df = run_mes_with_exhaustion(str(input_path))
        
        # Save results with error handling
        try:
            output_path = results_dir / f"{input_path.stem}.csv"
            output_df.to_csv(output_path, index=False)
            os.chmod(output_path, 0o644)
            print(f"Results successfully saved to: {output_path}")
        except Exception as e:
            print(f"Error saving results to {output_path}: {str(e)}")
            # Try alternative location if original fails
            fallback_path = Path.home() / "slurm_results" / f"{input_path.stem}.csv"
            fallback_path.parent.mkdir(parents=True, exist_ok=True)
            output_df.to_csv(fallback_path, index=False)
            print(f"Results saved to fallback location: {fallback_path}")
            
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        sys.exit(1)