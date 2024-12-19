from pabutools.election import parse_pabulib
from pabutools.rules import method_of_equal_shares
from pabutools.election import Cardinality_Sat
import pandas as pd
import sys
import os
from pathlib import Path

def run_mes_once(pabulib_file: str) -> pd.DataFrame:
    """
    Run the Method of Equal Shares (MES) once and return detailed results.
    
    Args:
        pabulib_file (str): Path to Pabulib file
        
    Returns:
        pd.DataFrame: Results dataframe containing the allocation details
    """
    # Parse the instance and profile
    instance, profile = parse_pabulib(pabulib_file)
    
    # Ensure budget is an integer
    instance.budget_limit = int(instance.budget_limit)
    
    # Run MES
    result = method_of_equal_shares(
        instance=instance,
        profile=profile,
        sat_class=Cardinality_Sat,
        analytics=True
    )
    
    # Calculate efficiency metrics
    total_cost = sum(p.cost for p in result)
    efficiency = total_cost / instance.budget_limit if instance.budget_limit > 0 else None
    
    # No budget increases since we are not using exhaustion
    budget_increase_count = 0
    max_increase = 0
    min_increase = 0
    avg_increase = 0
    
    # Create results DataFrame
    data = {
        'selected_projects': [list(result)],
        'efficiency': [efficiency],
        'budget_increase_count': [budget_increase_count],
        'max_budget_increase': [max_increase],
        'min_budget_increase': [min_increase],
        'avg_budget_increase': [avg_increase],
    }
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_mes_once.py <file_path>")
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
    # /data/coml-humanchess/univ5678/results/waterflow_equal_shares/approval/mes_waterflow_results_approval
    results_dir = base_dir / "results/waterflow_equal_shares/approval/mes_waterflow_no_completion_results_approval"
    

    try:
        # Create results directory with parents and proper permissions
        results_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(results_dir, 0o755)
        
        print(f"Processing file: {input_path}")
        print(f"Results will be saved to: {results_dir / f'{input_path.stem}.csv'}")
        
        output_df = run_mes_once(str(input_path))
        
        # Save results with error handling
        try:
            output_path = results_dir / f"{input_path.stem}.csv"
            output_df.to_csv(output_path, index=False)
            os.chmod(output_path, 0o644)
            print(f"Results successfully saved to: {output_path}")
        except Exception as e:
            print(f"Error saving results to {output_path}: {str(e)}")
            # Try alternative location if original fails
            fallback_path = Path.home() / "mes_results" / f"{input_path.stem}.csv"
            fallback_path.parent.mkdir(parents=True, exist_ok=True)
            output_df.to_csv(fallback_path, index=False)
            print(f"Results saved to fallback location: {fallback_path}")
            
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        sys.exit(1)
