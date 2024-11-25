from pabutools.election import parse_pabulib
from typing import List, Tuple, Dict
from pabutools.rules import method_of_equal_shares as mes
from pabutools.election import (
    Instance,
    Project,
    ApprovalProfile,
    ApprovalBallot,
    Cost_Sat,
    Cardinality_Sat,
)
from pabutools.tiebreaking import app_score_tie_breaking
from collections import defaultdict
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
import multiprocessing
import copy
from pathlib import PureWindowsPath, Path
from collections import OrderedDict
import numpy as np
import os 
import sys

def cost_utility(project):
    return project.cost

def profile_preprocessing(profile):
    # Check if the input is already processed
    if profile and isinstance(profile[0], dict) and "approved" in profile[0] and "name" in profile[0]:
        return profile

    to_return = []
    
    for i, voter in enumerate(profile):
        voter_id = i + 1  # Using 1-based indexing for IDs
        
        to_return.append({
            "approved": list(voter),  # Convert set to list
            "name": voter_id
        })
    
    return to_return

def greedy_project_change_uniform(instance, profile, selected_projects_with_bpb, payments, p, L, utility_function):
    """
    Determine the minimal budget increase necessary for a project to change the outcome
    for uniform utilities.

    :param instance: The election instance containing project metadata.
    :param profile: A list of voter preferences.
    :param selected_projects: List of currently selected projects.
    :param payments: A dictionary mapping (voter, project) pairs to payment amounts.
    :param p: The project being considered for potential inclusion.
    :param L: A pre-computed list of payment thresholds.
    :return: Minimum d > 0 such that p certifies the instability of the current allocation.
    """
    # voters = range(len(profile))
    voter_list = profile
    # print(f"voter_list {voter_list}")
    # projects = instance.project_meta
    d = float("inf")
    ell = 0
    i = len(selected_projects_with_bpb)
    N_p = sum(1 for voter in voter_list if p in voter['approved'])

    # print(f"voter_list {voter_list}")
    # print(f"p {p}")
    # print(f"N_p has value {N_p}")

    # Calculate O_p(X): supporters of p not currently paying for it
    O_p_X = [
        voter['name']
        for voter in voter_list
        if p in voter['approved'] and payments.get((voter['name'], p), 0) == 0
    ]

    # Main loop
    while i > 0 and ell < len(O_p_X):
        remaining_buyers = N_p - ell
        if N_p == 0:
            PvP = float("inf")
        else:
            PvP = p.cost / (N_p - ell)

        #NOTE THIS IS for cost not uniform TODO
        # while i > 0 and (
        #     remaining_buyers < selected_projects_with_bpb[i - 1][1]
        #     or (
        #         remaining_buyers == selected_projects_with_bpb[i - 1][1]
        #         and selected_projects_with_bpb[i - 1][0].name > p.name
        #     )
        # ):
        while i > 0 and ((utility_function(p)/PvP < selected_projects_with_bpb[i-1][1]) or (utility_function(p)/PvP == selected_projects_with_bpb[i-1][1]) and selected_projects_with_bpb[i-1][0].name < p.name):
            i -= 1

        L_val = L[i][ell]
        if L_val[1] > PvP:
            payment = payments.get((L_val[0], selected_projects_with_bpb[i - 1][0]), 0)
            for project, _ in selected_projects_with_bpb:
                voter_payment = payments.get((L_val[0], project), 0)
        d = min(d, PvP - L_val[1])
        ell += 1

    return d

def add_opt_cost(instance, profile, sorted_selected_with_bpb, payments, shares):
    # print("NEW ADD OPT RUN")
    """
    Determine the minimum budget increase needed for the solution to become unstable
    for uniform utilities.

    :param instance: Pabulib instance object
    :param profile: Pabulib profile object
    :param selected_projects: list of currently selected projects ordered by bpb with lex tie breaking
    :param payments: Current payment allocations
    :return: Minimum d > 0 such that (selected_projects, payments) is unstable for instance with increased budget
    """

    sorted_selected_with_bpb = list(sorted_selected_with_bpb.items()) 

    print(f"In add opt uniform, sorted_selected_with_bpb {sorted_selected_with_bpb}")
    voters = range(len(profile))
    voter_set = profile
    #@comment for Isaac - why are you doing this:
    #for i, v in enumerate(voter_set):
    #    v.name = i
    projects = instance.project_meta
    approval_sets = [voter for voter in profile]
    #project_costs = {p: p.cost for p in projects}
    total_budget = instance.budget_limit
    #selected_projects = list(selected_projects)

    d = float("inf")
    # print(voter_set)
    # print(voter_set[0])
    print(f"In add opt uniform, shares {shares}")
    budget_list = [[v,shares[v['name']]] for v in voter_set]

    # Sort the dictionary by value
    budget_list.sort(key=lambda x: x[1])
    L0 = copy.deepcopy(budget_list)

    def bpb(project):
        return sum(
            1
            for voter in voter_set
            if project in voter and payments.get((voter, project), 0) != 0
        )

    L = [copy.deepcopy(L0)]
    L_curr = ""
    # print(f"sorted selected {sorted_selected}")
    for k in range(1, len(sorted_selected_with_bpb) + 1):
        L_curr = copy.deepcopy(L[-1])
        project = sorted_selected_with_bpb[k - 1]
        for i in range(len(L_curr)):
            voter_name = L_curr[i][0]['name']
            L_curr[i][1] += payments.get((voter_name, project), 0)
        L_curr.sort(key=lambda x: x[1])
        L.append(L_curr)

    def get_L_Op(p):
        new_list = []
        for project_list in L:
            new_proj_list = []
            for i, pair in enumerate(project_list):  # pair = [voter, contribution]
                if (
                    p in pair[0]['approved'] 
                    and payments.get((pair[0]['name'], p), 0) == 0
                ):
                    new_proj_list.append(pair)
            new_proj_list.sort(key=lambda x: x[1])
            new_list.append(new_proj_list)

        return new_list

    def convert_to_float(list_of_lists):
        return [[float(item) for item in sublist] for sublist in list_of_lists]

    # Call GreedyProjectChange for each project
    for p in projects:
        print(f"p {p}")
        L_Op = get_L_Op(p)
        # print(f"L_Op {L_Op}")
        GP = greedy_project_change_uniform(
            instance, profile, sorted_selected_with_bpb, payments, p, L_Op
        )
        print(f"GP {GP}")
        if GP < 0:
            print(f"P_negative_GP {p}")

        d = min(d, GP)

    return d


def add_opt_cost_heuristic(instance, profile, sorted_selected_with_bpb, payments, shares):
    """
    Determine the minimum budget increase needed for the solution to become unstable
    for uniform utilities.

    :param instance: Pabulib instance object
    :param profile: Pabulib profile object
    :param selected_projects: list of currently selected projects ordered by bpb with lex tie breaking
    :param payments: Current payment allocations
    :return: Minimum d > 0 such that (selected_projects, payments) is unstable for instance with increased budget
    """

    sorted_selected_with_bpb = list(sorted_selected_with_bpb.items()) 

    print(f"In add opt uniform, sorted_selected_with_bpb {sorted_selected_with_bpb}")
    voters = range(len(profile))
    voter_set = profile
    projects = instance.project_meta
    approval_sets = [voter for voter in profile]
    #project_costs = {p: p.cost for p in projects}
    total_budget = instance.budget_limit
    #selected_projects = list(selected_projects)

    d = float("inf")
    print(f"In add opt uniform, shares {shares}")
    budget_list = [[v,shares[v['name']]] for v in voter_set]

    # Sort the dictionary by value
    budget_list.sort(key=lambda x: x[1])
    L0 = copy.deepcopy(budget_list)

    def bpb(project):
        return sum(
            1
            for voter in voter_set
            if project in voter and payments.get((voter, project), 0) != 0
        )

    L = [copy.deepcopy(L0)]
    L_curr = ""
    for k in range(1, len(sorted_selected_with_bpb) + 1):
        L_curr = copy.deepcopy(L[-1])
        project = sorted_selected_with_bpb[k - 1]
        for i in range(len(L_curr)):
            voter_name = L_curr[i][0]['name']
            L_curr[i][1] += payments.get((voter_name, project), 0)
        L_curr.sort(key=lambda x: x[1])
        L.append(L_curr)

    def get_L_Op(p):
        new_list = []
        for project_list in L:
            new_proj_list = []
            for i, pair in enumerate(project_list):  # pair = [voter, contribution]
                if (
                    p in pair[0]['approved'] 
                    and payments.get((pair[0]['name'], p), 0) == 0
                ):
                    new_proj_list.append(pair)
            new_proj_list.sort(key=lambda x: x[1])
            new_list.append(new_proj_list)

        return new_list

    def convert_to_float(list_of_lists):
        return [[float(item) for item in sublist] for sublist in list_of_lists]

    # Call GreedyProjectChange for each project
    for p in projects:
        if p in sorted_selected_with_bpb.keys():
            continue
        L_Op = get_L_Op(p)
        # print(f"L_Op {L_Op}")
        GP = greedy_project_change_uniform(
            instance, profile, sorted_selected_with_bpb, payments, p, L_Op
        )
        print(f"GP {GP}")
        if GP < 0:
            print(f"P_negative_GP {p}")

        d = min(d, GP)

    return d


def exact_method_of_equal_shares_uniform(instance, profile, utility_function=lambda x: 1, budget=0):
    """
    returns     
    return selected projects with bpb, X_payments, shares, total_cost
    
    """
    
    if budget > 0:
        instance.budget_limit = budget

    budget = instance.budget_limit
    projects = instance.project_meta  # dict with keys: project name and property cost

    # print(f"Projects {projects}")
    # print(f"Instance {instance}")
    num_voters = len(profile)
    X_payments = {(voter['name'], project): 0 for voter in profile for project in projects}

    # Initialize voter shares
    shares = {voter['name']: budget / num_voters for voter in profile}
    funded_projects = OrderedDict()  # remembers the order in which elements were inserted
    total_cost = 0
    project_support = {project:[voter['name'] for voter in profile if project in voter['approved']] for project in projects} # dict of list of supporters of projects for efficiency

    def calculate_bang_per_buck(project, number_paying_voters, utility_function):
        return utility_function(project) * number_paying_voters / project.cost

    def filter_sd(sd, keys_to_keep):
        return [item for item in sd if item[0] in keys_to_keep]

    while True:
        best_project = None
        max_bang_per_buck = 0
        best_index = 0
        best_supp_shares = None  # Keep track of sd corresponding to best_project

        #move sorting outside of project loop
        supp_shares_base = sorted(shares.items(), key=lambda item: item[1])

        for project in projects:
            # print(f"project under consideration {project}")
            if project in funded_projects.keys():
                continue

            if not project_support[project]:
                continue
            # filtered_shares = filter_dict(shares, approving_voters)
            # sd_0 = sorted(filtered_shares.items(), key=lambda item: item[1])
            supp_shares = filter_sd(supp_shares_base, project_support[project])

            number_paying_voters = len(supp_shares)
            for i in range(len(supp_shares)):
                # print(f"project under consideration {project}")
                # print(f" num paying voters {number_paying_voters}")
                max_contribution = project.cost / number_paying_voters  # Equal share contribution
                # print(f"Max contributions for {project} is {max_contribution}")
                # print(f"SD for {project} is {sd}")
                # print(f"SD[i][1] for {project} is {sd[i][1]}")
                if max_contribution <= supp_shares[i][1]:
                    bang_per_buck = calculate_bang_per_buck(
                        project, number_paying_voters, utility_function
                    )
                    # print(f"BPB for {project} is {bang_per_buck}")
                    # print(f"Current max BPB for {project} is {max_bang_per_buck}")
                    if bang_per_buck > max_bang_per_buck:
                        max_bang_per_buck = bang_per_buck
                        best_project = (project, bang_per_buck)  # best project now a tuple with project, bang per buck
                        best_index = i
                        best_supp_shares = supp_shares
                    elif (bang_per_buck == max_bang_per_buck and best_project is not None):
                        # Tie-breaking based on project name
                        # print("TIE BREAKING")
                        if project.name > best_project[0].name:
                            best_project = (project, bang_per_buck)
                            best_index = i
                            best_supp_shares = supp_shares
                    break  # Found a valid group, no need to remove more voters
                number_paying_voters -= 1

        if best_project is None:
            break

        # Fund the project
        contribution = best_project[0].cost / (len(best_supp_shares) - best_index)
        # print(f"BEST PROJECT SELECTED {best_project[0]} contribution: {contribution}")

        # can't we just we just filter for for brest project from scratch rather than storing?
        for voter_name, _ in best_supp_shares[best_index:]:
            shares[voter_name] -= contribution
            X_payments[(voter_name, best_project[0])] = contribution

        funded_projects[best_project[0]] = best_project[1]
        total_cost += best_project[0].cost

    # print(f"pay {X_payments}")
    # print(f"shares {shares}")
    # return list(funded_projects.keys()), X_payments, shares, total_cost

    # return list(funded_projects.keys()), funded_projects, X_payments, shares, total_cost
    return funded_projects, X_payments, shares, total_cost

exact_method_of_equal_shares_cost = partial(
    exact_method_of_equal_shares_uniform, 
    utility_function=cost_utility
)

greedy_project_change_cost = partial(
    greedy_project_change_cost, 
    utility_function=cost_utility
)

def exact_method_of_equal_shares_with_completion_cost_exhaustive_heuristic(pabulib_file: str, budget=0):
    """
    Run the Exact Equal Shares method with efficient budget completion.

    :param pabulib_file: Path to Pabulib file
    :return: Final selected projects, percentage of budget used, the final budget limit, the number of budget increases, the min budget increase, the max budget increase, and the average budget increase
    """
    instance, profile = parse_pabulib(pabulib_file)

    monotonic_violation = 0
    exceeded_non_exhaustive_case = 0

    min_increase = 0
    max_increase = float('inf')

    profile = profile_preprocessing(profile)
    number_total_projects = len(instance)

    initial_budget = instance.budget_limit
    if budget > 0:
        instance.budget_limit = budget
    selected_projects_with_bpb, payments, shares, total_cost = exact_method_of_equal_shares_cost(
        instance, profile
    )
    most_efficient_project_set = copy.deepcopy(selected_projects_with_bpb)
    budget_increase_count = 0
    budget_increase_list = []
    efficiency_tracker = total_cost / initial_budget

    prev_total_cost = 0
    prev_project_set = {}
    final_efficiency = 0
    total_cost = 0

    cnt = 0
    while True:
        cnt += 1
        prev_total_cost = total_cost
        prev_project_set = copy.deepcopy(selected_projects_with_bpb)
        min_budget_increase = add_opt_cost_heuristic(
            instance, profile, selected_projects_with_bpb, payments, shares
        )
        print(f"min {min_budget_increase}")

        if min_budget_increase == float("inf"):
            break

        budget_increase_count += 1

        instance.budget_limit += min_budget_increase * len(profile)

        selected_projects_with_bpb, payments, shares, total_cost = exact_method_of_equal_shares_cost(
            instance, profile
        )
        if total_cost > initial_budget:
            exceeded_non_exhaustive_case = 1
        else:
            if total_cost <= initial_budget:
                efficiency_candidate = total_cost / initial_budget

                budget_increase_list.append(min_budget_increase)

                # this is redundant I think
                prev_project_set = copy.deepcopy(selected_projects_with_bpb)
                prev_total_cost = total_cost
                #the above is redundent
                if efficiency_candidate > efficiency_tracker:
                    if exceeded_non_exhaustive_case:
                        monotonic_violation = 1 #this means that we have a greater efficiency in this case, IE the total buget used went down at some point and the efficiency increased
                    efficiency_tracker = efficiency_candidate
                    most_efficient_project_set = copy.deepcopy(selected_projects_with_bpb)
                    payments_2 = copy.deepcopy(payments)
                    total_cost_2 = total_cost
        if len(selected_projects_with_bpb) == number_total_projects:
            final_efficiency = prev_total_cost / initial_budget
            break


    # old = instance.budget_limit
    # instance.budget_limit -= min_budget_increase * len(profile)

    # Create a dictionary with the data
    data = {
        'most_efficient_project_set': [most_efficient_project_set],
        'highest_efficiency_attained': [efficiency_tracker],
        'final_project_set': [prev_project_set],
        'final_efficiency': [final_efficiency],
        'budget_increase_count': [budget_increase_count],
        'len_budget_increase_list': [len(budget_increase_list)],
        'max_budget_increase': [max(budget_increase_list)],
        'min_budget_increase': [min(budget_increase_list)],
        'avg_budget_increase': [sum(budget_increase_list)/len(budget_increase_list)],
        'monotonic_violation': [monotonic_violation]
    }

    # Convert the dictionary to a DataFrame
    pabulib_file_cleaned = pabulib_file.replace(".pb", "")
    df = pd.DataFrame(data)
    # Ensure the 'results' folder exists

    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Construct the file path in the 'results' folder
    file_path = os.path.join(results_dir, f"{pabulib_file_cleaned}.csv")

    # # Save the DataFrame to a CSV file i
    # results_dir = "results"
    # os.makedirs(results_dir, exist_ok=True)

    # # Construct the file path in the 'results' folder
    # file_path = os.path.join(results_dir, f"{pabulib_file_cleaned}.csv")

    # Save the DataFrame to a CSV file in the 'results' folder
    df.to_csv(file_path, index=False)
    # df.to_csv(f"{pabulib_file_cleaned}.csv", index=False)

    print(f"File saved as {file_path}")
    return df
    # return most_efficient_project_set, efficiency_tracker, prev_project_set, final_efficiency, budget_increase_count, len(budget_increase_list), max(budget_increase_list), min(budget_increase_list), sum(budget_increase_list)/len(budget_increase_list), monotonic_violation
    #most efficient project set, best efficiency, final project set, final effifiency, number of budget increases, number of budget increases again, max increase, min increase, average increase


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python exact_method_of_equal_shares_with_completion_cost_exhaustive_heuristic.py <file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        sys.exit(1)
    
    # Get absolute paths for input and output
    input_path = Path(file_path).resolve()
    
    # Create results directory with absolute path
    # First check for SLURM working directory
    if 'DATA' in os.environ:
        base_dir = Path(os.environ['DATA'])
    else:
        base_dir = input_path.parent
    
    # /data/coml-humanchess/univ5678/results/exact_equal_shares/cost/exact_method_of_equal_shares_with_completion_cost_exhaustive_heuristic_results
    results_dir = base_dir / "results/exact_equal_shares/cost/exact_method_of_equal_shares_with_completion_cost_exhaustive_heuristic"

    try:
        # Create results directory with parents and proper permissions
        results_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(results_dir, 0o755)  # rwxr-xr-x permissions
        
        # Get output filename
        output_filename = f"{input_path.stem}.csv"
        output_path = results_dir / output_filename
        
        # Run analysis
        print(f"Processing file: {input_path}")
        print(f"Results will be saved to: {output_path}")
        
        output_df = exact_method_of_equal_shares_with_completion_cost_exhaustive_heuristic(str(input_path))
        
        # Save results with error handling
        try:
            output_df.to_csv(output_path, index=False)
            os.chmod(output_path, 0o644)  # rw-r--r-- permissions
            print(f"Results successfully saved to: {output_path}")
        except Exception as e:
            print(f"Error saving results to {output_path}: {str(e)}")
            # Try alternative location if original fails
            fallback_path = Path.home() / "slurm_results" / output_filename
            fallback_path.parent.mkdir(parents=True, exist_ok=True)
            output_df.to_csv(fallback_path, index=False)
            print(f"Results saved to fallback location: {fallback_path}")
            
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        sys.exit(1)

    