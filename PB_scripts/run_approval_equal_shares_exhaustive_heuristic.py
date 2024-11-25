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

def exact_method_of_equal_shares_approval(instance, profile, utility_function=lambda x: 1, budget=0):
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
    return list(funded_projects.keys()), X_payments, shares, total_cost

def greedy_project_change_approvals(instance, profile, selected_projects, payments, project):
    # print(payments)
    #same as uniform
    """
    Determine the minimal budget increase necessary for a project to change the outcome for approval utilities.

    :param instance: Pabulib instance
    :param profile: Pabulib profile
    :param selected_projects: Current set of selected projects
    :param payments: Current payment allocations
    :param project: Project to consider for change
    :return: Minimum budget increase needed for the project to cause an outcome change
    """
    # print(f"project {project}")
    # print(f"selected_projects 0 {selected_projects[0]}")
    # print(f"Lex {project > selected_projects[0]}")
    # print(f"Lex 2 {project < selected_projects[0]}")
    # print(f"selected_projects 0 {selected_projects[0]}")

    # instance, profile, selected_projects_with_bpb, payments, p, L, utility_function
    num_voters = len(profile)
    project_cost = project.cost
    initial_budget = instance.budget_limit
    profile = profile_preprocessing(profile)

    # Identify supporters of the project
    supporters = [voter['name'] for voter in profile if project in voter['approved']]
    supporters_not_paying = [voter for voter in supporters if payments.get((voter, project), 0) == 0] #OpX

    # Compute the leftover budget for each supporter
    leftover_budgets = {
        voter: ((initial_budget / num_voters)
        - sum(payments.get((voter, p), 0) for p in selected_projects))
        for voter in supporters_not_paying
    }
    # print(f"payments {payments}")
    # Compute the maximum payment each supporter is currently making, (project, payment)
    max_payments = {
        voter: max(((p, payments.get((voter, p), 0)) for p in selected_projects), key=lambda x: x[1], default=(None, 0))
        for voter in supporters_not_paying
    }
    # print(f"max payments: {max_payments}")

    # Sort the supporters by their leftover budgets and max payments
    #leftovers is (voter, payment)
    sorted_leftovers = sorted(leftover_budgets.items(), key=lambda x: x[1])

    #sorted_max_payments is (voter, (project, payment))
    sorted_max_payments = sorted(max_payments.items(), key=lambda x: (x[1][1], x[1][0])) #adds tie breaking

    # print(f"sorted_max_payments[0][0][0] {sorted_max_payments[0][0][0]}")


    # print(f"Length supporters {len(supporters)}")

    # Initialize variables
    min_increase = float("inf")
    solvent = set()  # Voters who would deviate from their current projects, solvent


    # print(f"Length supporters not paying {len(supporters_not_paying)}")
    # print(f"Length supporters paying {len([voter for voter in supporters if payments.get((voter, project), 0) > 0])}")
    # print(f"Selected projects {selected_projects}")
    # print(f"Current project {project}")

    liquid = set(supporters_not_paying)  # Voters willing to allocate leftover budgetm liquid
    i, j = 0, 0
    while liquid or solvent: #deviant is solvant

        if i >= len(sorted_leftovers):  # Out-of-bounds access
            break

        # Calculate the per voter price (pvp)
        total_voters = (
            len(liquid)
            + len(solvent)
            + len(
                [voter for voter in supporters if payments.get((voter, project), 0) > 0] #supporters already paying
            )
        )
        pvp = project_cost / total_voters


        #TODO add lex tie-breaking
        # Check if current max payment is less than pvp (deviant case)
        # print("START")
        # print(f"pvp {float(pvp)}")
        if j < len(sorted_max_payments):
            if sorted_max_payments[j][1][1] < pvp or (sorted_max_payments[j][1][1] == pvp and sorted_max_payments[j][1][0].name > project.name):
                # print(f"sorted_max_payments[j][1][1] {sorted_max_payments}")
                solvent.discard(sorted_max_payments[j][0])
                j += 1
        # Check if current leftover budget is less than pvp (reassign to deviant)
        #voter = sorted_leftovers[0][0] is the voter
        # elif sorted_max_payments[i][1][1] > pvp or (sorted_max_payments[i][1][1] == pvp and sorted_max_payments[i][1][0] > project):
        #max payments: {voter: (project, payment)}

            #we have here that max payment of c_v_i <= pvp if this fails, but that their leftover is > pvp? This is a problemo
            elif sorted_leftovers[i][0] in liquid and max_payments[sorted_leftovers[i][0]][1] > pvp or (max_payments[sorted_leftovers[i][0]][1] == pvp and max_payments[sorted_leftovers[i][0]][0].name < project.name): #should this be the other way? Should just define a lex
                # print(f"max_payments[sorted_leftovers[i][0]][1] {max_payments[sorted_leftovers[i][0]][1]} pvp {pvp}")
                # print(f"max_payments[sorted_leftovers[i][0]][0] {max_payments[sorted_leftovers[i][0]][0]} project {project}")
                # print(f"max_payments[sorted_leftovers[i][0]][0] not name {max_payments[sorted_leftovers[i][0]][0]} not name {project}")
                # print(f"max_payments[sorted_leftovers[i][0]][0] name {max_payments[sorted_leftovers[i][0]][0].name} name {project.name}")
                # print(f"max_payments[sorted_leftovers[i][0]][0] {max_payments[sorted_leftovers[i][0]][0].name > project.name}")
                # print(f"max_payments[sorted_leftovers[i][0]][0] {max_payments[sorted_leftovers[i][0]][0] > project}")
                
                solvent.add(sorted_leftovers[i][0])
                liquid.remove(sorted_leftovers[i][0])
                i += 1
                
            else:
                # Calculate the necessary budget increment
                # print(f"max_payments[sorted_leftovers[i][0]] {float(max_payments[sorted_leftovers[i][0]][1])}")
                # print(f"sorted_leftovers overall: {sorted_leftovers}")
                # print(f"pvp {float(pvp)}")
                required_increase = pvp - sorted_leftovers[i][1] #if the project is implemented then clearly pvp has to be higher than the leftover budget 
                # print(f"required_increase {float(required_increase)}")
                min_increase = min(min_increase, required_increase)
                liquid.remove(sorted_leftovers[i][0])
                i += 1
                    #we have here that max payment of c_v_i <= pvp if this fails, but that their leftover is > pvp? This is a problemo
        elif sorted_leftovers[i][0] in liquid and max_payments[sorted_leftovers[i][0]][1] > pvp or (max_payments[sorted_leftovers[i][0]][1] == pvp and max_payments[sorted_leftovers[i][0]][0].name < project.name):
            solvent.add(sorted_leftovers[i][0])
            liquid.remove(sorted_leftovers[i][0])
            i += 1
            # print(f"i outer loop {i}")
        else:
            # Calculate the necessary budget increment
            # print(f"max_payments[sorted_leftovers[i][0]] {float(max_payments[sorted_leftovers[i][0]][1])}")
            # print(f"sorted_leftovers overall: {sorted_leftovers}")
            # print(f"pvp {float(pvp)}")
            required_increase = pvp - sorted_leftovers[i][1] #if the project is implemented then clearly pvp has to be higher than the leftover budget 
            # print(f"required_increase {float(required_increase)}")
            min_increase = min(min_increase, required_increase)
            liquid.remove(sorted_leftovers[i][0])
            i += 1
        

    return min_increase

def add_opt_approval(instance, profile, selected_projects, payments, shares):
    print("NEW ADD OPT RUN")
    """
    Determine the minimum budget increase needed for the solution to become unstable
    for approval utilities.

    :param instance: Pabulib instance object
    :param profile: Pabulib profile object
    :param selected_projects: Set of currently selected projects
    :param payments: Current payment allocations
    :return: Minimum d > 0 such that (selected_projects, payments) is unstable for instance with increased budget
    """
    voter_set = profile
    projects = instance.project_meta

    d = float("inf")

    budget_dict = [[v['name'],shares[v['name']]] for v in voter_set]

    # Sort the dictionary by value
    budget_dict.sort(key=lambda x: x[1])

    # Call GreedyProjectChange for each project
    for p in projects:
        GP = greedy_project_change_approvals(
            instance, profile, selected_projects, payments, p #TODO why do we not need L_Op here
        )
        if GP < 0:
            print(f"P_negative_GP {p}")

        d = min(d, GP)
        print(d)

    return d

def add_opt_approval_heuristic(instance, profile, selected_projects, payments, shares):
    print("NEW ADD OPT RUN")
    """
    Determine the minimum budget increase needed for the solution to become unstable
    for approval utilities.

    :param instance: Pabulib instance object
    :param profile: Pabulib profile object
    :param selected_projects: Set of currently selected projects
    :param payments: Current payment allocations
    :return: Minimum d > 0 such that (selected_projects, payments) is unstable for instance with increased budget
    """
    voter_set = profile
    projects = instance.project_meta

    d = float("inf")

    budget_dict = [[v['name'],shares[v['name']]] for v in voter_set]

    # Sort the dictionary by value
    budget_dict.sort(key=lambda x: x[1])

    # Call GreedyProjectChange for each project
    for p in projects:
        if p in selected_projects:
            continue
        GP = greedy_project_change_approvals(
            instance, profile, selected_projects, payments, p #TODO why do we not need L_Op here
        )
        if GP < 0:
            print(f"P_negative_GP {p}")

        d = min(d, GP)
        print(d)

    return d

def exact_method_of_equal_shares_with_completion_approval_exhasutive_heuristic(pabulib_file: str, budget=0):
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
    selected_projects_with_bpb, payments, shares, total_cost = exact_method_of_equal_shares_approval(
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
        min_budget_increase = add_opt_approval_heuristic(
            instance, profile, selected_projects_with_bpb, payments, shares
        )
        print(f"min {min_budget_increase}")

        if min_budget_increase == float("inf"):
            break

        budget_increase_count += 1

        instance.budget_limit += min_budget_increase * len(profile)

        selected_projects_with_bpb, payments, shares, total_cost = exact_method_of_equal_shares_approval(
            instance, profile
        )
        if total_cost > initial_budget:
            exceeded_non_exhaustive_case = 1
        else:
            if total_cost <= initial_budget:
                efficiency_candidate = total_cost / initial_budget
                # if min_budget_increase <  min_increase:
                #     min_increase = min_budget_increase

                # if min_budget_increase >  max_increase:
                #     max_increase = min_budget_increase

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
    
    df.to_csv(f"{pabulib_file_cleaned}.csv", index=False)

    print(f"File saved as {file_path}")
    return df
    #return most_efficient_project_set, efficiency_tracker, prev_project_set, final_efficiency, budget_increase_count, len(budget_increase_list), max(budget_increase_list), min(budget_increase_list), sum(budget_increase_list)/len(budget_increase_list), monotonic_violation
    #most efficient project set, best efficiency, final project set, final effifiency, number of budget increases, number of budget increases again, max increase, min increase, average increase

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_approval_equal_shares.py <file_path>")
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
    # /data/coml-humanchess/univ5678/results/exact_equal_shares/approval/exact_method_of_equal_shares_with_completion_approval_exhasutive_heuristic_results
    results_dir = base_dir / "results/exact_equal_shares/approval/exact_method_of_equal_shares_with_completion_approval_exhasutive_heuristic_results"
    
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
        
        output_df = exact_method_of_equal_shares_with_completion_approval_exhasutive_heuristic(str(input_path))
        
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