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