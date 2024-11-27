def get_general_info(dict_of_states):
    """ Gets the general information from a dictionary of states
    such as walls and ground, returns this info in dictionary.
    """
    keys_to_extract = {'wall', 'ground'}
    sub_dict = {key: dict_of_states[key] for key in keys_to_extract if key in dict_of_states}

    return sub_dict

def get_dynamic_info(dict_of_states):
    """
    Gets dynamic info such as agents' positions and coin
    positions from a dictionary of states, consolidating keys
    that end with directions (north, south, east, west).
    """
    keys_to_avoid = {'wall', 'ground'}
    directions = {'north', 'south', 'east', 'west'}
    sub_dict = {}

    for key, value in dict_of_states.items():
        # Skip keys to avoid
        if key in keys_to_avoid:
            continue

        # Normalize the key by removing direction suffixes
        key_parts = key.split('_')
        if key_parts[-1] in directions:
            norm_key = '_'.join(key_parts[:-1])  # Remove the direction part
        else:
            norm_key = key

        # Consolidate values for normalized keys
        if norm_key not in sub_dict:
            sub_dict[norm_key] = []
        sub_dict[norm_key].extend(value)

    return sub_dict

def get_changes_symm(dict_of_states_0, dict_of_states_1):
    """ Takes two successive timesteps dictionaries, remove
    general information and return dictionary of changes
    """
    dict_of_states_0 = get_dynamic_info(dict_of_states_0)
    dict_of_states_1 = get_dynamic_info(dict_of_states_1)

    all_keys = set(dict_of_states_0.keys()).union(set(dict_of_states_1.keys()))  # Union of keys from both dictionaries
    differences = {}

    for key in all_keys:
        # get the value for keys e.g coin_red for both dictionaries
        list1 = dict_of_states_0.get(key, [])  # Default to empty list if the key doesn't exist
        list2 = dict_of_states_1.get(key, [])

        # Convert lists to sets to compare differences
        diff = list(set(list1).symmetric_difference(set(list2)))  # Symmetric difference: union of sets w/o intersection
        # i.e unique values

        if diff:  # Include only keys with differences
            differences[key] = diff

    return differences

def get_changes_diff(dict_of_states_0, dict_of_states_1):
    """ Takes two successive timesteps dictionaries, remove
    general information and return dictionary of changes
    """
    dict_of_states_0 = get_dynamic_info(dict_of_states_0)
    dict_of_states_1 = get_dynamic_info(dict_of_states_1)

    differences = {}

    for key in dict_of_states_1:
        # Ensure the key exists in dic_a and compare the lists
        set_a = set(dict_of_states_0.get(key, []))  # Use an empty list if key is not in dic_a
        set_b = set(dict_of_states_1[key])

        # Find differences in set_b compared to set_a
        diff = set_b - set_a
        if diff:
            differences[key] = list(diff)

    return differences

def dynamic_process(dict_of_dicts):
    """ Takes a timestep dictionary and returns a dictionary
    of changes (empty if no changes)
    """
    num_of_timesteps = len(dict_of_dicts.keys()) # length of trajectory sampled
    process_dict = {}

    for t in range(num_of_timesteps-1):
        pre = dict_of_dicts[f"Timestep {t}"]
        post = dict_of_dicts[f"Timestep {t+1}"]
        diff = get_changes_diff(pre, post)
        process_dict[f"a_{t}"] = diff

    return process_dict

# dic_a = {"wall": [(1,2), (3,4), (5,6)], "ground": [(1,1), (2,2),(3,3)], "green_coin": [(1,5), (2,5)], "p_green": [(2,2)]}
# dic_b = {"wall": [(1,2), (3,4), (5,6)], "ground": [(1,1), (2,2),(3,3)], "green_coin": [(1,5), (2,5), (2,3)], "p_green": [(2,3)]}

# dic_c = {"wall": [(1,2), (3,4), (5,6)], "ground": [(1,1), (2,2),(3,3)], "green_coin": [(1,5), (2,5)], "p_green": [(2,2)]}
# dic_d = {"wall": [(1,2), (3,4), (5,6)], "ground": [(1,1), (2,2),(3,3)], "green_coin": [(1,5), (2,4), (3,4), (1,1)], "p_green": [(2,5)]}

# print(get_changes_symm(dic_a, dic_b))
# print(get_changes_diff(dic_a, dic_b))

# print(get_changes_symm(dic_c, dic_d))
# print(get_changes_diff(dic_c, dic_d))