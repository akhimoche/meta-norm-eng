import pytest
import utils.proc_funcs as up

# Sample dictionaries for testing
sample_states_0 = {
    'wall': [(1, 1), (2, 1)],
    'ground': [(1, 0), (2, 0)],
    'coin_red': [(3, 3), (4, 4)],
    'coin_blue': [(5, 5)],
    'agent': [(7, 7)]
}

sample_states_1 = {
    'wall': [(1, 1), (2, 1)],
    'ground': [(1, 0), (2, 0)],
    'coin_red': [(3, 3), (6, 6)],
    'coin_blue': [(5, 5)],
    'agent': [(8, 8)]
}

# 1. Test get_general_info
def test_get_general_info():
    result = up.get_general_info(sample_states_0)
    expected = {
        'wall': [(1, 1), (2, 1)],
        'ground': [(1, 0), (2, 0)]
    }
    assert result == expected, "get_general_info should extract only 'wall' and 'ground'."

def test_get_general_info_missing_keys():
    partial_states = {'wall': [(1, 1)], 'coin_red': [(3, 3)]}
    result = up.get_general_info(partial_states)
    expected = {'wall': [(1, 1)]}
    assert result == expected, "get_general_info should handle missing 'ground' key."

# 2. Test get_dynamic_info
def test_get_dynamic_info():
    result = up.get_dynamic_info(sample_states_0)
    expected = {
        'coin_red': [(3, 3), (4, 4)],
        'coin_blue': [(5, 5)],
        'agent': [(7, 7)]
    }
    assert result == expected, "get_dynamic_info should exclude 'wall' and 'ground'."

def test_get_dynamic_info_empty_dict():
    result = up.get_dynamic_info({})
    expected = {}
    assert result == expected, "get_dynamic_info should return an empty dictionary when input is empty."

# 3. Test get_changes
def test_get_changes_no_changes():
    result = up.get_changes(sample_states_0, sample_states_0)
    expected = {}
    assert result == expected, "get_changes should return an empty dictionary when there are no differences."

def test_get_changes_with_differences():
    result = up.get_changes(sample_states_0, sample_states_1)
    expected = {
        'coin_red': [(4, 4), (6, 6)],
        'agent': [(7, 7), (8, 8)]
    }
    assert result == expected, "get_changes should correctly identify changes between dictionaries."

def test_get_changes_additional_keys():
    dict_t0 = {'coin_red': [(1, 1)], 'agent': [(2, 2)]}
    dict_t1 = {'coin_red': [(1, 1), (3, 3)], 'agent': [(2, 2)], 'coin_blue': [(5, 5)]}
    result = up.get_changes(dict_t0, dict_t1)
    expected = {'coin_red': [(3, 3)], 'coin_blue': [(5, 5)]}
    assert result == expected, "get_changes should handle additional keys in the second dictionary."

def test_get_changes_removed_keys():
    dict_t0 = {'coin_red': [(1, 1)], 'coin_blue': [(5, 5)]}
    dict_t1 = {'coin_red': [(1, 1)]}
    result = up.get_changes(dict_t0, dict_t1)
    expected = {'coin_blue': [(5, 5)]}
    assert result == expected, "get_changes should detect keys removed from the second dictionary."