import copy
import sympy as sp


## coeff_dicts
def remove_antisymm_zeros(coeff_dict):
    """coeff_dict: a coeff_dict for an exterior vector or cochain
    result: coeff_dict, but with keys like (e1,e2,e1) removed"""
    result_dict = copy.copy(coeff_dict)
    for key in list(result_dict):
        if len(set(key)) != len(key):
            result_dict.pop(key)
    return result_dict


def remove_antisymm_zeros_cochains(coeff_dict):
    """coeff_dict: a coeff_dict for an exterior vector or cochain
    result: coeff_dict, but with keys like (e1,e1,e2) removed,
    leaving keys like (e1,e2,e1) representing nontrivial cochains"""

    result_dict = copy.copy(coeff_dict)
    for key in list(result_dict):
        if len(set(key[0 : len(key) - 1])) != len(key) - 1:
            result_dict.pop(key)
    return result_dict


def merge_coeff_dicts(dict1, dict2):
    result = {}
    for key in set(dict1.keys()).union(set(dict2.keys())):
        coeff = 0
        if key in dict1:
            coeff += dict1[key]
        if key in dict2:
            coeff += dict2[key]
        result[key] = coeff
    return remove_zeros_from_cd(result)


def remove_zeros_from_cd(coeff_dict):
    """coeff_dict: a dict with integer values
    returns: a copy of coeff_dict with all keys of value 0 removed"""
    return {A: coeff_dict[A] for A in coeff_dict if coeff_dict[A] != 0}


def str_from_coeff_dict(coeff_dict):
    """coeff_dict: a dict with keys that are printable objects
                or tuples of printable objects and integer values
    returns: a string representing the dict"""
    coeff_dict = remove_zeros_from_cd(coeff_dict)
    if coeff_dict == {}:
        return "0"
    key_list = list(coeff_dict.keys())
    result = ""
    for key in key_list:
        result += process_key_to_str(coeff_dict, key)
    if result[0:3] == " + ":
        return result[3 : len(result)]
    return result[1 : len(result)]


def process_key_to_str(coeff_dict, key):
    """adds value*key to the str r unless key is a dict, in which case
    the method is called recursively on the keys of this dict."""
    r = ""
    if type(coeff_dict[key]) == dict:  # if values are dicts, process those
        for k in coeff_dict[key]:
            r += process_key_to_str(coeff_dict[key], k)
    else:
        if type(key) == tuple:
            key_str = "(" + ",".join([str(A) for A in key]) + ")"
        else:
            key_str = str(key)
        if coeff_dict[key] == 1:
            r += " + " + key_str
        elif coeff_dict[key] == -1:
            r += " - " + key_str
        elif coeff_dict[key] != 0:
            if type(coeff_dict[key]) == sp.Add:
                coeff_str = "(" + str(coeff_dict[key]) + ")"
            else:
                coeff_str = str(coeff_dict[key])
            r += " + " + coeff_str + "*" + key_str
    return r
