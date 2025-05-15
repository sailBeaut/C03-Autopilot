import numpy as np
import matplotlib.pyplot as plt
import random as rd


def calculate_accuracy_change_3step(accuracy0, accuracy1, accuracy2):
    delta_acc1 = accuracy0 - accuracy1
    delta_acc2 = accuracy1 - accuracy2
    return delta_acc1, delta_acc2

def increment_parameter(parameter, increment):
    return parameter + increment

def decrement_parameter(parameter, decrement):
    return parameter - decrement

def choose_random_param():
    parameters = [0,1,2,3]
    # 1 for k2_numvalue, 2 for c2_numvalue, 3 for a_velo, 4 for flatten_coeff
    return rd.choice(parameters)

def increment_or_decrement_parameter(chosen_parameter, increment, decrement, k2_numvalue, c2_numvalue, a_velo, flatten_coeff,  increment_or_decrement_list):
    if increment == True:  
        if chosen_parameter == 0:
            k2_update = k2_numvalue + increment_or_decrement_list[0]
        elif chosen_parameter == 1:
            c2_update = c2_numvalue + increment_or_decrement_list[1]
        elif chosen_parameter == 2:
            a_velo_update = a_velo + increment_or_decrement_list[2]
        elif chosen_parameter == 3:
            flatten_coeff_update = flatten_coeff + increment_or_decrement_list[3]
    elif decrement == True:
        if chosen_parameter == 0:
            k2_update = k2_numvalue - increment_or_decrement_list[0]
        elif chosen_parameter == 1:
            c2_update = c2_numvalue - increment_or_decrement_list[1]
        elif chosen_parameter == 2:
            a_velo_update = a_velo - increment_or_decrement_list[2]
        elif chosen_parameter == 3:
            flatten_coeff_update = flatten_coeff - increment_or_decrement_list[3]
    return k2_update, c2_update, a_velo_update, flatten_coeff_update


def calculate_average_accuracy_dof2(accuracy_DOF2):
    return np.mean(accuracy_DOF2)

def compare_accuracies_and_choose_to_continue(delta_acc1, delta_acc2, sensitivity, increment, decrement):
    if increment == True:
        if delta_acc1 > 0 and delta_acc2 > 0:
            if delta_acc1 <= sensitivity and delta_acc2 <= sensitivity:
                continue_param_inc = False
                continue_param_dec = True
            else:
                continue_param_inc = True
                continue_param_dec = False
        continue_param = True

    elif decrement == True:
        if delta_acc1 > 0 and delta_acc2 > 0:
            if delta_acc1 <= sensitivity and delta_acc2 <= sensitivity:
                continue_param_dec = False
                continue_param_inc = False #Stop the increments
            else:
                continue_param_dec = True
                continue_param_inc = False
        continue_param = True

    elif increment == False and decrement == False:
        continue_param = False
    return continue_param_inc, continue_param_dec, continue_param

def count_epochs_per_parameter(chosen_parameter, continue_list, epoch_list):
    if chosen_parameter == 0 and continue_list[0] == 1:
        epoch_list[0] += 1
    elif chosen_parameter == 1 and continue_list[1] == 1:
        epoch_list[1] += 1
    elif chosen_parameter == 2  and continue_list[2] == 1:
        epoch_list[2] += 1
    elif chosen_parameter == 3  and continue_list[3] == 1:
        epoch_list[3] += 1
    return epoch_list