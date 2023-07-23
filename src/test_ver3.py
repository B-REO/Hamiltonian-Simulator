""" This is the first version of SnH(Simulation for nonunitary Hamiltonian).

In this program, you get a random Hamiltonian and initial state 

and simulate time-development of Hamiltonian.
"""

import math
import numpy as np

from matplotlib import pyplot as plt

def hamiltonian_maker(order):
    ''' In this part, you get a random nonunitary Hamiltonian.

    Args:
        order (int): parameter to keep the norm of the initial sate as 1

    Returns:
        np.array(2*2) (complex): nonunitary Hamiltonian

    Examples:
        Usage

        >>> print(time_Hamitonian_ver3.hamiltonian_maker(3))
        [[0.29128271+0.52816302j 0.38886643+0.75013579j]
         [0.45691417+0.19569145j 0.30565112+0.57487933j]]
    '''
    mat = np.array([[0.8 + 0.j, 3 + 0.j], [0.1 + 0.j, 0.5 + 0.j]])
    mat_con = np.conjugate(mat.T)
    zero = [1.0 + 0.j, 0.0 + 0.j]
    unit_judge = np.dot(np.dot(mat_con, mat), zero)
    while (np.abs((np.linalg.norm(unit_judge) - 1.0)) > 10**(-order)):
        for i in range(len(mat[0])):
            for j in range(len(mat[0])):
                mat[i][j] = np.random.rand() + np.random.rand()*1.j
        mat_con = np.conjugate(mat.T)
        unit_judge = np.dot(np.dot(mat_con, mat), zero)
    return mat

def initial_maker(order):
    '''In this part, you get a random quantum state.

    Args:
        order (int): parameter to accuary of quantume state
    
    Returns:
        np.array(2*1) (complex): quantum state

    Examples:
        Usage

        >>> print(time_Hamitonian_ver3.initial_maker(5))
        [(0.76426372+0.26254808j), (0.58137188+0.094815771j)]
    '''
    init_state = [1.0 + 0.j, 3.0 + 0.j]
    while (np.abs(np.linalg.norm(init_state) - 1) > 10**(-order)):
        for i in range(len(init_state)):
            init_state[i] = np.random.rand() + np.random.rand()*1.j
    return init_state

def exp_maker(mat, time_int):
    ''' In this part, you get time development of nonunitary Hamiltonian.

    Args:
        mat (np.array(2*2) (complex)): nonunitary Hamiltonian
        time_int (float): time interval for time-development

    Returns:
        np.array(2*2) (complex): time development of nonunitary Hamiltonian

    Examples:
        Usages

        >>> import time_Hamitonian_ver3 as t_H
        >>> print(t_H.exp_maker(t_H.hamiltonian_maker(5), 0.0001))
        [[ 1.00000136e+00-9.00322588e-05j  1.16144185e-05-3.11963834e-05j]
         [-1.15639188e-05-3.12081853e-05j  9.99998633e-01-9.14366202e-05j]]
    '''
    eig_val, eig_mat = np.linalg.eig(mat)
    time_ham = np.array([[0.0 + 1.j, 0.0], [0.0, 0.0 + 1.j]])
    for i in range(len(mat[0])):
        time_ham[i][i] = math.e**(-np.abs(eig_val[i])*time_int*1.0j)
    exp_ham = np.array([[1.0 + 0.j, 1.0 + 0.j], [1.0 + 0.j, 1.0 + 0.j]])
    eig_mat_con = np.array([[1.0 + 0.j, 1.0 + 0.j], [1.0 + 0.j, 1.0 + 0.j]])
    for i in range(len(eig_mat[0])):
        eig_norm = np.linalg.norm(eig_mat[i])
        for j in range(len(eig_mat[i])):
            eig_mat_con[i][j] = eig_mat[i][j]/eig_norm
    exp_ham = np.dot(eig_mat_con, np.dot(time_ham, np.linalg.inv(eig_mat_con)))
    return exp_ham

def list_maker(init, exp_ham, sim_time):
    ''' In this part, you get datalist of time-develpoment.

    Args:
        init (np.array(2*1), (complex)): initial quantum state
        exp_ham (np.array(2*2), (complex)): time-development of Hamiltonian
        sim_time (int): number of times Hamiltonian is acted upon

    Returns:
        time_list (int): 
        
        time_list (int): time to try

        emp (int): the list whose compornents are zero

        zero_prob (float): probably of zero state

        one_prob (float): probably of one state

        zero_real (float): real part of zero state

        zero_imag (float): imaginaly part of zero state

        one_real (float): real part of one state

        one_imag (float): imaginaly part of one state

        norm_list (float): norm of quantum state

        They are output in a single list. The order of the elements in the list is the same as above.

    Examples:
        Usage

        >>> import time_Hamitonian_ver3 as t_H
        >>> a = t_H.initial_maker(4)
        >>> b = t_H.hamiltonian_maker(4)
        >>> c = t_H.exp_maker(b, 0.0001)
        >>> d = t_H.list_maker(a, c, 100000)
        >>> print(type(d))
        <class 'list'>
    '''
    time_list = []
    zero_prob = []
    one_prob = []
    zero_real = []
    one_real = []
    zero_imag = []
    one_imag = []
    emp = []
    norm_list = []
    all_list = []
    for i in range(sim_time):
        init = np.dot(exp_ham, init)
        time_list.append(i + 1)
        emp.append(0)
        zero_prob.append(np.abs(init[0])**2)
        one_prob.append(np.abs(init[1])**2)
        zero_real.append(init[0].real)
        zero_imag.append(init[0].imag)
        one_real.append(init[1].real)
        one_imag.append(init[1].imag)
        norm_list.append(np.linalg.norm(init))
    all_list.append(time_list)
    all_list.append(emp)
    all_list.append(zero_prob)
    all_list.append(one_prob)
    all_list.append(zero_real)
    all_list.append(zero_imag)
    all_list.append(one_real)
    all_list.append(one_imag)
    all_list.append(norm_list)
    return all_list

def viewer (times, emp, zero_prob, one_prob,
            zero_real, zero_imag, one_real, one_imag, norm):
    ''' In this part, you make 3D model of result.

    Args:
        times (list, (int)): times of simulation
        emp (list, (int)): all elements is zero
        zero_prob(list, (float)): probably of zero state
        one_prob(list, (float)): probably of one state
        zero_real(list, (float)): real part of zero state
        zero_imag(list, (float)): imaginaly part of zero state
        one_real(list, (float)): real part of one state
        one_imag(list, (float)): imaginaly part of one state
        norm(list, (float)): norm of initial state

    How to use:
        select the combination of list you want to see

    Examples:
        ax.plot(time, zero_real, emp, "-")
        #ax.plot(time, emp, zero_imag, "-")

    Returns:
        3D plot of the combination of data list
    '''
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(times, zero_real, emp, "-")
    ax.plot(times, one_real, emp, "-")
    ax.plot(times, emp, zero_imag, "-")
    ax.plot(times, emp, one_imag, "-")
    ax.plot(times, zero_prob, one_prob, "-")
    ax.plot(times, norm, emp, "-")
    plt.xlabel("times")
    plt.ylabel("probably of zero")
    plt.clabel("probably of one")
    plt.show()

if __name__ == "__main__":
    main()