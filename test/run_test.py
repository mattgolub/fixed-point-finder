'''
run_test.py
Tests the FixedPointFinder package
Written using Python 2.7.12
@ Matt Golub, October 2018.
Please direct correspondence to mgolub@stanford.edu.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb
import numpy as np
import tensorflow as tf
import sys

FIXED_POINT_FINDER_PATH = '../'
TEST_PATH = './'
sys.path.insert(0, FIXED_POINT_FINDER_PATH)
from FixedPointFinder import FixedPointFinder
from FixedPoints import FixedPoints
from test_utils import get_ground_truth_path, build_test_rnn, generate_initial_states_and_inputs

'''Test the correctness of fixed points identified by FixedPointFinder on a
set of RNN's where ground truth fixed points have been previously identified,
numerically confirmed, and saved for comparison.
'''

N_HIDDEN_LIST = [2, 3, 4]
N_INPUTS = 1

n_tests = len(N_HIDDEN_LIST)
fpf_hps = {'do_rerun_outliers': True, 'verbose': False}
session = tf.Session()

did_pass_tests1 = [False] * n_tests
did_pass_tests2 = [False] * n_tests

for test_idx in range(n_tests):

    n_hidden = N_HIDDEN_LIST[test_idx]
    test_no = test_idx+1

    print('')
    print('******************************************************************')
    print('Running test %d of %d.' % (test_no, n_tests))
    print('******************************************************************')
    print('')

    ground_truth_path = get_ground_truth_path(TEST_PATH, n_hidden)
    ground_truth_fps = FixedPoints()
    ground_truth_fps.restore(ground_truth_path)

    # *************************************************************************
    # STEP 1: Create an RNN with prespecified parameters **********************
    # *************************************************************************
    rnn_cell = build_test_rnn(n_hidden, N_INPUTS, session)
    initial_states, inputs = generate_initial_states_and_inputs(
        n_hidden, N_INPUTS)

    # *************************************************************************
    # STEP 2: Find, analyze, and visualize the fixed points of the RNN ********
    # *************************************************************************

    # Setup the fixed point finder
    fpf = FixedPointFinder(rnn_cell,
                           session,
                           **fpf_hps)

    # Run the fixed point finder
    unique_fps, all_fps = fpf.find_fixed_points(initial_states, inputs)

    print('%d unique fixed point(s) identified.' %
          unique_fps.n)
    print('%d unique fixed point(s) in ground truth set.\n' %
          ground_truth_fps.n)

    # Count the number of identified fixed points that are indeed fixed points
    # (according to the ground truth set).
    n_correct = 0
    for idx_unique in range(unique_fps.n):
        if unique_fps[idx_unique] in ground_truth_fps:
            n_correct += 1

    if n_correct == unique_fps.n:
        print('Test %d.1: PASSED.' % test_no)
        print('\tAll identified fixed points match fixed points')
        print('\tin the ground truth set.')
        did_pass_tests1[test_idx] = True
    else:
        print('Test %d.1: FAILED.' % test_no)
        print('\t%d of %d identified fixed points do not have matches' %
              (unique_fps.n - n_correct, unique_fps.n))
        print('\tin the ground truth set.')

    # Count the number of ground truth fixed points that were found.
    n_gt_found = 0
    for idx_gt in range(ground_truth_fps.n):
        if ground_truth_fps[idx_gt] in unique_fps:
            n_gt_found += 1

    if n_gt_found == ground_truth_fps.n:
        print('Test %d.2: PASSED.' % test_no)
        print('\tAll ground truth fixed points match fixed points')
        print('\tin the identified set.')
        did_pass_tests2[test_idx] = True
    else:
        print('Test %d.2: FAILED.' % test_no)
        print('\t%d of %d ground truth fixed points do not have matches' %
              (ground_truth_fps.n - n_gt_found, ground_truth_fps.n))
        print('\tin the identified set.')

print('')
if all(did_pass_tests1) and all(did_pass_tests2):
    print('FixedPointFinder PASSED all tests.')
else:
    print('FixedPointFinder FAILED one or more tests.')
print('')