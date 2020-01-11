'''
FixedPoints Class
Supports FixedPointFinder
Written using Python 2.7.12 and TensorFlow 1.10.
@ Matt Golub, October 2018.
Please direct correspondence to mgolub@stanford.edu.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb
import numpy as np
import cPickle

class FixedPoints(object):
    '''
    A class for storing fixed points and associated data.
    '''

    # List of class attributes
    # All of these refer to Numpy arrays with axis 0 as the batch dimension.
    # Thus, each is concatenatable using np.concatenate(..., axis=0).
    _data_attrs = [
            'xstar',
            'x_init',
            'inputs',
            'F_xstar',
            'qstar',
            'dq',
            'n_iters',
            'J_xstar',
            'eigval_J_xstar',
            'eigvec_J_xstar'
            ]

    def __init__(self,
                 xstar=None,
                 x_init=None,
                 inputs=None,
                 F_xstar=None,
                 qstar=None,
                 dq=None,
                 n_iters=None,
                 J_xstar=None,
                 eigval_J_xstar=None,
                 eigvec_J_xstar=None,
                 do_alloc_nan=False,
                 n=None,
                 n_states=None,
                 n_inputs=None,
                 tol_unique=1e-3,
                 dtype=np.float32,
                 verbose=False):
        '''
        Initializes a FixedPoints object with all input arguments as class
        properties.

        Optional args:

            xstar: [n x n_states] numpy array with row xstar[i, :]
            specifying an the fixed point identified from x_init[i, :].
            Default: None.

            x_init: [n x n_states] numpy array with row x_init[i, :]
            specifying the initial state from which xstar[i, :] was optimized.
            Default: None.

            inputs: [n x n_inputs] numpy array with row inputs[i, :]
            specifying the input to the RNN during the optimization of
            xstar[i, :]. Default: None.

            F_xstar: [n x n_states] numpy array with F_xstar[i, :]
            specifying RNN state after transitioning from the fixed point in
            xstar[i, :]. If the optimization succeeded (e.g., to 'tol') and
            identified a stable fixed point, the state should not move
            substantially from the fixed point (i.e., xstar[i, :] should be
            very close to F_xstar[i, :]). Default: None.

            qstar: [n,] numpy array with qstar[i] containing the
            optimized objective (1/2)(x-F(x))^T(x-F(x)), where
            x = xstar[i, :]^T and F is the RNN transition function (with the
            specified constant inputs). Default: None.

            dq: [n,] numpy array with dq[i] containing the absolute
            difference in the objective function after (i.e., qstar[i]) vs
            before the final gradient descent step of the optimization of
            xstar[i, :]. Default: None.

            n_iters: [n,] numpy array with n_iters[i] as the number of
            gradient descent iterations completed to yield xstar[i, :].
            Default: None.

            J_xstar: [n x n_states x n_states] numpy array with
            J_xstar[i, :, :] containing the Jacobian of the RNN state
            transition function at fixed point xstar[i, :]. Default: None,
            which results in an appropriately sized numpy array of NaNs.
            Default: None.

            do_alloc_nan: Bool indicating whether to initialize all data
            attributes (all optional args above) as NaN-filled numpy arrays.
            Default: False.

                If True, n, n_states and n_inputs must be provided. These values
                are otherwise ignored:

                n: Positive int specifying the number of fixed points to
                allocate space for.

                n_states: Positive int specifying the dimensionality of the
                network state (a.k.a. the number of hidden units).

                n_inputs: Positive int specifying the dimensionality of the
                network inputs.

            tol_unique: Positive scalar specifying the numerical precision
            required to label two fixed points as being unique from one
            another. Two fixed points will be considered unique if they differ
            by this amount (or more) along any dimension. This tolerance is
            used to discard numerically similar fixed points. Default: 1e-3.

            dtype: Data type for representing all of the object's data.
            Default: numpy.float32.

            verbose: Bool indicating whether to print status updates.

        Note:
            xstar, x_init, inputs, F_xstar, and J_xstar are all numpy arrays,
            regardless of whether that type is consistent with the state type
            of the rnncell from which they originated (i.e., whether or not
            the rnncell is an LSTM). This design decision reflects that a
            Jacobian is most naturally expressed as a single matrix (as
            opposed to a collection of matrices representing interactions
            between LSTM hidden and cell states). If one requires state
            representations as type LSTMStateCell, use
            FixedPointFinder._convert_to_LSTMStateTuple.

        Returns:
            None.

        '''

        self.tol_unique = tol_unique
        self.dtype = dtype
        self.verbose = verbose

        if do_alloc_nan:
            if n is None:
                raise ValueError('n must be provided if '
                                 'do_alloc_nan == True.')
            if n_states is None:
                raise ValueError('n_states must be provided if '
                                 'do_alloc_nan == True.')
            if n_inputs is None:
                raise ValueError('n_inputs must be provided if '
                                 'do_alloc_nan == True.')

            self.n = n
            self.n_states = n_states
            self.n_inputs = n_inputs

            self.xstar = self._alloc_nan((n, n_states))
            self.x_init = self._alloc_nan((n, n_states))
            self.inputs = self._alloc_nan((n, n_inputs))
            self.F_xstar = self._alloc_nan((n, n_states))
            self.qstar = self._alloc_nan((n))
            self.dq = self._alloc_nan((n))
            self.n_iters = self._alloc_nan((n))
            self.J_xstar = self._alloc_nan((n, n_states, n_states))

            self.eigval_J_xstar = self._alloc_nan((n, n_states))
            self.eigvec_J_xstar = self._alloc_nan((n, n_states, n_states))
            self.has_decomposed_jacobians = False

        else:
            if xstar is not None:
                self.n, self.n_states = xstar.shape
            elif x_init is not None:
                self.n, self.n_states = x_init.shape
            elif F_xstar is not None:
                self.n, self.n_states = F_xstar.shape
            elif J_xstar is not None:
                self.n, self.n_states, _ = J_xstar.shape
            else:
                self.n = None
                self.n_states = None

            if inputs is not None:
                self.n_inputs = inputs.shape[1]
                if self.n is None:
                    self.n = inputs.shape[0]
            else:
                self.n_inputs = None

            self.xstar = xstar
            self.x_init = x_init
            self.inputs = inputs
            self.F_xstar = F_xstar
            self.qstar = qstar
            self.dq = dq
            self.n_iters = n_iters
            self.J_xstar = J_xstar
            self.eigval_J_xstar = eigval_J_xstar
            self.eigvec_J_xstar = eigvec_J_xstar
            self.has_decomposed_jacobians = eigval_J_xstar is not None

    def _alloc_nan(self, shape, dtype=None):
        '''Returns a nan-filled numpy array.

        Args:
            shape: int or tuple representing the shape of the desired numpy
            array.

        Returns:
            numpy array with the desired shape, filled with NaNs.

        '''
        if dtype is None:
            dtype = self.dtype

        result = np.zeros(shape, dtype=dtype)
        result.fill(np.nan)
        return result

    def get_unique(self):
        '''Identifies unique fixed points. Among duplicates identified,
        currently an arbitrary one is retained.

        Args:
            None.

        Returns:
            A FixedPoints object containing only the unique fixed points and
            their associated data. Uniqueness is determined down to tol_unique.
        '''

        ''' To do:
                Consider updating to leverage __contains__. This would likely
                involve slow python for loops.

                Of a set of matching fixed points (down to tol_unique), retain
                the one with the smallest qstar. Currently, an arbitrary match
                is retained.
        '''

        def unique_rows(x, approx_tol):
            # Quick and dirty. Can update using pdist if necessary
            d = int(np.round(np.max([0 -np.log10(approx_tol)])))
            ux, idx = np.unique(x.round(decimals=d),
                                axis=0,
                                return_index=True)
            return ux, idx

        if self.xstar is not None:
            if self.inputs is not None:
                data = np.concatenate((self.xstar, self.inputs), axis=1)
            else:
                data = self.xstar
        else:
            raise ValueError('Cannot find unique fixed points because '
                'self.xstar is None.')

        unique_data, idx = unique_rows(data, self.tol_unique)

        return self[idx]

    @staticmethod
    def concatenate(fps_seq):
        ''' Join a sequence of FixedPoints objects.

        Args:
            fps_seq: sequence of FixedPoints objects. All FixedPoints objects must have the following attributes in common:
                n_states
                n_inputs
                has_decomposed_jacobians

        Returns:
            A FixedPoints objects containing the concatenated FixedPoints data.
        '''

        kwargs = {}
        for attr_name in FixedPoints._data_attrs:
            if all((hasattr(fps, attr_name) for fps in fps_seq)):
                cat_list = [getattr(fps, attr_name) for fps in fps_seq]
                cat_attr = np.concatenate(cat_list, axis=0)
                kwargs[attr_name] = cat_attr

        return FixedPoints(**kwargs)

    def update(self, new_fps):
        ''' Combines the entries from another FixedPoints object into this
        object.

        All non-data class attributes remain as in this FixedPoints object
        (e.g., tol_unique, dtype, verbose).

        Args:
            new_fps: a FixedPoints object containing the entries to be
            incorporated into this FixedPoints object.

        Returns:
            None

        Raises:
            ValueError if any data attributes are found in one but not both
            FixedPoints objects (especially relevant for decomposed Jacobians).
        '''

        for attr_name in self._data_attrs:
            this_has = hasattr(self, attr_name)
            that_has = hasattr(new_fps, attr_name)
            if this_has and that_has:
                cat_attr = np.concatenate(
                    (getattr(self, attr_name),
                    getattr(new_fps, attr_name)),
                    axis=0)
                setattr(self, attr_name, cat_attr)
            elif this_has != that_has:
                raise ValueError('One but not both FixedPoints objects have %s. FixedPoints.update does not currently support this configuration.' % attr_name)

        self.n = self.n + new_fps.n

    def decompose_jacobians(self, do_batch=True, str_prefix=''):
        '''Adds the following fields to the FixedPoints object:

        eigval_J_xstar: [n x n_states] numpy array containing with
        eigval_J_xstar[i, :] containing the eigenvalues of J_xstar[i, :, :].

        eigvec_J_xstar: [n x n_states x n_states] numpy array containing with
        eigvec_J_xstar[i, :, :] containing the eigenvectors of
        J_xstar[i, :, :].
        '''

        if self.has_decomposed_jacobians:
            print('%sJacobians have already been decomposed, '
                'not repeating.' % str_prefix)
            return

        n = self.n # number of FPs represented in this object
        n_states = self.n_states # dimensionality of each state

        if do_batch:
            # Batch eigendecomposition
            print('%sDecomposing Jacobians in a single batch.' % str_prefix)

            # Check for NaNs in Jacobians
            valid_J_idx = ~np.any(np.isnan(self.J_xstar), axis=(1,2))

            if np.all(valid_J_idx):
                # No NaNs, nothing to worry about.
                e_vals_unsrt, e_vecs_unsrt = np.linalg.eig(self.J_xstar)
            else:
                # Set eigen-data to NaN if there are any NaNs in the
                # corresponding Jacobian.
                e_vals_unsrt = self._alloc_nan(
                    (n, n_states), dtype=np.complex64)
                e_vecs_unsrt = self._alloc_nan(
                    (n, n_states, n_states), dtype=np.complex64)

                e_vals_unsrt[valid_J_idx], e_vecs_unsrt[valid_J_idx] = \
                    np.linalg.eig(self.J_xstar[valid_J_idx])

        else:
            print('%sDecomposing Jacobians one-at-a-time.' % str_prefix)
            e_vals = []
            e_vecs = []
            for J in self.J_xstar:

                if np.any(np.isnan(J)):
                    e_vals_i = self._alloc_nan((n_states,))
                    e_vecs_i = self._alloc_nan((n_states, n_states))
                else:
                    e_vals_i, e_vecs_i = np.linalg.eig(J)

                e_vals.append(np.expand_dims(e_vals_i, axis=0))
                e_vecs.append(np.expand_dims(e_vecs_i, axis=0))

            e_vals_unsrt = np.concatenate(e_vals, axis=0)
            e_vecs_unsrt = np.concatenate(e_vecs, axis=0)

        print('%sSorting by Eigenvalue magnitude.' % str_prefix)
        # Sort by eigenvalue magnitude in decreasing order
        sort_idx = np.argsort(np.abs(e_vals_unsrt))[:,::-1]

        # Apply the sort
        # There must be a faster way, but I'm too lazy to find it at the moment
        self.eigval_J_xstar = \
            self._alloc_nan((n, n_states), dtype=np.complex64)
        self.eigvec_J_xstar = \
            self._alloc_nan((n, n_states, n_states), dtype=np.complex64)

        for k in range(n):
            sort_idx_k = sort_idx[k]
            self.eigval_J_xstar[k] = e_vals_unsrt[k][sort_idx_k]
            self.eigvec_J_xstar[k] = e_vecs_unsrt[k][:, sort_idx_k]

        self.has_decomposed_jacobians = True

    def __setitem__(self, index, fps):
        '''Implements the assignment opperator.

        All compatible data from fps are copied. This excludes tol_unique,
        dtype, n, n_states, and n_inputs, which retain their original values.

        Usage:
            fps_to_be_partially_overwritten[index] = fps
        '''

        if not isinstance(fps, FixedPoints):
            raise TypeError('fps must be a FixedPoints object but was %s.' %
                type(fps))

        if isinstance(index, int):
            # Force the indexing that follows to preserve numpy array ndim
            index = range(index, index+1)

        ''' Future work, test the following replacement for the rest of the
        code in this function.

        for attr_name in self._data_attrs:
            attr = getattr(self, attr_name)
            if attr is not None:
                attr[index] = getattr(fps, attr_name)
        '''

        if self.xstar is not None:
            self.xstar[index] = fps.xstar

        if self.x_init is not None:
            self.x_init[index] = fps.x_init

        if self.inputs is not None:
            self.inputs[index] = fps.inputs

        if self.F_xstar is not None:
            self.F_xstar[index] = fps.F_xstar

        if self.qstar is not None:
            self.qstar[index] = fps.qstar

        if self.dq is not None:
            self.dq[index] = fps.dq

        if self.J_xstar is not None:
            self.J_xstar[index] = fps.J_xstar

        if self.has_decomposed_jacobians:
            self.eigval_J_xstar[index] = fps.eigval_J_xstar
            self.eigvec_J_xstar[index] = fps.eigvec_J_xstar

    def __getitem__(self, index):
        '''Indexes into a subset of the fixed points and their associated data.

        Usage:
            fps_subset = fps[index]

        Args:
            index: a slice object for indexing into the FixedPoints data.

        Returns:
            A FixedPoints object containing a subset of the data from the
            current FixedPoints object, as specified by index.
        '''

        if isinstance(index, int):
            # Force the indexing that follows to preserve numpy array ndim
            index = range(index, index+1)

        xstar = self._safe_index(self.xstar, index)
        x_init = self._safe_index(self.x_init, index)
        inputs = self._safe_index(self.inputs, index)
        F_xstar = self._safe_index(self.F_xstar, index)
        qstar = self._safe_index(self.qstar, index)
        dq = self._safe_index(self.dq, index)
        n_iters = self._safe_index(self.n_iters, index)
        J_xstar = self._safe_index(self.J_xstar, index)

        if self.has_decomposed_jacobians:
            eigval_J_xstar = self._safe_index(self.eigval_J_xstar, index)
            eigvec_J_xstar = self._safe_index(self.eigvec_J_xstar, index)
        else:
            eigval_J_xstar = None
            eigvec_J_xstar = None

        dtype = self.dtype
        tol_unique = self.tol_unique

        indexed_fps = FixedPoints(xstar,
            x_init=x_init,
            inputs=inputs,
            F_xstar=F_xstar,
            qstar=qstar,
            dq=dq,
            n_iters=n_iters,
            J_xstar = J_xstar,
            eigval_J_xstar=eigval_J_xstar,
            eigvec_J_xstar=eigvec_J_xstar,
            dtype=dtype,
            tol_unique=tol_unique)

        return indexed_fps

    def __len__(self):
        '''Returns the number of fixed points stored in the object.'''
        return self.n_inits

    @staticmethod
    def _safe_index(x, idx):
        '''Safe method for indexing into a numpy array that might be None.

        Args:
            x: Either None or a numpy array.

            idx: Positive int or index-compatible argument for indexing into x.

        Returns:
            Self explanatory.

        '''
        if x is None:
            return None
        else:
            return x[idx]

    def find(self, fp):
        '''Searches in the current FixedPoints object for matches to a
        specified fixed point. Two fixed points are defined as matching
        if none of the element-wise differences between their .xstar or
        .inputs properties differ by more than tol_unique.

        Args:
            fp: A FixedPoints object containing exactly one fixed point.

        Returns:
            [n_occurrences,] numpy array specifying indices into the current
            FixedPoints object where matches to fp were found.
        '''

        # If not found or comparison is impossible (due to type or shape),
        # follow convention of np.where and return an empty numpy array.
        result = np.array([], dtype=int)

        if isinstance(fp, FixedPoints):
            if fp.n_states == self.n_states and fp.n_inputs == self.n_inputs:

                self_data = np.concatenate((self.xstar, self.inputs), axis=1)
                arg_data = np.concatenate((fp.xstar, fp.inputs), axis=1)

                elementwise_abs_diff = np.abs(self_data - arg_data)
                hits = np.all(elementwise_abs_diff <= self.tol_unique, axis=1)

                result = np.where(hits)[0]

        return result

    def __contains__(self, xstar):
        '''Checks whether a specified fixed point is contained in the object.

        Args:
            xstar: [n_states,] or [1, n_states] numpy array specifying the fixed
            point of interest.

        Returns:
            bool indicating whether self.xstar contains any fixed point with a maximum elementwise difference from xstar that is less than tol_unique.
        '''
        idx = self.find(xstar)

        return idx.size > 0

    def save(self, save_path):
        '''Saves all data contained in the FixedPoints object.

        Args:
            save_path: A string containing the path at which to save
            (including directory, filename, and arbitrary extension).

        Returns:
            None.
        '''
        if self.verbose:
            print('Saving FixedPoints object.')
        file = open(save_path,'w')
        file.write(cPickle.dumps(self.__dict__))
        file.close()

    def restore(self, restore_path):
        '''Restores data from a previously saved FixedPoints object.

        Args:
            restore_path: A string containing the path at which to find a
            previously saved FixedPoints object (including directory, filename,
            and extension).

        Returns:
            None.
        '''
        if self.verbose:
            print('Restoring FixedPoints object.')
        file = open(restore_path,'r')
        restore_data = file.read()
        file.close()
        self.__dict__ = cPickle.loads(restore_data)

    def print_summary(self):
        '''Prints a summary of the fixed points.

        Args:
            None.

        Returns:
            None.
        '''

        print('\nThe q function at the fixed points:')
        print(self.qstar)

        print('\nChange in the q function from the final iteration '
              'of each optimization:')
        print(self.dq)

        print('\nNumber of iterations completed for each optimization:')
        print(self.n_iters)

        print('\nThe fixed points:')
        print(self.xstar)

        print('\nThe fixed points after one state transition:')
        print(self.F_xstar)
        print('(these should be very close to the fixed points)')

        if self.J_xstar is not None:
            print('\nThe Jacobians at the fixed points:')
            print(self.J_xstar)
