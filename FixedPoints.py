'''
FixedPoints Class
Supports FixedPointFinder

Written for Python 3.6.9
@ Matt Golub, October 2018

If you are using FixedPointFinder in research to be published, 
please cite our accompanying paper in your publication:

Golub and Sussillo (2018), "FixedPointFinder: A Tensorflow toolbox for 
identifying and characterizing fixed points in recurrent neural networks," 
Journal of Open Source Software, 3(31), 1003.
https://doi.org/10.21105/joss.01003

Please direct correspondence to mgolub@cs.washington.edu
'''

import pdb
import numpy as np
import pickle

from Timer import Timer

class FixedPoints(object):
    '''
    A class for storing fixed points and associated data.
    '''

    ''' List of class attributes that represent data corresponding to fixed
    points. All of these refer to Numpy arrays with axis 0 as the batch
    dimension. Thus, each is concatenatable using np.concatenate(..., axis=0).
    '''
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
            'eigvec_J_xstar',
            'is_stable',
            'cond_id']

    ''' List of class attributes that apply to all fixed points
    (i.e., these are not indexed per fixed point). '''
    _nonspecific_attrs = [
        'dtype',
        'dtype_complex',
        'tol_unique',
        'verbose',
        'do_alloc_nan']

    def __init__(self,
                 xstar=None, # Fixed-point specific data
                 x_init=None,
                 inputs=None,
                 F_xstar=None,
                 qstar=None,
                 dq=None,
                 n_iters=None,
                 J_xstar=None,
                 eigval_J_xstar=None,
                 eigvec_J_xstar=None,
                 is_stable=None,
                 cond_id=None,
                 n=None,
                 n_states=None,
                 n_inputs=None, # Non-specific data
                 do_alloc_nan=False,
                 tol_unique=1e-3,
                 dtype=np.float32,
                 dtype_complex=np.complex64,
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

            eigval_J_xstar: [n x n_states] numpy array with
            eigval_J_xstar[i, :] containing the eigenvalues of
            J_xstar[i, :, :].

            eigvec_J_xstar: [n x n_states x n_states] numpy array with
            eigvec_J_xstar[i, :, :] containing the eigenvectors of
            J_xstar[i, :, :].

            is_stable: [n,] numpy array with is_stable[i] indicating as bool
            whether xstar[i] is a stable fixed point.

            do_alloc_nan: Bool indicating whether to initialize all data
            attributes (all optional args above) as NaN-filled numpy arrays.
            Default: False.

                If True, n, n_states and n_inputs must be provided. These
                values are otherwise ignored:

                n: Positive int specifying the number of fixed points to
                allocate space for.

                n_states: Positive int specifying the dimensionality of the
                network state (a.k.a. the number of hidden units).

                n_inputs: Positive int specifying the dimensionality of the
                network inputs.

            tol_unique: Positive scalar specifying the numerical precision
            required to label two fixed points as being unique from one
            another. Two fixed points are considered unique if the 2-norm of
            the difference between their concatenated (xstar, inputs) is
            greater than this tolerance. Default: 1e-3.

            dtype: Data type for representing all of the object's data.
            Default: numpy.float32.

            cond_id: [n,] numpy array with cond_id[i] indicating the condition ID corresponding to inputs[i].

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

        # These apply to all fixed points
        # (one value each, rather than one value per fixed point).
        self.tol_unique = tol_unique
        self.dtype = dtype
        self.dtype_complex = dtype_complex
        self.do_alloc_nan = do_alloc_nan
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

            self.eigval_J_xstar = self._alloc_nan(
                (n, n_states), dtype=dtype_complex)
            self.eigvec_J_xstar = self._alloc_nan(
                (n, n_states, n_states), dtype=dtype_complex)

            # not forcing dtype to bool yet, since np.bool(np.nan) is True,
            # which could be misinterpreted as a valid value.
            self.is_stable = self._alloc_nan((n))

            self.cond_id = self._alloc_nan((n))

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
            self.is_stable = is_stable
            self.cond_id = cond_id

        self.assert_valid_shapes()

    def __setitem__(self, index, fps):
        '''Implements the assignment operator.

        All compatible data from fps are copied. This excludes tol_unique,
        dtype, n, n_states, and n_inputs, which retain their original values.

        Usage:
            fps_to_be_partially_overwritten[index] = fps
        '''

        assert isinstance(fps, FixedPoints),\
            ('fps must be a FixedPoints object but was %s.' % type(fps))

        if isinstance(index, int):
            # Force the indexing that follows to preserve numpy array ndim
            index = list(range(index, index+1))

        manual_data_attrs = ['eigval_J_xstar', 'eigvec_J_xstar', 'is_stable']

        # This block added for testing 9/17/20 (replaces commented code below)
        for attr_name in self._data_attrs:
            if attr_name not in manual_data_attrs:
                attr = getattr(self, attr_name)
                if attr is not None:
                    attr[index] = getattr(fps, attr_name)

        ''' Previous version of block above:

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
        '''

        # This manual handling no longer seems necessary, but I'll save that
        # change and testing for a rainy day.
        if self.has_decomposed_jacobians:
            self.eigval_J_xstar[index] = fps.eigval_J_xstar
            self.eigvec_J_xstar[index] = fps.eigvec_J_xstar
            self.is_stable[index] = fps.is_stable

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
            index = list(range(index, index+1))

        kwargs = self._nonspecific_kwargs
        manual_data_attrs = ['eigval_J_xstar', 'eigvec_J_xstar', 'is_stable']

        for attr_name in self._data_attrs:

            attr_val = getattr(self, attr_name)

            # This manual handling no longer seems necessary, but I'll save
            # that change and testing for a rainy day.
            if attr_name in manual_data_attrs:
                if self.has_decomposed_jacobians:
                    indexed_val = self._safe_index(attr_val, index)
                else:
                    indexed_val = None
            else:
                indexed_val = self._safe_index(attr_val, index)

            kwargs[attr_name] = indexed_val

        indexed_fps = FixedPoints(**kwargs)

        return indexed_fps

    def __len__(self):
        '''Returns the number of fixed points stored in the object.'''
        return self.n

    def __contains__(self, fp):
        '''Checks whether a specified fixed point is contained in the object.

        Args:
            fp: A FixedPoints object containing exactly one fixed point.

        Returns:
            bool indicating whether any fixed point matches fp.
        '''

        idx = self.find(fp)

        return idx.size > 0

    def get_unique(self):
        '''Identifies unique fixed points. Among duplicates identified,
        this keeps the one with smallest qstar.

        Args:
            None.

        Returns:
            A FixedPoints object containing only the unique fixed points and
            their associated data. Uniqueness is determined down to tol_unique.
        '''
        assert (self.xstar is not None),\
            ('Cannot find unique fixed points because self.xstar is None.')

        if self.inputs is None:
            data_nxd = self.xstar
        else:
            data_nxd = np.concatenate((self.xstar, self.inputs), axis=1)

        idx_keep = []
        idx_checked = np.zeros(self.n, dtype=bool)
        for idx in range(self.n):

            if idx_checked[idx]:
                # If this FP matched others, we've already determined which
                # of those matching FPs to keep. Repeating would simply
                # identify the same FP to keep.
                continue

            # Don't compare against FPs we've already checked
            idx_check = np.where(~idx_checked)[0]
            fps_check = self[idx_check] # only check against these FPs
            idx_idx_check = fps_check.find(self[idx]) # indexes into fps_check
            idx_match = idx_check[idx_idx_check] # indexes into self

            if len(idx_match)==1:
                # Only matches with itself
                idx_keep.append(idx)
            else:
                qstars_match = self.qstar[idx_match]
                idx_candidate = idx_match[np.argmin(qstars_match)]
                idx_keep.append(idx_candidate)
                idx_checked[idx_match] = True

        return self[idx_keep]

    def transform(self, U, offset=0.):
        ''' Apply an affine transformation to the state-space representation.
        This may be helpful for plotting fixed points in a given linear
        subspace (e.g., PCA or an RNN readout space).


        Args:
            U: shape (n_states, k) numpy array projection matrix.

            offset (optional): shape (k,) numpy translation vector. Default: 0.

        Returns:
            A FixedPoints object.
        '''
        kwargs = self.kwargs

        # These are all transformed. All others are not.
        for attr_name in ['xstar', 'x_init', 'F_xstar']:
            kwargs[attr_name] = np.matmul(getattr(self, attr_name), U) + offset

        if self.has_decomposed_jacobians:
            kwargs['eigval_J_xstar'] = self.eigval_J_xstar
            kwargs['eigvec_J_xstar'] = \
                np.matmul(U.T, self.eigvec_J_xstar) + offset

        transformed_fps = FixedPoints(**kwargs)

        return transformed_fps

    def find(self, fp):
        '''Searches in the current FixedPoints object for matches to a
        specified fixed point. Two fixed points are defined as matching
        if the 2-norm of the difference between their concatenated (xstar,
        inputs) is within tol_unique).

        Args:
            fp: A FixedPoints object containing exactly one fixed point.

        Returns:
            shape (n_matches,) numpy array specifying indices into the current
            FixedPoints object where matches to fp were found.
        '''

        # If not found or comparison is impossible (due to type or shape),
        # follow convention of np.where and return an empty numpy array.
        result = np.array([], dtype=int)

        if isinstance(fp, FixedPoints):
            if fp.n_states == self.n_states and fp.n_inputs == self.n_inputs:

                if self.inputs is None:
                    self_data_nxd = self.xstar
                    arg_data_nxd = fp.xstar
                else:
                    self_data_nxd = np.concatenate(
                        (self.xstar, self.inputs), axis=1)
                    arg_data_nxd = np.concatenate(
                        (fp.xstar, fp.inputs), axis=1)

                norm_diffs_n = np.linalg.norm(
                    self_data_nxd - arg_data_nxd, axis=1)

                result = np.where(norm_diffs_n <= self.tol_unique)[0]

        return result

    def update(self, new_fps):
        ''' Combines the entries from another FixedPoints object into this
        object.

        Args:
            new_fps: a FixedPoints object containing the entries to be
            incorporated into this FixedPoints object.

        Returns:
            None

        Raises:
            AssertionError if the non-fixed-point specific attributes of
            new_fps do not match those of this FixedPoints object.

            AssertionError if any data attributes are found in one but not both
            FixedPoints objects (especially relevant for decomposed Jacobians).

            AssertionError if the updated object has inconsistent data shapes.
        '''

        self._assert_matching_nonspecific_attrs(self, new_fps)

        for attr_name in self._data_attrs:

            this_has = hasattr(self, attr_name)
            that_has = hasattr(new_fps, attr_name)

            assert this_has == that_has,\
                ('One but not both FixedPoints objects have %s. '
                 'FixedPoints.update does not currently support this '
                 'configuration.' % attr_name)

            if this_has and that_has:
                cat_attr = np.concatenate(
                    (getattr(self, attr_name),
                    getattr(new_fps, attr_name)),
                    axis=0)
                setattr(self, attr_name, cat_attr)

        self.n = self.n + new_fps.n
        self.assert_valid_shapes()

    def decompose_jacobians(self, do_batch=True, str_prefix=''):
        '''Adds the following fields to the FixedPoints object:

        eigval_J_xstar: [n x n_states] numpy array with eigval_J_xstar[i, :]
        containing the eigenvalues of J_xstar[i, :, :].

        eigvec_J_xstar: [n x n_states x n_states] numpy array containing with
        eigvec_J_xstar[i, :, :] containing the eigenvectors of
        J_xstar[i, :, :].

        Args:
            do_batch (optional): bool indicating whether to perform a batch
            decomposition. This is typically faster as long as sufficient
            memory is available. If False, decompositions are performed
            one-at-a-time, sequentially, which may be necessary if the batch
            computation requires more memory than is available. Default: True.

            str_prefix (optional): String to be pre-pended to print statements.

        Returns:
            None.
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
                    (n, n_states), dtype=self.dtype_complex)
                e_vecs_unsrt = self._alloc_nan(
                    (n, n_states, n_states), dtype=dtype_complex)

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
        # For each FP, sort eigenvectors by eigenvalue magnitude
        # (decreasing order).
        mags_unsrt = np.abs(e_vals_unsrt) # shape (n,)
        sort_idx = np.argsort(mags_unsrt)[:,::-1]

        # Apply the sort
        # There must be a faster way, but I'm too lazy to find it at the moment
        self.eigval_J_xstar = \
            self._alloc_nan((n, n_states), dtype=self.dtype_complex)
        self.eigvec_J_xstar = \
            self._alloc_nan((n, n_states, n_states), dtype=self.dtype_complex)
        self.is_stable = np.zeros(n, dtype=bool)

        for k in range(n):
            sort_idx_k = sort_idx[k]
            e_vals_k = e_vals_unsrt[k][sort_idx_k]
            e_vecs_k = e_vecs_unsrt[k][:, sort_idx_k]
            self.eigval_J_xstar[k] = e_vals_k
            self.eigvec_J_xstar[k] = e_vecs_k

            # For stability, need only to look at the leading eigenvalue
            self.is_stable[k] = np.abs(e_vals_k[0]) < 1.0

        self.assert_valid_shapes()

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

        self.assert_valid_shapes()

        file = open(save_path,'wb')
        file.write(pickle.dumps(self.__dict__))
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
        file = open(restore_path,'rb')
        restore_data = file.read()
        file.close()
        self.__dict__ = pickle.loads(restore_data)

        # Hacks to bridge between different versions of saved data
        if not hasattr(self, 'do_alloc_nan'):
            self.do_alloc_nan = False

        if not hasattr(self, 'eigval_J_xstar'):
            n = self.n
            n_states = self.n_states
            dtype_complex = np.complex64
            self.eigval_J_xstar = self._alloc_nan(
                (n, n_states), dtype=dtype_complex)
            self.eigvec_J_xstar = self._alloc_nan(
                (n, n_states, n_states), dtype=dtype_complex)

            self.is_stable = self._alloc_nan((n))

            self.cond_id = self._alloc_nan((n))

        self.assert_valid_shapes()

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

    def print_shapes(self):
        ''' Prints the shapes of the data attributes of the fixed points.

        Args:
            None.

        Returns:
            None.
        '''

        for attr_name in FixedPoints._data_attrs:
            attr = getattr(self, attr_name)
            print('%s: %s' % (attr_name, str(attr.shape)))

    def assert_valid_shapes(self):
        ''' Checks that all data attributes reflect the same number of fixed
        points.

        Raises:
            AssertionError if any non-None data attribute does not have
            .shape[0] as self.n.
        '''
        n = self.n
        for attr_name in FixedPoints._data_attrs:
            data = getattr(self, attr_name)
            if data is not None:
                assert data.shape[0] == self.n,\
                    ('Detected %d fixed points, but %s.shape is %s '
                    '(shape[0] should be %d' %
                    (n, attr_name, str(data.shape), n))

    @staticmethod
    def concatenate(fps_seq):
        ''' Join a sequence of FixedPoints objects.

        Args:
            fps_seq: sequence of FixedPoints objects. All FixedPoints objects
            must have the following attributes in common:
                n_states
                n_inputs
                has_decomposed_jacobians

        Returns:
            A FixedPoints objects containing the concatenated FixedPoints data.
        '''

        assert len(fps_seq) > 0, 'Cannot concatenate empty list.'
        FixedPoints._assert_matching_nonspecific_attrs(fps_seq)

        kwargs = {}

        for attr_name in FixedPoints._nonspecific_attrs:
            kwargs[attr_name] = getattr(fps_seq[0], attr_name)

        for attr_name in FixedPoints._data_attrs:
            if all((hasattr(fps, attr_name) for fps in fps_seq)):

                cat_list = [getattr(fps, attr_name) for fps in fps_seq]

                if all([l is None for l in cat_list]):
                    cat_attr = None
                elif any([l is None for l in cat_list]):
                    # E.g., attempting to concat cond_id when it exists for
                    # some fps but not for others. Better handling of this
                    # would be nice. And yes, this would catch the all above,
                    # but I'm keeping these cases separate to facilitate an
                    # eventual refinement.
                    cat_attr = None
                else:
                    cat_attr = np.concatenate(cat_list, axis=0)

                kwargs[attr_name] = cat_attr

        return FixedPoints(**kwargs)

    @property
    def is_single_fixed_point(self):
        return self.n == 1

    @property
    def has_decomposed_jacobians(self):

        if not hasattr(self, 'eigval_J_xstar'):
            return False

        return self.eigval_J_xstar is not None

    @property
    def kwargs(self):
        ''' Returns dict of keyword arguments necessary for reinstantiating a
        (shallow) copy of this FixedPoints object, i.e.,

        fp_copy  = FixedPoints(**fp.kwargs)
        '''

        kwargs = self._nonspecific_kwargs

        for attr_name in self._data_attrs:
            kwargs[attr_name] = getattr(self, attr_name)

        return kwargs

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

    @staticmethod
    def _assert_matching_nonspecific_attrs(fps_seq):

        for attr_name in FixedPoints._nonspecific_attrs:
            items = [getattr(fps, attr_name) for fps in fps_seq]
            for item in items:
                assert item == items[0],\
                    ('Cannot concatenate FixedPoints because of mismatched %s '
                     '(%s is not %s)' %
                     (attr_name, str(items[0]), str(item)))

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

    @property
    def _nonspecific_kwargs(self):
        # These are not specific to individual fixed points.
        # Thus, simple copy, no indexing required
        return {
            'dtype': self.dtype,
            'tol_unique': self.tol_unique
            }
