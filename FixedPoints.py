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

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pdb
import numpy as np
import tf_utils

class FixedPoints(object):
    '''
    A class for storing fixed points and associated data.
    '''

    def __init__(self,
                 xstar=None,
                 x_init=None,
                 inputs=None,
                 F_xstar=None,
                 qstar=None,
                 dq=None,
                 n_iters=None,
                 J_xstar=None,
                 do_alloc_nan=False,
                 n=None,
                 n_states=None,
                 n_inputs=None,
                 tol_unique=1e-3,
                 dtype=np.float32):
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

            do_alloc_nan: Bool indicating whether to initialize all data attributes (all optional args above) as NaN-filled numpy arrays.
            If True, n, n_states and n_inputs must be provided. These values
            are otherwise ignored:

                n: Positive int specifying the number of fixed points to
                allocate space for.

                n_states: Positive int specifying the dimensionality of the
                network state (a.k.a. the number of hidden units).

                n_inputs: Positive int specifying the dimensionality of the
                network inputs.

            tol_unique: Positive scalar specifying the numerical
            precision required to label two fixed points as being unique from
            one another. Two fixed points will be considered unique if they
            differ by this amount (or more) along any dimension. This
            tolerance is used to discard numerically similar fixed points.
            Default: 1e-3.

            dtype: Data type for representing all of the object's data.
            Default: numpy.float32.

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

        if do_alloc_nan:
            if n is None:
                raise ValueError(       'n must be provided if '
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

    def _alloc_nan(self, shape):
        '''Returns a nan-filled numpy array.

        Args:
            shape: int or tuple representing the shape of the desired numpy
            array.

        Returns:
            numpy array with the desired shape, filled with NaNs.

        '''
        result = np.zeros(shape, dtype=self.dtype)
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

        unique_xstar, idx = unique_rows(self.xstar, self.tol_unique)

        return self[idx]

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

    def plot(self,
        state_traj=None,
        plot_batch_idx=None,
        plot_start_time=0,
        plot_stop_time=None,
        mode_scale=0.25):

        '''Plots a visualization and analysis of the unique fixed points.

        1) Finds a low-dimensional subspace for visualization via PCA. If
        state_traj is provided, PCA is fit to [all of] those RNN state
        trajectories. Otherwise, PCA is fit to the identified unique fixed
        points. This subspace is 3-dimensional if the RNN state dimensionality
        is >= 3.

        2) Plots the PCA representation of the stable unique fixed points as
        black dots.

        3) Plots the PCA representation of the unstable unique fixed points as
        red dots.

        4) Plots the PCA representation of the modes of the Jacobian at each
        fixed point. By default, only unstable modes are plotted.

        5) (optional) Plots example RNN state trajectories as blue lines.

        Args:
            state_traj (optional): [n_batch x n_time x n_states] numpy
            array or LSTMStateTuple with .c and .h as
            [n_batch x n_time x n_states/2] numpy arrays. Contains example
            trials of RNN state trajectories.

            plot_batch_idx (optional): Indices specifying which trials in
            state_traj to plot on top of the fixed points. Default: plot all
            trials.

            plot_start_time (optional): int specifying the first timestep to
            plot in the example trials of state_traj. Default: 0.

            plot_stop_time (optional): int specifying the last timestep to
            plot in the example trials of stat_traj. Default: n_time.

            stop_time (optional):

            mode_scale (optional): Non-negative float specifying the scaling
            of the plotted eigenmodes. A value of 1.0 results in each mode
            plotted as a set of diametrically opposed line segments
            originating at a fixed point, with each segment's length specified
            by the magnitude of the corresponding eigenvalue.

        Returns:
            None.
        '''

        def plot_123d(ax, z, **kwargs):
            '''Plots in 1D, 2D, or 3D.

            Args:
                ax: Matplotlib figure axis on which to plot everything.

                z: [n x n_states] numpy array containing data to be plotted,
                where n_states is 1, 2, or 3.

                any keyword arguments that can be passed to ax.plot(...).

            Returns:
                None.
            '''
            n_states = z.shape[1]
            if n_states ==3:
                ax.plot(z[:, 0], z[:, 1], z[:, 2], **kwargs)
            elif n_states == 2:
                ax.plot(z[:, 0], z[:, 1], **kwargs)
            elif n_states == 1:
                ax.plot(z, **kwargs)

        def plot_fixed_point(ax, xstar, J, pca,
            scale=1.0, max_n_modes=3, do_plot_stable_modes=False):
            '''Plots a single fixed point and its dominant eigenmodes.

            Args:
                ax: Matplotlib figure axis on which to plot everything.

                xstar: [1 x n_states] numpy array representing the fixed point
                to be plotted.

                J: [n_states x n_states] numpy array containing the Jacobian of the
                RNN transition function at fixed point xstar.

                pca: PCA object as returned by sklearn.decomposition.PCA. This
                is used to transform the high-d state space representations
                into 3-d for visualization.

                scale (optional): Scale factor for stretching (>1) or shrinking
                (<1) lines representing eigenmodes of the Jacobian. Default:
                1.0 (unity).

                max_n_modes (optional): Maximum number of eigenmodes to plot.
                Default: 3.

                do_plot_stable_modes (optional): bool indicating whether or
                not to plot lines representing stable modes (i.e.,
                eigenvectors of the Jacobian whose eigenvalue magnitude is
                less than one).

            Returns:
                None.
            '''
            n_states = xstar.shape[1]
            e_vals, e_vecs = np.linalg.eig(J)
            sorted_e_val_idx = np.argsort(np.abs(e_vals))

            if max_n_modes > len(e_vals):
                max_n_modes = e_vals

            for mode_idx in range(max_n_modes):

                # -[1, 2, ..., max_n_modes]
                idx = sorted_e_val_idx[-(mode_idx+1)]

                # Magnitude of complex eigenvalue
                e_val_mag = np.abs(e_vals[idx])

                if e_val_mag > 1.0 or do_plot_stable_modes:

                    # Already real. Cast to avoid warning.
                    e_vec = np.real(e_vecs[:,idx])

                    # [1 x d] numpy arrays
                    xstar_plus = xstar + scale*e_val_mag*e_vec
                    xstar_minus = xstar - scale*e_val_mag*e_vec

                    # [3 x d] numpy array
                    xstar_mode = np.vstack((xstar_minus, xstar, xstar_plus))

                    if e_val_mag < 1.0:
                        color = 'k'
                    else:
                        color = 'r'

                    if n_states >= 3:
                        # [3 x 3] numpy array
                        zstar_mode = pca.transform(xstar_mode)
                    else:
                        zstar_mode = x_star_mode

                    plot_123d(ax, zstar_mode, color=color)

            is_stable = all(np.abs(e_vals) < 1.0)
            if is_stable:
                color = 'k'
            else:
                color = 'r'

            if n_states >= 3:
                zstar = pca.transform(xstar)
            else:
                zstar = xstar

            plot_123d(
                ax, zstar, color=color, marker='.', markersize=12)

        FIG_WIDTH = 6 # inches
        FIG_HEIGHT = 6 # inches
        FONT_WEIGHT = 'bold'

        xstar = self.xstar
        J_xstar = self.J_xstar

        if state_traj is not None:
            if tf_utils.is_lstm(state_traj):
                state_traj_bxtxd = tf_utils.convert_from_LSTMStateTuple(
                    state_traj)
            else:
                state_traj_bxtxd = state_traj

            [n_batch, n_time, n_states] = state_traj_bxtxd.shape

            # Ensure plot_start_time >= 0
            plot_start_time = np.max([plot_start_time, 0])

            if plot_stop_time is None:
                plot_stop_time = n_time
            else:
                # Ensure plot_stop_time <= n_time
                plot_stop_time = np.min([plot_stop_time, n_time])

            plot_time_idx = range(plot_start_time, plot_stop_time)

        n_inits, n_states = np.shape(xstar)

        fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT),
                         tight_layout=True)
        if n_states >= 3:
            pca = PCA(n_components=3)

            if state_traj is not None:
                state_traj_btxd = np.reshape(state_traj_bxtxd,
                    (n_batch*n_time, n_states))
                pca.fit(state_traj_btxd)
            else:
                pca.fit(xstar)

            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel('PC 1', fontweight=FONT_WEIGHT)
            ax.set_zlabel('PC 3', fontweight=FONT_WEIGHT)
            ax.set_ylabel('PC 2', fontweight=FONT_WEIGHT)

            # For generating figure in paper.md
            ax.set_xticks([-2, -1, 0, 1, 2])
            ax.set_yticks([-1, 0, 1])
            ax.set_zticks([-1, 0, 1])
        else:
            # For 1D or 0D networks (i.e., never)
            pca = None
            ax = fig.add_subplot(111)
            ax.xlabel('Hidden 1', fontweight=FONT_WEIGHT)
            if n_states == 2:
                ax.ylabel('Hidden 2', fontweight=FONT_WEIGHT)

        if state_traj is not None:
            if plot_batch_idx is None:
                plot_batch_idx = range(n_batch)

            for batch_idx in plot_batch_idx:
                x_idx = state_traj_bxtxd[batch_idx]

                if n_states >= 3:
                    z_idx = pca.transform(x_idx[plot_time_idx, :])
                else:
                    z_idx = x_idx[plot_time_idx, :]
                plot_123d(ax, z_idx, color='b', linewidth=0.2)

        for init_idx in range(n_inits):
            plot_fixed_point(
                ax,
                xstar[init_idx:(init_idx+1)],
                J_xstar[init_idx],
                pca,
                scale=mode_scale)

        plt.ion()
        plt.show()
        plt.pause(1e-10)
