import warnings
warnings.filterwarnings("ignore")
import sys
import torch
PATH_TO_FIXED_POINT_FINDER = '../'
sys.path.insert(0, PATH_TO_FIXED_POINT_FINDER)
import FixedPointFinder
import numpy as np
from RecurrentWhisperer import RecurrentWhisperer
from FlipFlop import FlipFlop
from run_FlipFlop import train_FlipFlop, find_fixed_points
from plot_utils import plot_fps
import pickle
import os
import pdb


def line_psth(activations, neuron_ind, color, ax, stderr=True, alpha=.1, label=None):
    activation_mean = np.mean(activations[:,:,neuron_ind],axis=0)
    activation_std = np.std(activations[:,:,neuron_ind],axis=0)

    ax.plot(activation_mean, color=color, label=label)
    if stderr:
        ax.fill_between(np.arange(activations.shape[1]), activation_mean - activation_std, activation_mean + activation_std, alpha=alpha, color=color)
    ax.set_title(f"Neuron {neuron_ind}")

def trial_line_psth(activations, trial_ind, neuron_ind, color, ax, stderr=True, alpha=.1):
    ax.plot(activations[trial_ind,:,neuron_ind], color=color)

def get_preds_and_states(model,model_hidden_units, X, stim_index=0):
    d_activations = []
    preds = []
    if not stim_index:
        stim_index = np.random.choice(len(X))
    with torch.no_grad():
        h_out = torch.zeros(1,1,model_hidden_units)
        for i in range(X.shape[1]):
            pred, h_out = model(X[stim_index:stim_index+1,i:i+1], h_out)
            preds.append(pred.detach().numpy()[0]) # remove unsqueezed dims
            d_activations.append(h_out[0])
    return np.vstack(d_activations), np.vstack(preds), stim_index

def get_multi_trial_preds_and_states(model, num_hidden_units, x, inds=[]):
    all_activations, all_preds = [], []
    if not len(inds): inds = range(len(x))
    for ind in inds:
        if ind > 500:
            break
        activations, predictions, stim_index = get_preds_and_states(model ,num_hidden_units , x, stim_index=ind)
        all_activations.append(activations), all_preds.append(predictions)
    return np.stack(all_activations), np.stack(all_preds)

def imshow_psth(activations, color, ax, stderr=True, alpha=.1):
    activation_mean = np.mean(activations,axis=0)
    ax.imshow(activation_mean.T)


def _get_valid_mask(n_batch, n_time, valid_bxt=None):
   

    if valid_bxt is None:
        valid_bxt = np.ones((n_batch, n_time), dtype=np.bool)
    else:

        assert (valid_bxt.shape[0] == n_batch and
            valid_bxt.shape[1] == n_time),\
            ('valid_bxt.shape should be %s, but is %s'
             % ((n_batch, n_time), valid_bxt.shape))

        if not valid_bxt.dtype == np.bool:
            valid_bxt = valid_bxt.astype(np.bool)

    return valid_bxt

def _add_gaussian_noise(data, noise_scale=0.0):

    # Add IID Gaussian noise
    if noise_scale == 0.0:
        return data # no noise to add
    if noise_scale > 0.0:
        return data + noise_scale * self.rng.randn(*data.shape)
    elif noise_scale < 0.0:
        raise ValueError('noise_scale must be non-negative,'
                         ' but was %f' % noise_scale)

            
def find_fixed_points(model, valid_predictions, inputs):
    ''' Find, analyze, and visualize the fixed points of the trained RNN.

    Args:
        model: FlipFlop object.

            Trained RNN model, as returned by train_FlipFlop().

        valid_predictions: dict.

            Model predictions on validation trials, as returned by
            train_FlipFlop().

    Returns:
        None.
    '''

    '''Initial states are sampled from states observed during realistic
    behavior of the network. Because a well-trained network transitions
    instantaneously from one stable state to another, observed networks states
    spend little if any time near the unstable fixed points. In order to
    identify ALL fixed points, noise must be added to the initial states
    before handing them to the fixed point finder. In this example, the noise
    needed is rather large, which can lead to identifying fixed points well
    outside of the domain of states observed in realistic behavior of the
    network--such fixed points can be safely ignored when interpreting the
    dynamical landscape (but can throw visualizations).'''

    NOISE_SCALE = 0.5 # Standard deviation of noise added to initial states
    N_INITS = 512 #1024 # The number of initial states to provide
#     n_bits = model.hps.data_hps['n_bits']
#     is_lstm = model.hps.rnn_type == 'lstm'

    '''Fixed point finder hyperparameters. See FixedPointFinder.py for detailed
    descriptions of available hyperparameters.'''
    fpf_hps = {}

    # Setup the fixed point finder
    fpf = FixedPointFinder.TorchFixedPointFinder(model.rnn_layer, **fpf_hps)

    # Study the system in the absence of input pulses (e.g., all inputs are 0)
#     inputs = np.zeros([1,n_bits])

    '''Draw random, noise corrupted samples of those state trajectories
    to use as initial states for the fixed point optimizations.'''
    initial_states = fpf.sample_states(valid_predictions['state'],
        n_inits=N_INITS,
        noise_scale=NOISE_SCALE)
    # Run the fixed point finder
    unique_fps, all_fps = fpf.find_fixed_points(initial_states, inputs)

    # Visualize identified fixed points with overlaid RNN state trajectories
    # All visualized in the 3D PCA space fit the the example RNN states.
    plot_fps(unique_fps, valid_predictions['state'],
        plot_batch_idx=list(range(30)),
        plot_start_time=10)

    print('Entering debug mode to allow interaction with objects and figures.')
    print('You should see a figure with:')
    print('\tMany blue lines approximately outlining a cube')
    print('\tStable fixed points (black dots) at corners of the cube')
    print('\tUnstable fixed points (red lines or crosses) '
        'on edges, surfaces and center of the cube')
    print('Enter q to quit.\n')
    pdb.set_trace()



def get_default_hps():
    # See FlipFlop.py for detailed descriptions.
    return    {
            'rnn_type': 'lstm',
            'n_hidden': 16,
            'min_loss': 1e-4,
            'log_dir': './logs/',
            'do_generate_pretraining_visualizations': True,

            'data_hps': {
                'n_batch': 512,
                'n_time': 64,
                'n_bits': 3,
                'p_flip': 0.5
                },

            # Hyperparameters for AdaptiveLearningRate
            'alr_hps': {
                'initial_rate': 1.0,
                'min_rate': 1e-5
                }
            }


class FlipFlopNet(torch.nn.Module):
    def __init__(self, input_dim=1, output_dim=1, hidden_size=32, lr=.001, n_epochs=100, verbosity=1):
        super().__init__()
        self.rnn_layer = torch.nn.RNN(input_size=input_dim,hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.output_layer = torch.nn.Linear(in_features=hidden_size, out_features=output_dim)
        self.n_epochs = n_epochs
        self.loss = []
        self.verbosity = verbosity
        self.lr = lr
    
    def init_params(self):
        for name, param in self.named_parameters(): 
            if 'bias' in name: continue
            torch.nn.init.xavier_uniform_(param.data)
            
    def forward(self,X, h_in=[]):
        if len(h_in): # supply hidden state
            rnn_out, h_activation = self.rnn_layer(X, h_in)
        else: # do not supply
            rnn_out, h_activation = self.rnn_layer(X)
        # does not flatten prediction
        prediction = self.output_layer(rnn_out)
        return prediction, h_activation

    def fit(self,x,y):
        # define/compile/train model
        batch_size = 1000
        self.init_params()
 
        self.train()
        loss_fxn = torch.nn.MSELoss()
        optim =  torch.optim.Adam(self.parameters(),lr=self.lr) 
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=800, gamma=0.7)

        for i in range(self.n_epochs):
            # X is a torch Variable
            optim.zero_grad()
            # predict
            pred, h_out = self(x)
            # carefully cache hidden layer activations
            # loss
            # loss = loss_fxn(y, pred)
            loss = loss_fxn(y, pred)
            # backprop + step model
            loss.backward()
                
            if self.verbosity:
                print(f"epoch {i} loss {loss.item()}")
            self.loss.append(loss.item())
            optim.step()
            scheduler.step()

    def predict(self,X):
        # predict using evidence accumulator
        with torch.no_grad():
            d_pred, h_out = self(X)
        return d_pred
    
    def save(self, path):
        save_path = os.path.join(path,'flip_flop_net_6.pt')
        torch.save(self,save_path)

    def load(self, path):
        load_path = os.path.join(path,'flip_flop_net.pt')
        self = torch.load(path)
        
    def get_loss(self):
        return self.loss

def mse(pred, y):
    return torch.mean((pred - y)**2)

if __name__ == "__main__":
   
    with open('torch_data/flip_flop_data.pkl', 'rb') as f:
        data_dict = pickle.load(f)

    train_data, valid_data = data_dict['train_data'],  data_dict['valid_data']
    
    flip_flop_net = FlipFlopNet(input_dim=3, output_dim=3, hidden_size=8, n_epochs=6000,  lr=.05)
    # # flip_flop_net = FlipFlopNet(input_dim=3, output_dim=3, hidden_size=64, n_epochs=50000,  lr=.001)

    x, y = torch.from_numpy(train_data['inputs']).float(), torch.from_numpy(train_data['output']).float()
    # flip_flop_net.fit(x,y)
    # flip_flop_net.save('./')
    
    train_x = torch.from_numpy(train_data['inputs']).float()
    train_y = torch.from_numpy(train_data['output']).float()

    test_x = torch.from_numpy(valid_data['inputs']).float()
    test_y = torch.from_numpy(valid_data['output']).float()
    flip_flop_net = torch.load('torch_data/flip_flop_net_6.pt')
    predictions = flip_flop_net.predict(test_x)
    print("flip flop net MSE: ", mse(predictions, test_y))
    
    inputs = torch.zeros(size=(train_x.shape))
    # remove sequence dim, seq len = 1 for FP Opt
    inputs = inputs[:,-1:,:]
    ## MAIN from run_flip_flop
    valid_predictions = {}
    state_traj, preds   = get_multi_trial_preds_and_states(flip_flop_net,8, train_x)
    valid_predictions['state'] = state_traj
    valid_predictions['output'] = preds
    # import matplotlib.pyplot as plt
    # import pdb; pdb.set_trace()

    # Step 1: Train an RNN to solve the N-bit memory task
    # STEP 2: Find, analyze, and visualize the fixed points of the trained RNN
    find_fixed_points(flip_flop_net, valid_predictions, inputs)
    