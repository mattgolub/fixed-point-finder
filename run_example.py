import warnings
warnings.filterwarnings("ignore")
import importlib
import FixedPointFinder as fpf
import sys
from RecurrentWhisperer import RecurrentWhisperer
from example.FlipFlop import FlipFlop
sys.path.insert(0, '/Users/xander/lfads_test/fixed-point-finder/example')
from example.run_FlipFlop import train_FlipFlop, find_fixed_points

def cpu_default_rw_non_hash_hyperparameters():
    return {'name': 'RecurrentWhisperer',
 'verbose': False,
 'log_dir': '/tmp/rnn_logs/',
 'run_script': None,
 'n_folds': None,
 'fold_idx': None,
 'min_loss': None,
 'max_train_time': None,
 'max_n_epochs_without_ltl_improvement': 200,
 'max_n_epochs_without_lvl_improvement': 200,
 'do_batch_predictions': False,
 'do_train_mode_predict_on_train_data': False,
 'do_log_output': False,
 'do_restart_run': False,
 'do_custom_restore': False,
 'do_save_tensorboard_summaries': True,
 'do_save_tensorboard_histograms': True,
 'do_save_tensorboard_images': True,
 'n_epochs_per_seso_update': 100,
 'n_epochs_per_ltl_update': 100,
 'n_epochs_per_lvl_update': 100,
 'n_epochs_per_visualization_update': 100,
 'do_generate_pretraining_visualizations': False,
 'do_save_pretraining_visualizations': False,
 'do_generate_training_visualizations': True,
 'do_save_training_visualizations': True,
 'do_generate_final_visualizations': True,
 'do_save_final_visualizations': True,
 'do_save_seso_ckpt': True,
 'max_seso_ckpt_to_keep': 1,
 'do_save_ltl_ckpt': True,
 'do_save_ltl_train_summary': True,
 'do_save_ltl_train_predictions': True,
 'do_generate_ltl_visualizations': True,
 'do_save_ltl_visualizations': True,
 'max_ltl_ckpt_to_keep': 1,
 'do_save_lvl_ckpt': True,
 'do_save_lvl_train_predictions': True,
 'do_save_lvl_train_summary': True,
 'do_save_lvl_valid_predictions': True,
 'do_save_lvl_valid_summary': True,
 'do_generate_lvl_visualizations': True,
 'do_save_lvl_visualizations': True,
 'max_lvl_ckpt_to_keep': 1,
 'fig_filetype': 'pdf',
 'fig_dpi': 600,
 'do_print_visualizations_timing': False,
 'predictions_filetype': 'npz',
 'summary_filetype': 'npz',
 'device_type': 'cpu',
 'device_id': 0,
 'cpu_device_id': 0,
 'per_process_gpu_memory_fraction': 1.0,
 'disable_gpus': True,
 'allow_gpu_growth': False,
 'allow_soft_placement': True,
 'log_device_placement': False}
RecurrentWhisperer._default_rw_non_hash_hyperparameters = cpu_default_rw_non_hash_hyperparameters

## MAIN from run_flip_flop
train_mode = 3

# Step 1: Train an RNN to solve the N-bit memory task
model, valid_predictions = train_FlipFlop(train_mode)
# STEP 2: Find, analyze, and visualize the fixed points of the trained RNN
find_fixed_points(model, valid_predictions)