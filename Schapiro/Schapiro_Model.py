#!/usr/bin/env python

# coding: utf-8

# In[1]:

import math
import multiprocessing as mp
import pickle
import random
import sys
import time
from typing import Dict
from typing import Tuple
from typing import List

import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import torch
torch.set_printoptions(precision = 1)

import leabra7 as lb

#get_ipython().run_line_magic('matplotlib', 'inline')

num_networks = int(sys.argv[1])
num_trials = int(sys.argv[2])


# In[2]:


ThetaTrough = lb.Phase("theta_trough", "minus")
ThetaPeak = lb.Phase("theta_peak", "minus")
ThetaPlus = lb.Phase("theta_plus", "plus")
TestPhase = lb.Phase("test", "minus")


# In[3]:


## Define Layer Specs

# Generic Layer Spec
layer_spec = lb.LayerSpec()

# EC Layer Spec
EC_layer_spec = lb.LayerSpec(
    inhibition_type = "kwta",
    kwta_pct = 0.25,
    kwta_pt = 0.5,
)

# DG Layer Spec
DG_layer_spec = lb.LayerSpec(
    inhibition_type = "kwta_avg",
    kwta_pct = 0.01,
    kwta_pt = 0.9,
)

# CA3 Layer Spec
CA3_layer_spec = lb.LayerSpec(
    inhibition_type = "kwta_avg",
    kwta_pct = 0.06,
    kwta_pt = 0.7,
)

# CA1 Layer Spec
CA1_layer_spec = lb.LayerSpec(
    inhibition_type = "kwta_avg",
    kwta_pct = 0.25,
    kwta_pt = 0.7,
)


# In[4]:


## Define Projections Spec

# Input -> EC_in
Input_projn_spec = lb.ProjnSpec(
    dist=lb.Uniform(low=0.25, high=0.75),
    lrate = 0,
    projn_type = "one_to_one",
    minus_phase = lb.NonePhase,
    plus_phase = ThetaPlus,
)

# EC_in -> DG
# EC_in -> CA3
EC_in_projn_spec = lb.ProjnSpec(
    dist=lb.Uniform(low=0.25, high=0.75),
    lrate = 0.2,
    sparsity = 0.25,
    minus_phase = lb.NonePhase,
    plus_phase = ThetaPlus,
)

# DG -> CA3 (Mossy Fiber)
DG_CA3_projn_spec = lb.ProjnSpec(
    dist=lb.Uniform(low=0.89, high=0.91),
    wt_scale_rel = 8.0,
    sparsity = 0.05,
    lrate = 0,
    minus_phase = lb.NonePhase,
    plus_phase = ThetaPlus,
)

# CA3 -> CA3
CA3_CA3_projn_spec = lb.ProjnSpec(
    dist=lb.Uniform(low=0.25, high=0.75),
    lrate = 0.2,
    minus_phase = lb.NonePhase,
    plus_phase = ThetaPlus,
)

# CA3 -> CA1 (Schaffer)
CA3_CA1_projn_spec = lb.ProjnSpec(
    dist=lb.Uniform(low=0.25, high=0.75),
    lrate = 0.05,
    minus_phase = ThetaPeak,
    plus_phase = ThetaPlus,
)

# EC_in -> CA1 Projection Spec
EC_in_CA1_projn_spec = lb.ProjnSpec(
    dist=lb.Uniform(low=0.25, high=0.75),
    wt_scale_abs = 3.0,
    lrate = 0.02,
    minus_phase = ThetaTrough,
    plus_phase = ThetaPlus,
)

# EC_out <--> CA1 Projection Spec
EC_out_CA1_projn_spec = lb.ProjnSpec(
    dist=lb.Uniform(low=0.25, high=0.75),
    lrate = 0.02,
    minus_phase = ThetaTrough,
    plus_phase = ThetaPlus,
)

# EC_out -> EC_in Projection Spec
EC_out_EC_in_projn_spec = lb.ProjnSpec(
    dist=lb.Uniform(low=0.49, high=0.51),
    lrate = 0,
    wt_scale_abs = 2.0,
    wt_scale_rel = 0.5,
    projn_type = "one_to_one",
    minus_phase = lb.NonePhase,
    plus_phase = ThetaPlus,
)


# In[5]:


def gen_net() -> lb.Net:
    # Create the Network
    net = lb.Net()

    ## Create Layers

    # Create input and output layers
    net.new_layer("Input", 8, layer_spec)
    net.new_layer("EC_in", 8, EC_layer_spec)
    net.new_layer("EC_out", 8, EC_layer_spec)

    # Create cortical layers
    net.new_layer("CA1", 100, CA1_layer_spec)
    net.new_layer("CA3", 80, CA3_layer_spec)

    # Create hippocampus
    net.new_layer("DG", 400, DG_layer_spec)

    ## Create Projections

    # Input Feed
    net.new_projn("Input: Input -> EC_in", "Input", "EC_in", spec=Input_projn_spec)
    net.new_projn("Loop: EC_out -> EC_in", "EC_out", "EC_in", spec=EC_out_EC_in_projn_spec)

    # Create MSP
    net.new_projn("MSP: EC_in -> CA1", "EC_in", "CA1", spec=EC_in_CA1_projn_spec)
    net.new_projn("MSP: CA1 -> EC_out", "CA1", "EC_out", spec=EC_out_CA1_projn_spec)
    net.new_projn("MSP: EC_out -> CA1", "EC_out", "CA1", spec=EC_out_CA1_projn_spec)

    # Create TSP
    net.new_projn("TSP: EC_in -> DG", "EC_in", "DG", spec=EC_in_projn_spec)
    net.new_projn("TSP: EC_in -> CA3", "EC_in", "CA3", spec=EC_in_projn_spec)
    net.new_projn("TSP: DG -> CA3", "DG", "CA3", spec=DG_CA3_projn_spec)
    net.new_projn("TSP: CA3 -> CA3", "CA3", "CA3", spec=CA3_CA3_projn_spec)
    net.new_projn("TSP: CA3 -> CA1", "CA3", "CA1", spec=CA3_CA1_projn_spec)

    return net


# In[6]:


# Function to plot data for a certain attribute for each unit of layer
def plot_by_unit(axes: List[matplotlib.axes.Axes],
                 log: pd.DataFrame, attr: str, title: str, location: List) -> None:
    for name, group in log.groupby("unit"):
        group.plot(x="time", y=attr, ax=axes[location],
                   title = title, label="unit " + str(name))


# ## Sequence Learning

# In[7]:


sequence_dict = dict()
sequence_dict[0] = [1]
sequence_dict[2] = [3]
sequence_dict[4] = [5]
sequence_dict[6] = [7]

sequence_dict[1] = [2, 4, 6]
sequence_dict[3] = [0, 4, 6]
sequence_dict[5] = [0, 2, 6]
sequence_dict[7] = [0, 2, 4]

def seq_next(prev: int) -> int:
    return random.choice(sequence_dict[prev])

def tensorfy(old_num: int, new_num: int) -> torch.Tensor:
    x = lb.to_cuda(torch.FloatTensor(8).zero_())
    x[new_num] = 1
    x[old_num] = 0.9
    return x

def gen_train_seq_epoch(epoch_len) -> List[torch.Tensor]:
    patterns: List[torch.Tensor] = []

    old_seed = random.randint(0,7)
    curr_seed = seq_next(old_seed)

    for _ in range(epoch_len):
        patterns += [tensorfy(old_seed, curr_seed)]

        old_seed = curr_seed
        curr_seed = seq_next(old_seed)

    return patterns

def gen_train_sep_epoch(epoch_len) -> List[torch.Tensor]:
    patterns: List[torch.Tensor] = []

    for _ in range(epoch_len):
        old_seed = random.choice([0, 2, 4, 6])
        curr_seed = seq_next(old_seed)

        patterns += [tensorfy(old_seed, curr_seed)]

    return patterns

def gen_test() -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    inputs: List[torch.Tensor] = []
    outputs: List[torch.Tensor] = []

    for curr_seed in range(8):
        if curr_seed%2 == 0:
            next_seed = curr_seed + 1
        else:
            next_seed = curr_seed - 1

        inputs += [tensorfy(None, curr_seed)]
        outputs += [tensorfy(curr_seed, next_seed)]

    return inputs, outputs


# ## Run Network

# In[8]:


def net_snapshot(network: lb.Net) -> Dict[str, torch.Tensor]:
    act_dict: Dict[str, torch.Tensor] = dict()

    for name, layer in network.layers.items():
        act_dict[name] = layer.units.act

    return act_dict


# In[9]:


def pearson_correlation(patterns: Dict[int, torch.Tensor]) -> torch.Tensor:
    dim = len(patterns.keys())
    corr_matrix = torch.FloatTensor(dim, dim).zero_()

    for i in range(dim):
        for j in range(i, dim):
            corr = float(pearsonr(patterns[i], patterns[j])[0])
            if math.isnan(corr):
                corr = 0.0
            corr_matrix[i, j] = corr_matrix[j, i] = corr

    return corr_matrix


# In[10]:


def learn_trial(network: lb.Net, pattern = torch.Tensor) -> None:
    # Theta Trough
    network.clamp_layer("Input", pattern)
    network.inhibit_projns("TSP: CA3 -> CA1", "Loop: EC_out -> EC_in")
    network.phase_cycle(ThetaTrough, num_cycles = 20)
    network.uninhibit_projns("TSP: CA3 -> CA1", "Loop: EC_out -> EC_in")

    # Theta Peak
    network.inhibit_projns("MSP: EC_in -> CA1", "Loop: EC_out -> EC_in")
    network.phase_cycle(ThetaPeak, num_cycles = 20)
    network.uninhibit_projns("MSP: EC_in -> CA1", "Loop: EC_out -> EC_in")
    # Theta Plus
    network.clamp_layer("EC_out", pattern)
    network.inhibit_projns("TSP: CA3 -> CA1")
    network.phase_cycle(ThetaPlus, num_cycles = 60)
    network.inhibit_projns("TSP: CA3 -> CA1")
    # Reset
    network.unclamp_layer("Input", "EC_out")
    network.end_trial()
    network.learn()


# In[28]:


def test_trial(network: lb.Net, input_pattern = torch.Tensor, output_pattern = torch.Tensor) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], bool, bool]:
    # Initial Response
    network.clamp_layer("Input", input_pattern)
    network.phase_cycle(TestPhase, num_cycles = 20)

    initial_response = net_snapshot(network)
#    print(initial_response["EC_out"])
    initial_guess = ((initial_response["EC_out"] > 0.5) == output_pattern.byte()).all()

    # Final Response
    network.phase_cycle(TestPhase, num_cycles = 60)

    final_response = net_snapshot(network)
    final_guess = ((final_response["EC_out"] > 0.5) == output_pattern.byte()).all()

    # Reset
    network.end_trial()

    return initial_response, final_response, initial_guess, final_guess


# In[12]:


def train_epoch(network: lb.Net, seq: bool, num_trial) -> None:
    startTime = time.time()
    if seq:
        epoch_pattern = gen_train_seq_epoch(num_trial)
    else:
        epoch_pattern = gen_train_sep_epoch(num_trial)

    for t in range(num_trial):
        learn_trial(network, epoch_pattern[t])

    print("epoch done " + str(time.time() - startTime))
    sys.stdout.flush()


# In[23]:


def test_epoch(network: lb.Net) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], float, float]:
    epoch_input, epoch_output = gen_test()

    initial_correct = 0
    final_correct = 0

    hidden_layers: Set[str] = {"DG", "CA3", "CA1"}

    initial_acts: Dict[str, Dict[int, torch.Tensor]] = {"DG": dict(), "CA3": dict(), "CA1": dict()}
    final_acts: Dict[str, Dict[int, torch.Tensor]] = {"DG": dict(), "CA3": dict(), "CA1": dict()}

    for t in range(len(epoch_input)):
        initial_response, final_response, initial_guess, final_guess = test_trial(network, epoch_input[t], epoch_output[t])

 #       print(initial_guess)

        for layer in hidden_layers:
            initial_acts[layer][t] = initial_response[layer]
            final_acts[layer][t] = final_response[layer]

        initial_correct += initial_guess
        final_correct += final_guess

    initial_matrix: Dict[str, torch.Tensor] = dict()
    final_matrix: Dict[str, torch.Tensor] = dict()

    for layer in hidden_layers:
        initial_matrix[layer] = pearson_correlation(initial_acts[layer])
        final_matrix[layer] = pearson_correlation(final_acts[layer])

    initial_accuracy = float(initial_correct) / len(epoch_input)
    final_accuracy = float(final_correct) / len(epoch_input)

  #  print("network done")

    return initial_matrix, final_matrix, initial_accuracy, final_accuracy



# In[19]:


def assemble(name: int, seq: bool, num_trials: int):
    net = gen_net()
    for e in range(10):
        train_epoch(net, seq, num_trials)
    print("begin assembly: "+str(name))
    sys.stdout.flush()
    result = test_epoch(net)
    print("assembly complete: "+str(name))
    sys.stdout.flush()
    if seq:
        pickle.dump(result, open("/tigress/noamm/schapiro/cpu/nets_seq_"+str(num_networks)+"_"+str(num_trials)+"_"+str(name)+".pkl", "wb"))
    else:
        pickle.dump(result, open("/tigress/noamm/schapiro/cpu/nets_sep_"+str(num_networks)+"_"+str(num_trials)+"_"+str(name)+".pkl", "wb"))
    print("dumped: "+str(name))
    sys.stdout.flush()

q_sep = mp.JoinableQueue()
q_seq = mp.JoinableQueue()

process_sep = [mp.Process(target = assemble, args = (i, False, num_trials)) for i in range(num_networks)]
process_seq = [mp.Process(target = assemble, args = (i, True, num_trials)) for i in range(num_networks)]

print("sep")
for process in process_sep:
    process.start()

print("seq")
for process in process_seq:
    process.start()

for process in process_sep:
    process.join()

for process in process_seq:
    process.join()

print("all done")
