# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: cache-notebooks//ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import os
import time
import json, re

import matplotlib.pyplot as plt
import numpy as np

from IPython.display import display, HTML, clear_output
# -

# %load_ext autoreload
# %autoreload 2

import arc_mdda

#data_train = arc_mdda.load_json_data('./ARC-Challenge/ARC-800-tasks/training/')
data_train = arc_mdda.load_json_data('./External/ARC-AGI/data/training/')
len(data_train), sorted(data_train.keys())[0::70]

# ### Basic visualisation

task_hash = '00d62c1b'  # 3345333e<< 00d62c1b 27a28665
task = data_train[task_hash]
#task;
arc_mdda.plot_2d_grid(task, plt);

sample_grid = arc_mdda.Grid( np.array( task['train'][0]['input'] ) )
print( sample_grid.to_txt_colours() ) 

# +
#print( sample_grid.to_txt_colour_planes() )
# -

new_grid = arc_mdda.Grid.empty() # background=1
print( new_grid.to_txt_colours() ) 

print( new_grid.to_txt_colour_planes() )  # background=0





# +
from typing import NewType
Color = NewType('Color', int)

def foo(c: Color, d: int):
  return c+d

ONE: Color=1
print( foo(ONE, 2) )
print( foo(ONE, 34.2) )


# -

def bar(d:int):
  return d*d
bar(3.23)

# ### Find number of training examples with particular last digits
# * Is the distribution ~even?

if False:
  digit_last_count=dict()
  for task_hash in data_train.keys():
    d=task_hash[-1]
    if d not in digit_last_count: digit_last_count[d]=0
    digit_last_count[d]+=1
  
  for k in sorted(digit_last_count.keys()):
    print(k,digit_last_count[k])
  # Seems somewhat balanced
# ### Find the distribution of sizes of problems

data_set=data_train # Could be data_eval...


def get_task_size(task, include_test=False): # Just thinking about the I/O grids...
  size_arr=[]
  task_arr = task['train'] 
  if include_test:
    task_arr = task_arr + task['test']
  for ex in task_arr:
    size  = len(ex['input']) * len(ex['input'][0])
    size += len(ex['output']) * len(ex['output'][0])
    size_arr.append(size)
  #print(size_arr)
  return sum(size_arr), max(size_arr)
  #size_total, size_max
get_task_size(data_set[task_hash])

# +
size_all=[]
for task_hash in sorted(data_set.keys()):
  task_size = get_task_size(data_set[task_hash])
  size_all.append( task_size )

size_all = np.array(size_all, dtype=np.float32)
# -

plt.scatter(size_all[:,0], size_all[:,1]);
plt.xlabel('total task size');
plt.ylabel('maximum training example size');

token_limit=2000 # Allows for all of the biggest tasks (just I/O though)
np.sum( size_all[:,0]<token_limit), np.sum( size_all[:,1]<token_limit), 
# Number with all task I/O < token_limit, Number with single biggest task I/O < token_limit



