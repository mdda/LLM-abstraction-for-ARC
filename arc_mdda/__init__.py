import os
import json, re

#import ast
#import hashlib
import matplotlib.pyplot as plt
import matplotlib.colors

# For getting the image into HTML base64 format
import base64
from io import BytesIO  

import numpy as np
#import copy
#import traceback
#import tiktoken
#from IPython.display import display, HTML, clear_output
#from io import StringIO
#from collections import defaultdict

def load_json_data(folder):
  json_files = [pos_json for pos_json in os.listdir(folder) if pos_json.endswith('.json')]
  data = {}
  for js in json_files:
    with open(os.path.join(folder, js)) as json_file:
      data[js.replace('.json', '')] = json.load(json_file)
  return data

#                   colors = 
#colours_for_cmap           = ["black", "dodgerblue", "red", "lightgreen", "yellow", "grey", "magenta", "orange", "lightblue", "brown", "black"]
colours_for_cmap           = ["black", "#1E93FF", "#F93C31", "lightgreen", "#FFDC00", "#999999", "#E53AA3", "#FF851B", "#87D8F1", "#921231", "black"]
#colours_for_cmap ARCPRIZE = ["black", "#1E93FF", "#F93C31", "#4FCC30", "#FFDC00", "#999999", "#E53AA3", "#FF851B", "#87D8F1", "#921231", "black"]
colours_for_representation = ["black", "blue",       "red", "green",      "yellow", "grey", "magenta", "orange", "azure",     "brown", "black"]

def plot_2d_grid(task, plt):
  cvals  = np.arange(0, 10+1, 1)
  norm   = plt.Normalize(min(cvals), max(cvals))
  tuples = list(zip(map(norm, cvals), colours_for_cmap))
  # ? https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.ListedColormap.html#matplotlib.colors.ListedColormap
  cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)

  def show_grid(ax, ex, title):
    ax.set_title(title)
    rows, cols = np.array(ex).shape
    ax.set_xticks(np.arange(cols+1)-0.5, minor=True)
    ax.set_yticks(np.arange(rows+1)-0.5, minor=True)
    #ax.grid(True, color='black', which='minor', linewidth=0.5, snap=False)
    ax.grid(True, color='black', which='minor', linewidth=1, snap=False)
    ax.set_xticks([]); ax.set_yticks([])
    # https://github.com/matplotlib/matplotlib/issues/12934
    #  "Misalignment between grid and image in imshow w/ examples"
    ax.imshow(np.array(ex), cmap=cmap, vmin=0, vmax=9)  # , aspect='equal'
    #ax.imshow(np.array(ex), cmap=cmap, vmin=0, vmax=9, extent=[0-0.5, cols-0.5, 0-0.5, rows-0.5])
    if rows<12 and cols<12:
      for i in range(rows): 
        for j in range(cols): 
          if ex[i][j]>0:
            ax.annotate(str(ex[i][j]), xy=(j, i), ha='center', va='center', color='black')     

  def plt_to_html_png(plt):
    tmpfile = BytesIO()
    plt.savefig(tmpfile, format='png')
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
    return '<img src=\'data:image/png;base64,{}\'>'.format(encoded)

  examples_max = max( len(task['train']), len(task['test']))
  fig, axs = plt.subplots(examples_max, 6, figsize=(10, examples_max*3*0.7))
  axs = axs.reshape(-1, 6)  # Reshape axs to have 2 dimensions
  #print(axs.shape)
  
  for i in range(examples_max):
    if i<len(task['train']):
      example = task['train'][i]
      show_grid(axs[i, 0], example['input'], f'Train Input {i}')
      show_grid(axs[i, 1], example['output'], f'Train Output {i}')
      if 'guess' in example: # plot gpt output if present
        show_grid(axs[i, 2], example['guess'], f'Guess {i}')
      else:
        axs[i, 2].axis('off')
    else:
      for j in range(0,3):
        axs[i, j].axis('off')
    if i<len(task['test']):
      example = task['test'][i]
      show_grid(axs[i, 3+0], example['input'], f'Test Input {i}')
      show_grid(axs[i, 3+1], example['output'], f'Test Output {i}')
      if 'guess' in example: # plot gpt output if present
        show_grid(axs[i, 3+2], example['guess'], f'Guess {i}')
      else:
        axs[i, 3+2].axis('off')
    else:
      for j in range(0,3):
        axs[i, 3+j].axis('off')
  plt.tight_layout()

  html = plt_to_html_png(plt)
  plt.show()

  return html

"""
# The basic data underlying is a numpy matrix
# A grid is the visual appearance
class Matrix(object):
  def __init__(self, matrix):
    self.matrix=matrix
  def to_grid(self):
    pass
  def to_coords(self):
    pass
  def to_objects(self):
    pass
"""

class Grid(object):
  def __init__(self, matrix, offset_row=0, offset_col=0):
    self.matrix_np=np.array(matrix, dtype=np.int32)
    self.offset_row=offset_row
    self.offset_col=offset_col
    
  @classmethod
  def empty(cls, rows=3, cols=5, background=0):
    return cls( np.zeros((rows, cols))+background )

  def to_txt_colours(self, num_to_token=colours_for_representation):
    obj=dict( rows=self.matrix_np.shape[0], cols=self.matrix_np.shape[1], offset_row=0, offset_col=0, )
    obj['grid']='\n'.join([
      ''.join([ f' {num_to_token[e]}' 
                for e in row[:] ]) 
      for row in self.matrix_np[:]
    ])
    return obj

  def to_txt_colour_planes(self, empty='empty', filled='filled', background=None):
    obj=dict( rows=self.matrix_np.shape[0], cols=self.matrix_np.shape[1], offset_row=0, offset_col=0, )
    colour_set = set([ int(e) for e in self.matrix_np.flat.copy() ])
    if background is not None:
      colour_set.discard( background )
    planes=[]
    for c in colour_set:
      grid = '\n'.join([
        ''.join([ f' {filled if e==c else empty}' 
                  for e in row[:] ]) 
        for row in self.matrix_np[:]
      ])
      planes.append( dict( colour=f' {colours_for_representation[c]}', grid=grid) )
    obj['planes']=planes
    return obj


def description_strip_initial_verbiage(s):
  i=s.find('...')
  s = s.replace('light blue', 'azure').replace('lightblue', 'azure')
  if s.endswith('.'): s=s[:-1]
  if i<0: return s
  return s[i+3:].strip()

