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

# ## Load up the tactics suggested by Gemini, and cluster them
#
# Code is standalone (no RAG dependence)

# %load_ext autoreload
# %autoreload 2

import os, time
import yaml, requests
import numpy as np
import matplotlib.pyplot as plt

# +
experiment_path = './experiments/gemini-solution-annotation-v1'
train_idx=0
task_hash_sample='caa06a1f'

def get_task_filename(task_hash, idx):
  return f"{experiment_path}/{task_hash}_{idx}.md"


# +
def get_embedding_ollama(txt, ollama_embedding_url='http://yishun:11434/api/embeddings'):
  payload = {
    "model": "unclemusclez/jina-embeddings-v2-base-code",
    "prompt": txt,
    "format": "json",
    "stream": False,
  }
  response = requests.post( ollama_embedding_url, json=payload, headers={"Content-Type": "application/json"},)
  if not response.ok:
    print(f"Ollama failed for {len(txt)=} :: '{txt}' ")  
    return None
  emb_np = np.array(response.json()['embedding'], dtype=np.float32)  
  return emb_np/np.linalg.norm(emb_np)   # These weren't normalised (surprisingly)
  
get_embedding_ollama("Hello world").shape

# +
import arc_mdda.model.gemini

def get_tactics_from_log(log_llm_filename):
  lines    = arc_mdda.model.gemini.read_file_as_lines(log_llm_filename)
  sections = arc_mdda.model.gemini.get_sections_or_raise(lines)
  tactics  = arc_mdda.model.gemini.parse_part_yaml( sections.get('### Part 3', []) )
  return tactics['tactics']

log_llm_filename=get_task_filename(task_hash_sample, train_idx)
tactics = get_tactics_from_log(log_llm_filename)
for tactic in tactics:
  print(tactic)
# -

# Now go through all llmlog files, and accumulate all the tactics suggested
tactic_arr=[]
for f in sorted(os.listdir(experiment_path)):
  if not f.endswith('.md'): continue
  task_hash = f.split('_')[0]
  log_llm_filename=get_task_filename(task_hash, train_idx)
  try:
    tactics = get_tactics_from_log(log_llm_filename)
    tactic_arr.extend(tactics)
  except Exception as e:
    print(f"Skipping {f} due to error")
len(tactic_arr)

sorted(tactic_arr, key=lambda d:d['heading'])[:10]

# +
# Now let's embed the tactic_arr
t0 = time.time()

emb_arr=[]
for tactic in tactic_arr:
  emb_np = get_embedding_ollama(f"*{tactic['heading']}* : {tactic['description']}")
  emb_arr.append(emb_np)
emb_arr_np = np.stack(emb_arr)
emb_arr = None # Clear out memory

t_elapsed, n = (time.time()-t0), emb_arr_np.shape[0]
print(f"Embedding all {n:d} tactics took : {t_elapsed:.2f}sec = {t_elapsed*1000./n:.3f}ms per embedding line")
# Embedding all 293 tactics took : 29.04sec = 99.127ms per embedding line
# Embedding all 2620 tactics took : 259.19sec = 98.929ms per embedding line
emb_arr_np.shape
# -

# ## Clustering
#
# * https://towardsdatascience.com/clustering-sentence-embeddings-to-identify-intents-in-short-text-48d22d3bf02e

# +
## https://umap-learn.readthedocs.io/en/latest/
# dnf install llvm-devel
# uv pip install llvmlite umap-learn
import umap

n_neighbors=5
random_state=42

umap_mapper = umap.UMAP(n_neighbors=n_neighbors, metric='cosine', #min_dist=0.0,
                        n_components=2, # Dimensionality for output
                        n_jobs=1, random_state=random_state).fit(emb_arr_np)
umap_mapper.embedding_.shape

# +
#umap.plot.connectivity(umap_mapper, show_points=True); # Not very interesting

# +
## https://hdbscan.readthedocs.io/en/latest/basic_hdbscan.html
# uv pip install hdbscan
import hdbscan

min_cluster_size=30

clusters = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                           metric='euclidean', 
                           cluster_selection_method='eom').fit(umap_mapper.embedding_)

#clusters.probabilities_  # Probability that label assigned is correct
n_clusters=clusters.labels_.max()+1 # Label assigned to each row of emb_arr (-1 is 'noise')
n_clusters

# +
## https://umap-learn.readthedocs.io/en/latest/document_embedding.html
# uv pip install matplotlib pandas datashader bokeh holoviews
import umap.plot

ax=umap.plot.points(umap_mapper, labels=clusters.labels_)
# -

# So : Can go through the clusters, and find the highest-probability tactic in each one...
#   Noise 'cluster' is a bit different, though
annotations_txy=[]
for cluster_label in range(n_clusters):
  best_example, best_score, cluster_count = None, -1, 0
  for idx_example, label in enumerate(clusters.labels_):
    if label!=cluster_label: continue
    score = clusters.probabilities_[idx_example]
    cluster_count += 1
    #print(f"{score:4f} vs {best_score:.4f}")
    if best_score<score:
      best_example, best_score = idx_example, score
  print(f"{cluster_label} : count={cluster_count:d}")  # , score={best_score:.4f} - always 1 for best
  for k,v in tactic_arr[best_example].items():
    print(f"  {k}: {v}")
  umap_coord = umap_mapper.embedding_[best_example, :]
  annotations_txy.append([ tactic_arr[best_example]['heading'], umap_coord[0], umap_coord[1] ] )

# +
## https://umap-learn.readthedocs.io/en/latest/clustering.html

embeddings = umap_mapper.embedding_ 
plt.scatter(embeddings[:, 0], embeddings[:, 1], c=clusters.labels_, cmap='Spectral') # s=0.1 = marker size
plt.grid(True)

for a in annotations_txy: # This just the 'best' examples found above
  plt.annotate(a[0], (a[1], a[2]),  # (txt, (x[i], y[i])) 
               rotation=75, horizontalalignment='center', verticalalignment='center', 
               fontsize='small', 
               #backgroundcolor='white', zorder=-1  # , alpha=0.5
              )

frame = plt.gca()
frame.axes.xaxis.set_ticklabels([])
frame.axes.yaxis.set_ticklabels([])
plt.show()
# Hmm : https://stackoverflow.com/questions/19073683/how-to-fix-overlapping-annotations-text
# -

#   Noise 'cluster' is a bit different, though
noise_count=0
for idx_example, label in enumerate(clusters.labels_):
  if label!=-1: continue
  score = clusters.probabilities_[idx_example]
  noise_count+=1
  #print(f"{score:4f} vs {best_score:.4f}")
  #if best_score<score:
  #  best_example, best_score = idx_example, score
  print(f"Noise sample : ")  # score={score:.4f} - always zero for noise 
  for k,v in tactic_arr[idx_example].items():
    print(f"  {k}: {v}")
print(f"\nNoise count={noise_count:d}")

# +
# Hmm : https://stackoverflow.com/questions/19073683/how-to-fix-overlapping-annotations-text
## https://python-graph-gallery.com/web-text-repel-with-matplotlib/
# pip install adjustText
import adjustText

embeddings = umap_mapper.embedding_ 

plt.subplots(figsize=(12, 12),) #facecolor='lightskyblue', layout='constrained')

plt.scatter(embeddings[:, 0], embeddings[:, 1], c=clusters.labels_, cmap='Spectral') # s=0.1 = marker size
plt.grid(True)

GREY50="#7F7F7F"
annotations_adj = []
for a in annotations_txy: # This just the 'best' examples found above
  annotations_adj.append( plt.text(a[1], a[2], a[0], fontsize='small', ha='center', va='center',
                          backgroundcolor='#eeeeee', 
                          #zorder=-1  # , alpha=0.5
                        ) )
frame = plt.gca()
frame.axes.xaxis.set_ticklabels([])
frame.axes.yaxis.set_ticklabels([])
adjustText.adjust_text(
    annotations_adj, 
    #expand_points=(2, 2),
    arrowprops=dict(arrowstyle="->", color=GREY50, lw=1),
    ax=frame.axes
)    
plt.show()
# -


