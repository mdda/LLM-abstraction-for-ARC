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

# ## Try to get some commentary from Gemini

import os #, sys

# %load_ext autoreload
# %autoreload 2

# ### Load in DSL manipulation functions

# +
from arc_mdda.dsl_manipulation import DSL_Interface

DSL = DSL_Interface()
len(DSL.functions), len(DSL.functionals)
# -

import arc_mdda.dsl_manipulation
grid_to_tuples_fn = arc_mdda.dsl_manipulation.grid_to_tuples

# ### Load in the Core Knowledge

import arc_mdda.core_knowledge

# ### Load in the training dataset

import arc_mdda
data_train = arc_mdda.load_json_data('./External/ARC-AGI/data/training/')
len(data_train), list(data_train.keys())[0:4]

# ### Now get access to the Gemini model

from arc_mdda.model import gemini
import textwrap

#model = gemini.get_model("gemini-1.5-pro-002")  # Surprisingly, slightly worse in initial test...
model = gemini.get_model("gemini-1.5-flash-002")   
model = gemini.RetryingLLM( model )

# ### Try out with a specific example

task_hash = '00d62c1b'  #inside/outside : Works
#task_hash = '007bbfb7'  #replicate pattern according to itself
#task_hash = '4290ef0e'  # problem has lots of natural chunks in solve
#task_hash = '00d62c1b'  # arc-dsl-llm docs example

definition = DSL.definitions[f'solve_{task_hash}']
def_lines = arc_mdda.dsl_manipulation.analyse_definition(definition, DSL.functions, DSL.constants, DSL.functionals)

# +
color_constants_exec = arc_mdda.dsl_manipulation.color_constants_base

#color_constants_llm  = [f"C[' {tok}']" for tok in arc_mdda.dsl_manipulation.color_scheme]
color_constants_llm  = arc_mdda.dsl_manipulation.color_scheme + arc_mdda.dsl_manipulation.color_scheme_addons

solver_py = arc_mdda.dsl_manipulation.create_virtual_solver(def_lines, color_constants=color_constants_exec)
solver_py_llm = arc_mdda.dsl_manipulation.create_virtual_solver(def_lines, 
                           color_constants=color_constants_llm, dsl_functions=DSL.functions, for_llm=True)
#print(solver_py)
print(solver_py_llm)

# +
task_data = data_train[task_hash]['train']
#task_data = data_train[task_hash]['test']
I = grid_to_tuples_fn( task_data[0]['input'] )
O = grid_to_tuples_fn( task_data[0]['output'] )

#exec(import_preamble + color_map_preamble + solver_new, solver_globals) # Defines solver_virtual(I)
#O_dict = solver_globals['solver_virtual'](I)
solver_function = arc_mdda.dsl_manipulation.get_solver_function(solver_py, DSL.arc_dsl)
O_dict = solver_function(I)

# +
#color_tokens='x r b y e q c z p u w q'.split(' ')
color_tokens = color_constants_llm

#detect_type(O_dict['x1'], )
for k,v in O_dict.items():
  t=arc_mdda.dsl_manipulation.detect_type(v)
  print(f" {k} is {t}")
  #print(f"    = {arc_mdda.dsl_manipulation.render_variable(v,t,color_tokens)}") 

# +
#' '.join(color_tokens)

# +
import arc_mdda.prompts.comment_code

prompt_parts = arc_mdda.prompts.comment_code.get_prompt_parts_to_deconstruct_solve(solver_py_llm, O_dict, DSL.BASE)

for idx, part in enumerate(prompt_parts):
  if idx in ([] #+[0,1,2,3,4,]+[5,6,7,8,]+[9,]
                 +[10,11,]   # Part 1
                 #+[12,13]     # Part 2 
                 #+[14,15,16] # Part 3
                 #+[17,18,]   # Part 4
                ):  # 
    print(part)
# -

PAUSE - before we spend money on calling an API...

answer = model.generate_content(prompt_parts).text
print(answer)

# ### Generate files for a subset of the training set

# +
experiment_path = './experiments/gemini-solution-annotation-v1'
os.makedirs(experiment_path, exist_ok=True) 

train_idx=0

def get_filtered_task_hashes():
  arr=[]
  for task_hash in sorted(data_train.keys()):
    d=task_hash[-1]
    #if d not in 'ef':    # (i.e. one eighth of them ~ 45)
    #if d not in 'cdef':  # (i.e. one quarter of them)
    #if d not in '89abcdef':  # (i.e. half of them)
    #  continue # Only do the task_hashes ending with these characters 
    arr.append(task_hash)
  return arr

def get_task_filename(task_hash, idx):
  return f"{experiment_path}/{task_hash}_{idx}.md"

for task_hash in get_filtered_task_hashes():
  #print(f"Checking {task_hash} @{train_idx}")
  
  task_data = data_train[task_hash]['train']
  #task_data = data_train[task_hash]['test']
  if train_idx>=len(task_data): continue # Nothing to do : There's no example to play with here

  log_llm_filename = get_task_filename(task_hash, train_idx)
  if not os.path.isfile(log_llm_filename): 
    print(f"  Running {task_hash} @{train_idx}")

    # Turn the given solver function definition into a callable
    definition = DSL.definitions[f'solve_{task_hash}']
    def_lines = arc_mdda.dsl_manipulation.analyse_definition(definition, DSL.functions, DSL.constants, DSL.functionals)
  
    solver_py = arc_mdda.dsl_manipulation.create_virtual_solver(def_lines, color_constants=color_constants_exec)
    solver_function = arc_mdda.dsl_manipulation.get_solver_function(solver_py, DSL.arc_dsl)
  
    # Run the train_idx example through the solver_function
    I = grid_to_tuples_fn( task_data[train_idx]['input'] )
    O = grid_to_tuples_fn( task_data[train_idx]['output'] )
    
    O_dict = solver_function(I)
    if task_hash not in "|228f6490|f8c80d96|":   # Two problem found (in second half of dataset...)
      assert( O == O_dict['O'] ) # Should work if the DSL solution is valid

    # Get gemini prompt for annotating solution code
    solver_py_llm = arc_mdda.dsl_manipulation.create_virtual_solver(def_lines, 
                                color_constants=color_constants_llm, dsl_functions=DSL.functions, for_llm=True)
    prompt_parts = arc_mdda.prompts.comment_code.get_prompt_parts_to_deconstruct_solve(solver_py_llm, O_dict, DSL.BASE)
  
    answer = model.generate_content(prompt_parts).text
    with open(log_llm_filename, 'wt') as f:
      f.write(answer)
  else:
    pass # already exists - don't recreate
    
  # Now the file exists for sure... let's analyse it!
"FINISHED"

# +
# This runs through all of the files, validating them and creating annotated code snippets (for RAG)
import yaml
import arc_mdda.model.rag

# Do some extraction from the files in the `experiment_path` folder...
count_processed, invalid_filenames=0, []
for task_hash in get_filtered_task_hashes():
#for task_hash in '29ec7d0e'.split(' '):
  #print(f"Checking {task_hash} @{train_idx}")
  log_llm_filename = get_task_filename(task_hash, train_idx)
  lines = arc_mdda.model.gemini.read_file_as_lines(log_llm_filename)
  
  #print(f"{task_hash} :")
  if len(lines)==0:
    print(f"  {log_llm_filename} missing")
    continue
 
  valid=True  # If this is False we should ask for a do-over...

  if valid:
    try:
      sections = arc_mdda.model.gemini.get_sections_or_raise(lines)
    except Exception as e:
      print(f"{task_hash} :: Failed to get sections {e}")
      # Should regenerate file for sure...
      valid=False
  
  base_function='solver_virtual_chunked' # 'solver_virtual_refactored'
  if valid:
    try:
      original   = arc_mdda.model.gemini.parse_part_1( sections.get('### Part 1', []) )
      refactored = arc_mdda.model.gemini.parse_part_2( sections.get('### Part 2', []), calling_function=base_function, debug=False )
      tactics    = arc_mdda.model.gemini.parse_part_yaml( sections.get('### Part 3', []) )
      steps      = arc_mdda.model.gemini.parse_part_yaml( sections.get('### Part 4', []) )

      # FAILURE MODES...  REJECT IF:
      #   subfunctions call subfunctions <- definitely could be a problem
      arc_mdda.model.gemini.check_for_calls_to_subfunctions(refactored, base_function=base_function)
    except Exception as e:
      #print(dir(e))
      print(f"{task_hash} :: Failed to parse : {e}")  # \n{e.with_traceback}
      # Should regenerate file for sure...
      valid=False

  if valid:
    try:
      code_with_comments_base       = arc_mdda.model.gemini.gather_comments_and_code( original, base_function='solver_virtual' ) 
      code_with_comments_refactored = arc_mdda.model.gemini.gather_comments_and_code( refactored, base_function=base_function ) 
      #print( code_with_comments )

      code_with_comments_refactored, fix_count = arc_mdda.model.gemini.fix_empty_comments_if_possible(
        code_with_comments_refactored, code_with_comments_base)
      if fix_count>0:
        print(f"  {fix_count=}")
      
      # FAILURE MODES...  REJECT IF:
      #   empty comments (most should have been fixed by procedure above)
      arc_mdda.model.gemini.check_for_empty_comments(code_with_comments_refactored)  # Raises if there's a problem
      #   input_ , output_ variable lists missing (can happen if Gemini chooses to overhelpfully rename all variables)
      arc_mdda.model.gemini.check_for_empty_input_and_output(code_with_comments_refactored)  # Raises if there's a problem
    except Exception as e:
      print(f"{task_hash} :: Code with comment had a problem : {e}")
      # Should regenerate file for sure...
      valid=False

  if valid: # Figure out the right 'tactics' and 'step' that's being used at each stage
    pass

  ## These goal *prompt* require the values in O_dict : Which aren't computed here...
  ##   Actually used in 08_code-generation-trail (and the local model trainer)
  #if valid: # SIMPLIFIED : Create prompt from full IO example(s) with 'goal (etc)' for RAG access with <train> </train> tokens
  #  # OLD : Now go through the 'steps' and find the most relevant code line... (i.e. input_variables[] and output_variables[] match, roughly)
  #  # NEW : Now go through the commented code, and spit out I/O, x_i, goal, x_j, goal, etc...
  #  prompt_goals = arc_mdda.prompts.stepwise_goals_for_code.get_prompt_parts_to_plan_code_using_goals_alone(
  #    code_with_comments_refactored, O_dict, DSL_BASE)

  if valid: # Output fragments for RAG searching
    rag_fragments=[]
    rag_fragments.extend( arc_mdda.model.rag.generate_code_blocks_from_commented_code(code_with_comments_base) )
    rag_fragments.extend( arc_mdda.model.rag.generate_code_blocks_from_commented_code(code_with_comments_refactored) )
   
    rag_fragments.extend( arc_mdda.model.rag.generate_code_blocks_from_subfunctions(refactored, base_function=base_function) )

    # Save the RAG fragments into yaml file
    with open(log_llm_filename.replace('.md','.rag.yaml'), 'w') as ragout:
      yaml.dump(rag_fragments, ragout) # , default_flow_style=False
      
  #print(f"  {valid=}")
  count_processed+=1
  #break
  
  if not valid: 
    print(f"  {valid=}")
    invalid_filenames.append(log_llm_filename)
# First run of 45 (9mins): redo 14 (4mins)// second run : redo 9 // third run : redo 7 // fourth run : redo 4 // 3 .. 3 .. 2
# 16 invalid out of 93 processed // 12 .. 7 .. 5 .. 4 .. 3 .. 3 .. 2 .. 2 
f"FINISHED : {len(invalid_filenames)} invalid out of {count_processed} processed"
# -
if False:  # Enable this specifically...
  print(f"Deleting {len(invalid_filenames)} files")
  for log_llm_filename in invalid_filenames:
    os.unlink(log_llm_filename)
    print(f"  Deleted {log_llm_filename}")

PAUSE

code_with_comments_base
#code_with_comments_refactored

steps

len(rag_fragments)

for k,v in rag_fragments:
  print(k)
  print(v)
  print()  




