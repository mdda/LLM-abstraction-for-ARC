
def comment_original_solution_OLD(part_num):
  return [
    f"""### Part {part_num} : Add comments to original solution
Add comments into the program code for function `solver_virtual(I)` above, at suitable points that break the code into reasonable code blocks.
Each code block can be as short as one line, or as long as necessary to encompass a complete subtask
Each set of comments should relate to the code block that follows.
Each comment should refer to the following ideas (each on a separate line, starting `#` as shown):
* `# Input: ` What input the code is expecting at that point (in terms of types, and in terms of the overall goal of the solution)
* `# Goal: ` What the goal of the next line of code are (both locally, and how it relates to the overall goal of the solution).  
* `# Output: ` What the expected output of this block (in terms of types, and in terms of the overall goal of the solution)
* (optional) `# Core Knowledge: ` If any elements of Core Knowledge are relevant to the block, describe them in an additional comment line.
""",
    f"""#### Part {part_num} Answer Format
Your answer should repeat the program code above, with your comments interspersed among the logical blocks within the code.
""",
  ]

def comment_original_solution(part_num):
  return [
    f"""### Part {part_num} : Add comments to original solution
Add comments into the program code for function `solver_virtual(I)` above, at the points indicated by `# comment`.
If it makes sense, comments can be skipped, so that lines of code are combined into more reasonable code blocks.
Each code block can be as short as one line, or as long as necessary to encompass a complete subtask.
Each set of comments should relate to the code block that follows.
""",
    f"""#### Part {part_num} Answer Format
Your answer should repeat the program code of `solver_virtual(I)` above, with the comments included according to the code blocks you decide.  
Each set of comments should be in the following format:
* `# Input: ` What input the code is expecting at that point (in terms of types, and in terms of the overall goal of the solution)
* `# Goal: ` What the goal of the next line of code are (both locally, and how it relates to the overall goal of the solution).  
* `# Output: ` What the expected output of this block (in terms of types, and in terms of the overall goal of the solution)
* (optional) `# Core Knowledge: ` If any elements of Core Knowledge are relevant to the block, describe them in an additional comment line.
""",
  ]

def get_reusable_components_txt_OLD(part_num, part_comment_format):
  return [
    f"""#### Part {part_num} Answer Format
For each new function component: 
* there should be only 1 or 2 input parameters for the function.
* comments for the entire function should be included in the same format as Part {part_comment_format}
* variables within the function can be named according to their purpose
* the output should be a single return value

Also output `solver_virtual_refactored(I)`:
* variables within `solver_virtual_refactored(I)` itself should not be renamed
* lines with unused variables should be omitted
* contains calls to the new functions as appropriate
* comments should be included in the same format as Part {part_comment_format} for each line in `solver_virtual_refactored(I)` 
* all calls to new functions must be from `solver_virtual_refactored(I)` - this is a depth-1 refactoring
""",
  ]

def get_reusable_components_py_OLD(part_num, part_comment_format):
  return [
    f"""#### Part {part_num} Answer Format
The following example illustrates the format of two function components and `solver_virtual_refactored(I)`:
```python
def find_adjacent_single_pairs(objects: Objects, color: Color) -> FrozenSet:  # New function component, Maximum of 2 input variables
  # Input: objects (Objects), a set of objects in the input grid
  # Goal: Find pairs of single-cell and specificed color objects that are vertically adjacent.
  # Output: pairs (FrozenSet), a set of (single-cell, color) object pairs that are vertically adjacent.
  # Core Knowledge: Object influence via contact (adjacency), Object filtering, Set operations
  single_cell_objs = size_filter(objects, 1) # Rename variables as appropriate in function components  
  color_objs = color_filter(objects, color)  
  all_pairs = cartesian_product(single_cell_objs, color_objs)
  adjacency_condition = combine_two_function_results(vertical_matching, get_first, get_last)
  pairs = keep_if_condition(all_pairs, adjacency_condition)
  return pairs

def recolor_single_cell_objects(pairs: FrozenSet) -> FrozenSet:  # New function component, function components must call at least 2 DSL functions
  # Input: pairs (FrozenSet), pairs of single-cell and grey objects
  # Goal: Recolor each single-cell object based on its adjacent grey object's color.
  # Output: recolored_objects (FrozenSet), a set of locations and recolored single-cell objects.
  # Core Knowledge: Object transformation (recoloring), Compositionality
  recoloring_function = combine_two_function_results(recolor, compose(color, get_first), get_last)
  recolored_objects = transform_and_flatten(recoloring_function, pairs)
  return recolored_objects

def solver_virtual_refactored(I):  # This function calls the function components
  x1 = as_objects(I)  # Retain original code if not refactored
  x6 = find_adjacent_single_pairs(x1, GREY)  # Retain original variable names in refactored caller
  x9 = recolor_single_cell_objects(x6)  # Retain original variable names in refactored caller 
  O = paint_onto_grid(I, x9)   # Retain original code if not refactored
  return O
```
""",
  ]

def get_reusable_components_py(part_num, part_comment_format):
  # Example from : experiments/ddf7fa4f_0.md
  return [
    f"""#### Part {part_num} Answer Format
The following example illustrates the format of two function components and `solver_virtual_refactored(I)`:
```python
def recolor_single_cell_objects(pairs: FrozenSet, color: Color) -> FrozenSet:  # New function, which calls at least 2 DSL functions
  # Input: pairs (FrozenSet), color (Color), pairs of single-cell and grey objects
  # Goal: Recolor each single-cell object based on its adjacent object's color.
  # Output: recolored_objects (FrozenSet), a set of locations and recolored single-cell objects.
  # Core Knowledge: Object transformation (recoloring), Compositionality
  recoloring_function = combine_two_function_results(recolor, compose(color, get_first), get_last)  # variables named appropriately
  recolored_objects = transform_and_flatten(recoloring_function, pairs) # variables named appropriately
  return recolored_objects

# ...  other new functions here

def solver_virtual_chunked(I):  # This function calls the new functions, replacing suitable chunks. Variable names in this function are the same as in `solver_virtual`
  # Input: I (Grid), the input grid.
  # Goal: Identify and separate objects within the input grid.
  # Output: x1 (Objects), a set of objects identified in the input grid.
  # Core Knowledge: Object cohesion (parsing grids, identifying distinct objects based on spatial contiguity)
  x1 = as_objects(I)  # Retain original code (and variable names) if not moved to new function

  # Input: x1 (Objects), a set of objects.
  # Goal: Filter objects based on their size (select only single-cell objects).
  # Output: x2 (Objects), a subset of x1 containing only single-cell objects.
  # Core Knowledge: Numbers and Counting priors (size filtering).
  x2 = size_filter(x1, 1)   # Retain original code (and variable names) if not moved to new function

  # ... other lines here - with each block also having comments in the format of Part {part_comment_format}.

  # Input: x2 (FrozenSet), pairs of single-cell and objects
  # Goal: Recolor each single-cell object based on its adjacent object's color.
  # Output: x9 (FrozenSet), a set of locations and recolored single-cell objects.
  # Core Knowledge: Object transformation (recoloring), Compositionality
  x9 = recolor_single_cell_objects(x2, GREY)  # Call new function, retain original variable names in caller 

  # Input: I (Grid), input grid; x9 (FrozenSet), recoloring instructions.
  # Goal: Update input grid by painting the recolored objects onto it.
  # Output: O (Grid), the output grid after recoloring.
  # Core Knowledge: Object manipulation (painting), Compositionality.
  O = paint_onto_grid(I, x9)   # Retain original code if not refactored
  return O
```
""",
  ]

def get_reusable_components_OLD(part_num, part_comment_format):
#Create zero, one or more new functions that would support Problem solvers like `solver_virtual(I)`.
#The intent of this part is to make reusable components that might also be useful for other Problems.
#These functions likely consist of self-contained code blocks (which may not appear on sequential lines in `solver_virtual(I)`).
  return [
    f"""### Part {part_num} : Create reusable components
Create zero, one or more new function components that would support Problem solvers like `solver_virtual(I)`.
Using these new function, refactor `solver_virtual(I)` into `solver_virtual_refactored(I)`.
Each of the new function components must be both **meaningful** and **reusable**:
* **meaningful** means that each new function component must include several DSL calls (each function component must do 2 or more DSL calls).
* **reusable** means that each new function component must be potentially useful in the solution of other Problems.

Relationship between `solver_virtual_refactored(I)` and the new function components:
* Function components are only called by `solver_virtual_refactored(I)`
* Each function component should be independent of the other function components

""",
  ] + get_reusable_components_py_OLD(part_num, part_comment_format)
  #] + get_reusable_components_txt_OLD(part_num, part_comment_format)

def get_reusable_components(part_num, part_comment_format):
  return [
    f"""### Part {part_num} : Create reusable components
Create a new version of `solver_virtual(I)` from Part {part_comment_format} called `solver_virtual_chunked(I)` which has the same functionality.
To create `solver_virtual_chunked(I)`, examine each line of code (and surrounding lines):
* move natural blocks of code (consisting of several lines of code each) into separate new functions, with a call from `solver_virtual_chunked(I)`.
* blocks of code must return concrete variables. 
* Callables should only be used be within a block
* if there are lines that are not easily isolated, leave them unchanged in `solver_virtual_chunked(I)`.
Comments in the same format as Part {part_comment_format} should be added to each line of `solver_virtual_chunked(I)`.
""",
  ] + get_reusable_components_py(part_num, part_comment_format) 


def get_high_level_tactics_py(part_num):
  return [
    f"""### Part {part_num} : High-level tactics
Outline potential high-level tactics that could be used to solve this problem, if `solver_virtual(I)` was unknown.
""",
    f"""#### Part {part_num} Answer Format
Here, a bulleted list of suggested tactics would be appropriate.  Mention relevant DSL functions as appropriate.  
Return 5 or more tactics in this format.
""",
    """#### Part 4 Examples
Some examples of tactics: 
* **Better Representation**: Seek a better representation of the input/output grid : `as_objects`
* **Filter by Property**: From previous list, select accoring to a property : `size_filter`, `most_common_color`, `extract_first_matching`, `equals`
* **Combine Results**: Combine results previous results into find grid : `fill`, `paint_onto_grid`
""",
  #* Identify components that might be useful / useless
  #* Identify (from the new internal state / generated variables) other properties that are relevant
  ]

def get_high_level_tactics_yaml(part_num):
  # Self-Discover: Large Language Models Self-Compose Reasoning Structures
  #  https://ar5iv.labs.arxiv.org/html/2402.03620
  return [
    f"""### Part {part_num} : High-level tactics
Outline potential high-level tactics that could be used to solve this problem, if `solver_virtual(I)` was unknown.
""",
    f"""#### Part {part_num} Answer Format
Fill in the following YAML structure (the comments explain the intent of the entries):
```yaml
tactics:
  - heading: "" # A short name for the tactic 
    description: "" # A description of the tactic 
    dsl_functions: [] # A list of relevant DSL functions (as appropriate)
```
Return 5 or more tactics in this format.
""",
    f"""#### Part {part_num} Examples
Some examples of tactics: 
```yaml
tactics:
  - heading: "Better Representation"
    description: "Seek a better representation of the input/output grid"
    dsl_functions: [as_objects]
  - heading: "Filter by Property"
    description: "From the list, select according to a property"
    dsl_functions: [size_filter, most_common_color, extract_first_matching, equals]
  - heading: "Combine Results"
    description: "Combine previous results into final grid"
    dsl_functions: [fill, paint_onto_grid]
```
""",
  ]

def overall_solution_description_bullets_OLD(part_num, part_comment_format):
  # OLD : The docstring should describe the an overall Problem strategy using the same comment structure as Part 1, but making sure to identify each element of Core Knowledge that is required to solve the overall Problem.
  return [
    f"""### Part {part_num} : Overall solution description
Describe the high-level steps involved in solving the overall Problem.  
This requires stating the overall expected contents of the Input grid, a sequence of steps required to solve the problem, and the expected contents of the Output grid.
The sequence of steps should be expressed in human form (not necessarily correspond directly to lines of code). 
The steps should be described generically (i.e. don't use specific color names or shape descriptions) so that the steps could be reused for other problems.
""",
    f"""#### Part {part_num} Answer Format
Restate the function `solver_virtual(I)` with a function body that contains only a docstring: 
The docstring must include the following ideas (each on a separate line, starting `#` as shown):
* `# Input: ` What input should be expected for the problem
* `# Step 1..N: ` The sequence of high-level steps that describe the key parts of solving the problem
  + For each step, if any variables in Part {part_comment_format} are relevant, list them after the text 'Variables:'
  + For each step, if any elements of Core Knowledge are relevant, list them after the text 'Core Knowledge:'
* `# Output: ` What the expected output for the problem solution
""",
  ]

def overall_solution_description_yaml(part_num, part_comment_format, part_tactics):
  return [
    f"""### Part {part_num} : Overall solution description
Describe the high-level steps involved in solving the overall Problem.  
This requires stating the overall expected contents of the Input grid, a sequence of steps required to solve the problem, and the expected contents of the Output grid.
The sequence of steps should be expressed in human form (not necessarily corresponding directly to lines of code). 
The steps should be described generically (i.e. don't use specific color names or shape descriptions) so that the steps could be reused for other problems.
""",
    f"""#### Part {part_num} Answer Format
Fill in the following YAML structure (the comments explain the intent of the entries):
```yaml
input: "" # What input should be expected for the problem
steps: # An array with elements that correspond to each high-level step
  - text: ""  #  describes this key part of solving the problem
    tactic_used: ""  # the tactic heading from Part {part_tactics} that is most relevant to this step
    core_knowledge: []  # if any elements of Core Knowledge are relevant to this step, list them (eg: ['Object Manipulation', ...])
    variables_input: [] # if any variables in Part {part_comment_format} are needed before doing this step, list them (eg: [x3, x4])
    variables_output: [] # if any variables in Part {part_comment_format} are created by this step, list them (eg: [x3, x4])
output: "" # What output should be expected for the problem solution
```
""",
  ]

from .. import core_knowledge
from .. import dsl_manipulation

def get_prompt_parts_to_deconstruct_solve(solver_py_llm, O_dict, DSL_BASE):
  return [
    core_knowledge.prelude,
    core_knowledge.description,
    #arc_mdda.core_knowledge.combinations,  # incorporated in the description
    '\n'.join(dsl_manipulation.get_constants_documentation(DSL_BASE)), 
    '\n'.join(dsl_manipulation.get_dsl_documentation(DSL_BASE)), 
    """## Problem-specific example variables
For this task, the input grid `I` and output grid `O` are as follows:
""",
    dsl_manipulation.print_variable_for_llm('I', O_dict['I']),
    dsl_manipulation.print_variable_for_llm('O', O_dict['O']),
    """\n## Problem-specific solution
For this task, the program code (expressed using the DSL above) that solves the problem is as follows:
""",
    "```python\n"+solver_py_llm+"\n```\n",
    """## Task
There are 4 parts to this task:
""", 
    *comment_original_solution(1),
    *get_reusable_components(2, 1),
    *get_high_level_tactics_yaml(3),
    *overall_solution_description_yaml(4, 1, 3),
  ]

