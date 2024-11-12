import time

from omegaconf import OmegaConf
conf = OmegaConf.load('config.yaml')

# Issue '429 Unable to submit request because the service is temporarily out of capacity. Try again later.'
#def get_model_gemini(model_name="gemini-1.0-pro-002", free=False):  # gemini-1.0-pro
#def get_model_gemini(model_name="gemini-1.5-flash-001", free=False):  # gemini-1.5-flash
def get_model(model_name="gemini-1.5-flash-002", free=False):  # gemini-1.5-flash
  if free:
    # This is the 'free' version (not VertexAI)
    import google.generativeai as genai
    from google.generativeai import GenerativeModel
    genai.configure(api_key=conf.APIKey.GEMINI_PUBLIC)
    print(f"'...{conf.APIKey.GEMINI_PUBLIC}[-6:]'")
  # https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/configure-safety-attributes#configure_thresholds
    safety_settings = [ {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_ONLY_HIGH"
      },  {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_ONLY_HIGH"
      },  {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_ONLY_HIGH"
      },  {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_ONLY_HIGH"
      },
    ]
    
  else:
    #  export GOOGLE_APPLICATION_CREDENTIALS="key-vertexai-iam.json"
    import vertexai
    from vertexai.generative_models import GenerativeModel, HarmCategory, HarmBlockThreshold
    PROJECT_ID = conf.APIKey.VERTEX_AI.PROJECT_ID
    REGION     = conf.APIKey.VERTEX_AI.REGION  # e.g. us-central1 or asia-southeast1
    vertexai.init(project=PROJECT_ID, location=REGION) 
    safety_settings = {
      HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,  # BLOCK_NONE is not allowed...
      HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
      HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
      HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    }    
  
  # Set up the model
  #   https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/gemini#:~:text=Top-K%20changes
  generation_config = {
    "temperature": 1.0,  # Default for gemini-1.0-pro-002
    "top_p": 1,  # Effectively look at all probabilities
    #"top_p": .9,  # Focus more on top probs
    #"top_k": 1,  # Not used by gemini-pro-1.0 (!)
    #"max_output_tokens": 2048,
    #"max_output_tokens": 4096,  # Needed for get_prompt_parts_to_deconstruct_solve (some YAML cut off)
    "max_output_tokens": 8000,  # Needed for get_prompt_parts_to_deconstruct_solve (!)
  }
  
  return GenerativeModel(model_name=model_name,
                         generation_config=generation_config,
                         safety_settings=safety_settings,
                        )  


class RetryingLLM(object):
  def __init__(self, model, retries_max=5, sleep_sec=5.0): 
    self.model=model
    self.retries_max = retries_max
    self.sleep_sec = sleep_sec

  def generate_content(self, prompt_parts):
    success, e_last = False, None
    for attempt in range(self.retries_max):
      try:
        response=self.model.generate_content(prompt_parts)
        success=True
        break
      except Exception as e:
        print(attempt, e)
        time.sleep(self.sleep_sec)
        print(f"Retry attempt {attempt+1}")
        e_last = e
    if not success:
      raise(e_last)
    return response


## Parsing output from Gemini prompt ('### Part N :' headings, with code blocks as structure)

import os 
def read_file_as_lines(log_llm):
  lines=[]
  if not os.path.isfile(log_llm): return lines
  with open(log_llm, 'rt') as f_ans: 
    return f_ans.readlines()

def get_sections_from_part_headings(lines):
  sections=dict()
  section=None
  for l in lines:
    if l.startswith('### Part '):
      section=l.split(':')[0].strip()
      if not section in sections: sections[section]=[] # Start with empty section
      continue
    if section is None: continue # Not set yet
    sections[section].append(l.rstrip())
  return sections
  
def get_sections_from_code_breaks(lines):
  sections=dict()
  section_num, section = 1, '### Part 1'
  for l in lines:
    if not section in sections: sections[section]=[] # Start with empty section
    sections[section].append(l.rstrip())
    #print(f"get_sections_from_code_breaks :: {section} : {l}")
    if l.strip()=="```":
      section_num+=1
      section=f'### Part {section_num}'
  #for section in list(sections.keys()): # Make keys concrete (since we might delete some inside loop)
  #  if len("".join(sections[section]).strip())==0: 
  #    del sections[section]  # Delete empty sections
  sections = {
    k:v for k,v in sections.items() if len("".join(v).strip())>0  # Skip empty sections
  }
  #print(sections)
  return sections

def get_sections_or_raise(lines):
  sections = get_sections_from_part_headings(lines)
  if len(sections.keys())!=4:  # Check whether Part headings gave us 4 sections
    #print(f"{task_hash} :: Parts : {len(sections.keys())=}")
    sections = get_sections_from_code_breaks(lines)
  if len(sections.keys())!=4:  # Actually looks like they all have 4 parts!
    #print(f"{task_hash} :: No Parts, and no Code segments : {len(sections.keys())=}")
    #continue # Have to abandon this one
    raise Exception(f"  :: No Parts, and no Code segments : {len(sections.keys())=}")
  return sections





def extract_code(lines, lang='python'): # This assumes only one block inside the array 'lines'
  within, code=False, []
  for l in lines:
    if l.strip()=="```":
      within=False
    if within:
      code.append(l.rstrip())
    if l.strip() == f"```{lang}":
      within=True
  return code

def get_python_functions(code):
  functions=dict()
  function=None
  for l in code:
    line = l.rstrip()
    if line.startswith('    '): line=line[2:]  # reduce 4-space indent to 2
    if line.startswith('def '):
      function=line[4:].split('(')[0].strip()
      if not function in functions: functions[function]=[line] # Start with the function definition line
      continue
    if function is None: continue # Not set yet
    functions[function].append(line)
    if line.strip().startswith('return'):
      function=None
  return functions




def parse_part_1(lines):  # This is a single python thing
  code = extract_code(lines) 
  functions = get_python_functions(code)
  return functions
  
def parse_part_2(lines, calling_function='solver_virtual_refactored', debug=False):  # 'solver_virtual_chunked'
  code = extract_code(lines) 
  functions = get_python_functions(code)
  if calling_function not in functions:
    print(f"No '{calling_function}' found in Part 2")
    print('\n'.join(code))
    #print(functions)
    return dict()  # FAILURE state
  # Count the non-comment, non-return lines
  for function in functions.keys():
    count=0
    for l in functions[function]:
      if l.strip().startswith('#'): continue # This is a comment line
      if l.strip().startswith('"""') and l.strip().endswith('"""') : continue # This is a comment line
      if l.strip().startswith('return') and not '(' in l: continue # This is a pure return statement
      #print(f"{function} : {l}")
      count+=1 # This line is countable
    if count<2:
      if debug:
        print(f"  {function}() does not contain enough DSL lines...")
        print( '\n'.join(functions[function]) )
  return functions

import yaml

def parse_part_yaml(lines):
  code = extract_code(lines, 'yaml') 
  return yaml.safe_load('\n'.join(code))

import re
is_standard_variable = re.compile(r'^[IO]|x\d+$')

from ..dsl_manipulation import DefLine

class CommentedCode(object):
  input, goal, output, core_knowledge = "","","",""
  code = []   # code is a list(DefLine)
  input_variables, internal_variables, output_variables = set(), set(), set()
  is_subfunction = False
  def __init__(self,):
    self.code = []  # This will be a list of DefLine (with hardly any information in them)
    self.input_variables, self.internal_variables, self.output_variables = set(), set(), set()
  def __repr__(self):
    return f"""
# Input: {self.input}
# Goal: {self.goal}
# Output: {self.output}
# Core Knowledge: {self.core_knowledge}
# Extra: input={self.input_variables} output={self.output_variables} is_subfunction={self.is_subfunction}
{'\n'.join([ dl.repr_fn() for dl in self.code ])}
""" # .lstrip()
  
  def has_no_comments(self):
    return len(self.input+self.output+self.goal+self.core_knowledge)==0
  
  def add_block_data(self, variable, args):
    if is_standard_variable.match(variable):
      self.internal_variables.add(variable)
    for a in args:
      if is_standard_variable.match(a):
        self.input_variables.add(a)

  

def gather_comments_and_code(functions, base_function='solver_virtual', debug=False):  # 'solver_virtual_chunked', 
  arr=[]  # This will become an array of commented code...
  if base_function not in functions: return arr  # Nothing to do...
  cc_current, before_code = CommentedCode(), True
  for l in functions[base_function]:
    l=l.strip()
    if len(l)==0: continue
    if debug:
      print(f"gather_comments_and_code : {base_function} : {l}")
    if l.startswith('def '):
      continue
    elif l.startswith('"""'):
      continue
    elif l.startswith('#'):
      if not before_code:  # We must have come to the end of the previous code segment
        arr.append(cc_current)
        cc_current, before_code = CommentedCode(), True
      l=l[1:].strip()  # Take off the '#\s*'
      header, txt = l.lower(), l
      if ':' in l:
        pos=l.find(':')
        header, txt = l[:pos].lower(), l[pos+1:]
      txt = txt.strip()
      if txt.endswith('.'): txt=txt[:-1].strip()
      if 'input' in header: cc_current.input = txt
      if 'goal' in header: cc_current.goal = txt
      if 'output' in header: cc_current.output = txt
      if 'knowledge' in header: cc_current.core_knowledge = txt
    elif l.startswith('return'):  
      arr.append(cc_current) # All done
      break
    else:
      before_code=False  # We're in a code thing now... get the parts of this one
      variable, call = l.split(' = ')
      first_bracket = call.find('(')
      function, args = call[:first_bracket], call[first_bracket+1:-1]  # Take off last bracket too
      args = [args] if ',' not in args else args.split(', ') 

      dl = DefLine(0, l, '') # , idx, txt, comment_for_llm (rough-and-ready)
      dl.set_call( variable, function, args ) # variable, function, args )
      if function not in functions:  # This is a DSL call (most likely)
        cc_current.add_block_data(variable, args)
        cc_current.code.append(dl)
      else: # Special case : We're calling a sub-function
        # We need to flush out the previous block (if there is one)
        if len(cc_current.code)>0:
          arr.append(cc_current)
          cc_current, before_code = CommentedCode(), True  # Start the gathering process again...
        # Now let's build the current code block
        cc_current.code.append(dl)
        cc_current.add_block_data(variable, args)
        cc_current.is_subfunction = True
        # If we don't have any comment block, let's descend and get what's there...
        if cc_current.has_no_comments():
          ## Let's get comments from sub-function we're referencing (it's the best we can do...)
          sub_function_commentedcode = gather_comments_and_code(functions, base_function=function)
          if len(sub_function_commentedcode)>0:
            ccsub=sub_function_commentedcode[0]
            cc_current.input = ccsub.input    # likely a bit Meh
            cc_current.goal = ccsub.goal
            cc_current.output = ccsub.output  # likely a bit Meh
            cc_current.core_knowledge = ccsub.core_knowledge
            #cc_current.code=[dl]  # This is a single line of code from the caller
        arr.append(cc_current)
        cc_current, before_code = CommentedCode(), True  # Start the gathering process again...

  # Clean out the arr for non-commented, non-code blocks
  arr_clean=[]
  for cc in arr:
    if cc.has_no_comments() and len(cc.code)==0:
      continue
    # Adjust all the cc blocks for input/internal/output
    input, internal = cc.input_variables.copy(), cc.internal_variables
    # Other internal variables might also be output, but these have been generated 'for no purpose' otherwise
    cc.output_variables = internal - input   
    cc.input_variables = input - internal
    arr_clean.append(cc)

  return arr_clean


#def combine_refactored_and_original_code_comments(arr_refactored, arr_original):
#  pass

def check_for_empty_comments(code_with_comments):  # or raise ...
  uncommented_block=False
  for cc in code_with_comments:
    if cc.has_no_comments(): # len(cc.input+cc.output+cc.goal+cc.core_knowledge)==0:
      uncommented_block=True
      raise Exception(f"  Uncommented block: {cc}")
  if uncommented_block:
    pass # ...
  return 

import copy
def fix_empty_comments_if_possible(code_with_comments_refactored, code_with_comments_base):
  def get_variable_definition_index(variable, code_with_comments):
    for idx, cc in enumerate(code_with_comments):
      for def_line in cc.code:
        if def_line.variable == variable:
          return idx
    return None
  code_with_comments_expanded, fix_count = [], 0
  for cc in code_with_comments_refactored:
    if cc.has_no_comments() and len(cc.code)>0:
      # Try and find the relevant line(s) in code_with_comments_base
      variable_idx_set, variable_wanted_set = set(), set()
      for def_line in cc.code:
        variable_wanted_set.add(def_line.variable)
        variable_idx = get_variable_definition_index(def_line.variable, code_with_comments_base)
        if variable_idx:
          variable_idx_set.add(variable_idx)
      for variable_idx in sorted(variable_idx_set):
        # Go through these comment blocks and add the code lines that correspond to variables we need
        cc_new = copy.deepcopy( code_with_comments_base[variable_idx] )
        cc_new.code = [dl for dl in cc_new.code if dl.variable in variable_wanted_set ]
        code_with_comments_expanded.append(cc_new)
      fix_count+=1
    else:
      code_with_comments_expanded.append(cc)
  return code_with_comments_expanded, fix_count


def check_for_empty_input_and_output(code_with_comments):  # or raise ...
  # This likely happens if gemini thinks it is doing a subfunction
  no_valid_io=False
  for cc in code_with_comments:
    if len(cc.input_variables)==0:  # Not so terrible...
      pass
      #no_valid_io=True
      #raise Exception(f"  No valid input: {cc}")
    if len(cc.output_variables)==0:
      no_valid_io=True
      raise Exception(f"  No valid output: {cc}")
  if no_valid_io:
    pass # ...
  return 


def check_for_calls_to_subfunctions(functions, base_function='solver_virtual', debug=False):
  calls_subfunctions=False  
  for function in functions.keys():
    if function==base_function: continue # This one is fine to call subfunctions
    code_with_comments = gather_comments_and_code(functions, base_function=function, debug=False)
    for cc in code_with_comments:
      for def_line in cc.code:
        if def_line.function in functions.keys():
          calls_subfunctions=True
          raise Exception(f"  Subfunction '{function}' calls another subfunction from : {cc}")
  if calls_subfunctions:
    pass # ...
  return 
