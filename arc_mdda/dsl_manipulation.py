import sys

# lines = [ 
#    (l[:l.find('#')].strip() if '#' in l else l) # Trim off later comments  
#    for l in [ 
#      s.strip() for s in definition.split('\n')  # Deal only with strip() lines
#    ] if not l.startswith('#') and len(l)>0      # Skip blanks and pure comments
#  ] 
class DefLine(object):
  variable, function, args, comment_for_llm = None, None, [], ""
  is_function_type, function_complexity=False,0
  uses, used_by = None, None
  def __init__(self, idx, txt, comment_for_llm):
    self.idx=idx
    self.txt=txt
    self.comment_for_llm=comment_for_llm
    self.uses, self.used_by = set(), set()
  def set_call(self, variable, function, args ):
    self.variable, self.function, self.args = variable, function, args
  def repr_fn(self):
    return f"{self.variable} = {self.function}({', '.join(self.args)})"
  def repr_base(self):
    return f"{self.idx:2d}[{self.function_complexity:1d}]: {self.repr_fn()}"
  def __repr__(self):
    return f"{self.repr_base(): <50s} : used_by:{self.used_by} .. uses:{self.uses}"
    
def analyse_definition(definition, dsl_functions, constants, dsl_functionals):
  #assert lines[0].startswith('def ')
  #assert 'return ' in lines[-1]
  # print(lines)
  def_lines=[]
  variables, calls = dict(I=0, ), set()  # variables is a dictionary name->idx defined
  idx=0
  for line in definition.split('\n')[:-2]:
    # Store off any '##' annotated comment (if there is one : expected to be infrequent) - it is directed towards LLM view
    comment_for_llm='' if '##' not in line else line[line.find('##')+2:].strip() 
    # Strip off all comments for next steps
    line = line[:line.find('#')].rstrip()  if '#' in line else line.rstrip()

    def_line = DefLine(idx, line, comment_for_llm)
    if idx==0 or len(line.lstrip())==0: # Skip blank lines (including pure LLM-comment lines, apparently)
      if idx==0:
        def_line.set_call( 'I', 'input_value', [])  # definition line might include LLM-comment
        def_lines.append(def_line)
        idx+=1
      continue

    variable, call = line.lstrip().split(' = ')  # Splits the function from the variable, but not the parameter '='
    function, args = call.split('(')
    
    assert variable not in dsl_functions
    assert variable not in variables  # Only define each variable once
    assert call not in calls          # List of functions used
    assert function in dsl_functions or function in variables
      
    #if '#' in args: # Strip off comments
    #  args=args[:args.find('#')]
    #args = args.strip()
    assert args[-1] == ')', f"'{args}' does not end in ')'"
    args = [args[:-1]] if ',' not in args else args[:-1].split(', ')
    
    uses = set()
    if function in variables:
      uses.add( variables[function] ) # The index of the def_line
      def_line.function_complexity+=2  # Using a self-defined function is 'worth' +2

    for arg in args:
      #print(f"\n'{arg}', {variables}, {dsl_interface}, {constants}")
      assert any([
          arg in variables, arg in dsl_functions, arg in constants, 
          arg=='I', arg in '0,1,2,3,4,5,6,7,8,9,10,-1,-2,True,False'
      ]), f"\n'{arg}', {variables}, {dsl_functions.keys()}, {constants}"
      
      if arg in variables:
        uses.add( variables[arg] )  # The index of the def_line
      #if arg in dsl_interface:
      #  def_line.function_complexity+=1
      def_line.function_complexity+=1  # Every argument is 'worth' +1
        
    variables[variable] = idx  # Only add this line's variable after tracing
    calls.add(call)

    def_line.set_call( variable, function, args )
    def_line.uses=uses
    def_line.is_function_type = (function in dsl_functionals)
    def_line.function_complexity+=1 # (since the line itself has a function call)
    
    def_lines.append(def_line)
    idx+=1

  # Add an extra def_line for the 'return O' statement at the end (different format from above)
  idx = len(def_lines)
  def_line = DefLine(idx, 'return O', '') # No LLM comment for this made-up line
  idx_ref = variables['O']
  def_line.set_call( '', 'return', ['O'] )  
  def_line.uses.add( idx_ref )  # The index of the def_line
  def_lines.append(def_line)
  
  for v in variables.keys():  #  This detects whether each variable gets mentioned...
    #print(f"{definition}, {v=}")
    assert sum([
        definition.count(vs) for vs in [
            f'({v})', f'({v}, ', f', {v})',
            f', {v}, ', f' {v} = ', f' {v}('
        ]
    ]) > 1 or v == 'O'

  # Get all the cross-references at once
  for idx, def_line in enumerate(def_lines):
    for idx_ref in def_line.uses:
      def_lines[idx_ref].used_by.add(idx)
  
  #for def_line in def_lines:
  #  print(def_line)
  return def_lines


#color_constants_zero   = 0
color_constants_zero   = 1010
color_constants_base   = [ 'COLOR_'+c for c in 'ZERO ONE TWO THREE FOUR FIVE SIX SEVEN EIGHT NINE ABOVE BELOW'.split(' ')]
color_constants_lookup = { k:i for i,k in enumerate(color_constants_base) }
color_constants_lookup['COLOR_BELOW']=-1

def create_virtual_solver(def_lines, color_constants=color_constants_base, dsl_functions=None, for_llm=False):
  variables, variable_lookup = ['I'], dict()
  block_end=True
  solver_lines=['def solver_virtual(I):']
  for idx, def_line in enumerate(def_lines[1:-1]):
    if for_llm and block_end: solver_lines.append('\n  # comment')

    #  f"{self.idx:2d}[{self.function_complexity:1d}]: {self.variable} = {self.function}({', '.join(self.args)})"
    variable = def_line.variable
    if variable!='O':
      variable = f"x{idx+1}"
      variable_lookup[def_line.variable] = variable  # We're remapping here...
      
    function = def_line.function
    if function in variable_lookup: # Remap the variables...
      function = variable_lookup[function]
      
    args = []
    for arg in def_line.args:
      if arg in variable_lookup: # Remap the variables...
        arg = variable_lookup[arg]
      if arg in color_constants_lookup:  # Remap the color table
        #print(arg, color_constants_lookup[arg])
        arg = color_constants[ color_constants_lookup[arg] ]
      args.append(str(arg))

    if function=='as_objects' and (for_llm or dsl_functions):  # Rewrite to have named variables for non-default values
      args_new=[args[0],]  # We'll rebuild with non-default value names
      if dsl_functions:
        args_new=[f"grid={args[0]}",]
      # Defaults chosen using `grep as_objects solvers.py | cut --delimiter=, -f 2- | sort | uniq -c`
      for i, (name, default) in enumerate([  # NB: Order is important, since we're stepping through args[position]
          ('each_object_single_color', True), 
          ('include_diagonal_neighbors', False), 
          ('discard_background', True), 
        ]):
        v = ('T' in args[i+1])  # True or False
        if v!=default:
          args_new.append(f'{name}={v}')
      args = args_new  # Replace with new version which has the non-default values named
    elif True and dsl_functions:  # Let's annotate the args with the known argument names!
      if function in dsl_functions: # i.e. it's not a user-defined function (we wouldn't know the arg_names in that case)
        arg_names = dsl_functions[function]  # Must exist!
        args_new=[]
        for i,arg in enumerate(args):
          args_new.append(f"{arg_names[i]}={arg}")
        args = args_new
    
    comment=''
    if for_llm and len(def_line.comment_for_llm)>0:
      comment=' # '+def_line.comment_for_llm
    
    solver_lines.append(f"  {variable} = {function}({', '.join(args)}){comment}")
    if def_line.is_function_type:
      block_end=False # Stop this being the end of a block...
    else:
      variables.append(variable)
      block_end=True  # Next line could be commented

  if for_llm:
    solver_lines.append(f'  return O')
  else:
    solver_lines.append(f'  return dict({",".join([f"{v}={v}" for v in variables])})')
  return '\n'.join(solver_lines)

"""
Grid = Tuple[Tuple[Color]]
Cell = Tuple[Color, IntegerTuple]
IntegerTuple = Tuple[Integer, Integer]
Color = NewType('Color', int)
Integer = int

#Boolean = bool
#Numerical = Union[Integer, IntegerTuple]
Object = FrozenSet[Cell]
IntegerSet = FrozenSet[Integer]
ColorSet = FrozenSet[Color]
Objects = FrozenSet[Object]
Indices = FrozenSet[IntegerTuple]
IndicesSet = FrozenSet[Indices]

Patch = Union[Object, Indices]
Element = Union[Object, Grid]
Piece = Union[Grid, Patch]

TupleTuple = Tuple[Tuple]
ContainerContainer = Container[Container]
"""

def detect_type(v):
  t=None
  if type(v) is bool: return 'Boolean'
  if type(v) is int:
    if v>1000: return 'Color'
    else:      return 'Integer'
  if type(v) is frozenset:
    if len(v)>0:
      u=next(iter(v))  # sample an entry
      type_u = detect_type(u)
      if type_u=='Cell':       return 'Object'
      elif type_u=='Integer':  return 'IntegerSet'
      elif type_u=='Color':    return 'ColorSet'
      elif type_u=='Object':   return 'Objects'
      elif type_u=='IntegerTuple': return 'Indices'
      elif type_u=='Indices':  return 'IndicesSet'
      else:
        return 'Set'  # [{type_u}]
    else:
      return 'Set'
  if str(type(v))=="<class 'function'>":
    # print(type(v), str(type(v)))
    return 'Function'  # Nothing much sensible to say...
  if type(v) is tuple:
    if type(v[0]) is tuple:
      if len(v[0])>0 and type(v[0][0]) is int and v[0][0]>1000:
        return 'Grid'
      return 'Tuple'  # Will recurse
    elif type(v[0]) is int:
      if v[0]>1000:  # First element of tuple is a Color
        if len(v)==2:
          if type(v[1]) is tuple:  # i.e. [Color, Tuple()]
            return 'Cell'
          else:
            return 'Tuple[Color, Color]'
        else:
          return 'Tuple'  # Will recurse to Tuple[Color...]
      else:
        return 'IntegerTuple'
    else:  # We have a tuple over other stuff
      return 'Tuple'
  #else:
  #  t=type(v)
  return t  


def render_variable(v, t, color_tokens):
  #ccz=arc_mdda.dsl_manipulation.color_constants_zero
  ccz, ct = color_constants_zero, color_tokens
  if t=='Grid':
    # This is a grid of COLORs, so need to do some translation magic
    #arr = ["Grid(C['"]
    arr = ["Grid("]
    for row in v:
      arr.append(' ' + ' '.join([ ct[c-ccz] for c in row ]) )
    #arr.append("'])")
    arr.append(")")
    return '\n'.join(arr)
  if t=='Integer' or t=='Boolean':
    return str(v)
  if t=='IntegerTuple':
    return f"IntegerTuple({','.join([str(e) for e in v])})"
  if t=='IntegerSet':
    return f"IntegerSet({','.join([str(e) for e in v])})"
  if t=='Color':
    #return f"C[' {ct[v-ccz]} ']"
    return f" {ct[v-ccz]}"
  if t=='Tuple[Color, Color]': # Super-rare
    #return f"Tuple[C[' {ct[v[0]-ccz]} {ct[v[1]-ccz]} ']]"
    return f"Tuple[ {ct[v[0]-ccz]} {ct[v[1]-ccz]}]"
  if t=='ColorSet':
    #return f"ColorSet[C[' {' ',join([ct[e-ccz] for e in v])} ']]"
    return f"ColorSet[ {' '.join([ct[e-ccz] for e in v])}]"
  if t=='Cell':
    #return f"(C[' {ct[v[0]-ccz]} '], ({','.join([str(e) for e in v[1]])})"
    return f"( {ct[v[0]-ccz]},({','.join([str(e) for e in v[1]])}))"
  if t=='Object':
    return f"Object({','.join([render_variable(e, 'Cell', ct) for e in v])})"
  if t=='Objects':
    return f"Objects(\n {',\n '.join([render_variable(e, 'Object', ct) for e in v])}\n)"
  if t=='Indices':
    arr = [f'({str(e[0])},{str(e[1])})' for e in v]
    return f"Indices({', '.join(arr)})"
  if t=='IndicesSet':
    return f"IndicesSet(\n {',\n '.join([render_variable(e, 'Indices', ct) for e in v])}\n)"
  if t=='Function':
    return f"function(...)"
  if t in '|Tuple|Set|':
    if len(v)==0: # We have no elements
      return f"{t}()"
    else:
      et=detect_type(v[0])
      if et is not None:
        if et in '|Color|Integer|Function|':  # These are short-ish, don't split line-wise
          return f"{t}({','.join([render_variable(e, et, ct) for e in v])})"
        return f"{t}(\n {',\n '.join([render_variable(e, et, ct) for e in v])}\n)"
    
  print(f"Unknown type : {t} for {v}")
  return str(v)  # Just return something!


#color_scheme_greenblatt_txt="black blue red green yellow grey pink orange purple brown"
color_scheme_greenblattish_txt="black blue red green yellow grey pink orange white brown"
color_scheme = color_scheme_greenblattish_txt.upper().split(' ')
color_scheme_addons=['ABOVE', 'BELOW',]

def get_constants_documentation(DSL_BASE='./External/arc-dsl'):
  arr=[]
  arr.append("## Color constants")
  #arr.append("There are is a flexible macro provided `C[' color']` that returns the Color ' color'")
  #arr.append(f"10 Color constants are useable: `C[' {' '.join(color_scheme)}']`")
  arr.append(f"10 Color constants are useable: ` {' '.join(color_scheme)}`")
  arr.append("The Color ` BELOW` is also defined, which is numerically below all other colors, so that `sort` operations can be performed with reliable results")
  #arr.append("The Color ` ABOVE` is defined, which is numerically above all other colors, so that `sort` operations can be performed with reliable results")
  arr.append("\n## Defined constants")
  arr.append("```python")
  with open(f"{DSL_BASE}/constants.py", 'rt') as constants_code:
    for l in constants_code.readlines():
      if '#' in l:  l = l[:l.find('#')] # Strip off comments
      l = l.rstrip()
      if len(l)==0: continue  # Skip empty and commented-out lines
      if l.startswith('COLOR') or l.startswith('from '): continue
      arr.append(l) # This is a good line
  arr.append("```\n")
  return arr

def get_dsl_documentation(DSL_BASE='./External/arc-dsl-llm'):
  arr=[]
  arr.append("## Useful functions with documentation")
  arr.append("```python")
  #arr.append("The following functions ...")
  with open(f"{DSL_BASE}/dsl.py", 'rt') as dsl_code:
    overload=False
    for l in dsl_code.readlines():
      if '#' in l:  l = l[:l.find('#')] # Strip off comments
      l = l.rstrip()
      if len(l)==0: continue  # Skip empty and commented-out lines
      if l.startswith('@overload'): 
        overload=True
        continue
      if l.startswith('def') and not overload:
        if ')' not in l:print(l)  # Should have full definition in single line
        arr.append(l) # This is a good line
      if l.strip().startswith('"""'): 
        arr.append(l) # This is a good line
      overload=False
  arr.append("```\n")
  return arr



grid_to_tuples = lambda g: tuple(tuple(c+color_constants_zero for c in r) for r in g)

exec_import_preamble="""
from arc_dsl.dsl import *
from arc_dsl.constants import *
"""

#color_constants_zero   = 1010
#color_constants_base   = [ 'COLOR_'+c for c in 'ZERO ONE TWO THREE FOUR FIVE SIX SEVEN EIGHT NINE ABOVE BELOW'.split(' ')]
#color_constants_lookup = { k:i for i,k in enumerate(color_constants_base) }
#color_constants_lookup['COLOR_BELOW']=-1

def get_color_map_preamble(arr):
  return (
    ",".join(arr)
    + " = "
    + ",".join([f"{i+color_constants_zero}" for i in range(-1,len(arr)-1)])
    + "\n"
  )
exec_color_map_preamble=get_color_map_preamble(['COLOR_BELOW'] + color_constants_base[:-1])

#color_scheme_greenblattish_txt="black blue red green yellow grey pink orange white brown"
#color_scheme = color_scheme_greenblattish_txt.upper().split(' ')
#color_scheme_addons=['ABOVE', 'BELOW',]
exec_color_map_preamble_llm=get_color_map_preamble(['BELOW'] + color_scheme)

def get_solver_function(solver_py, arc_dsl_ref,  # arc_dsl_ref is from `import arc_dsl`
                        exec_color_map_preamble=exec_color_map_preamble, 
                        function='solver_virtual'):  
  solver_globals_tmp = dict( arc_dsl=arc_dsl_ref )
  #solver_globals_tmp = { 'arc_dsl':arc_dsl, '__builtins__':None }  # Clearly we need some built-in functions

  # This defines `solver_virtual()` within solver_globals_tmp
  exec(exec_import_preamble + exec_color_map_preamble + solver_py, solver_globals_tmp) 

  return solver_globals_tmp[function]


#def definition_to_solver_callable():
#  def_lines = arc_mdda.dsl_manipulation.analyse_definition(definition, dsl_interface, constants, dsl_functionals)
#
#  solver_py = arc_mdda.dsl_manipulation.create_virtual_solver(def_lines, color_constants=color_constants_exec)
#  solver_py_llm = arc_mdda.dsl_manipulation.create_virtual_solver(def_lines, color_constants=color_constants_llm, for_llm=True)
#
#  solver_callable = arc_mdda.dsl_manipulation.get_solver_function(solver_py, arc_dsl)
#  return solver_callable

default_color_tokens = color_scheme + color_scheme_addons
def print_variable_for_llm(k, v, color_tokens=default_color_tokens):
  t=detect_type(v)
  return f"{k} = {render_variable(v,t,color_tokens)}"


# This allows us to encapsulate the arc_dsl module interface without bringing it into our main namespace
class DSL_Interface(object):
  def __init__(self, DSL_BASE='./External/arc-dsl-llm'):
    path = list(sys.path)
    sys.path.insert(0, DSL_BASE)
    try:
      # https://stackoverflow.com/questions/6861487/importing-modules-inside-python-class
      #self.arc_dsl = arc_dsl = __import__('arc_dsl')
      #self.arc_dsl_main = arc_dsl_main = __import__('arc_dsl.main')

      # https://stackoverflow.com/questions/9806963/how-to-use-the-import-function-to-import-a-name-from-a-submodule
      importlib = __import__('importlib')
      self.arc_dsl = arc_dsl = importlib.import_module('arc_dsl')
      self.arc_dsl_main = arc_dsl_main = importlib.import_module('arc_dsl.main')
    finally:
      sys.path[:] = path # restore
    #print(arc_dsl.__file__)
    print(arc_dsl_main.__file__)
    self.BASE = DSL_BASE
    self.functions, self.functionals = arc_dsl_main.get_functions(f"{DSL_BASE}/dsl.py")
    self.constants   = arc_dsl_main.get_constants(f"{DSL_BASE}/constants.py")
    self.definitions = arc_dsl_main.get_definitions(arc_dsl.solvers)


#