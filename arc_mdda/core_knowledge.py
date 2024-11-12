# LLM-friendly version

## Priors from the ARC paper
## III.1.2 Core Knowledge priors = page=46.62

prelude="""
# ARC Challenge problems
Each problem in the ARC Challenge requires understanding the way in which several "input grids" can be transformed into corresponding "output grids".  
Several demonstration pairs are shown, and the solution involves describing how an unknown "output grid" can be derived from the given test "input grid".
To do this, we will be doing extensive code analysis.
""".lstrip()

description_v0_OLD="""
## Core Knowledge
The transformations required for any given problem make use of Core Knowledge concepts :

* **Object cohesion**:
  + Ability to parse grids :
    - get "objects" based on color continuity or spatial contiguity
    - split into into zones, partitions (which may be grids themselves)
* **Object persistence**:
  + Objects are assumed to persist despite the presence of noise or occlusion by other objects.
    - In many cases, objects from the input grid persist on the output grid, but in a transformed form.
* **Object influence via contact**: 
  + Many problems feature physical contact between objects
    - e.g. one object being translated until it is in contact with another
    - e.g. a line might "grow" until it "rebounds" against another object.
* **Basic Geometry and Topology priors**:
  + ARC problems feature a range of elementary geometry and topology concepts, in particular:
    - Lines, and rectangular shapes.  Other objects that occur are more likely to be simple shapes than complex shapes.
    - Symmetries such as rotations, translations.
    - Shape upscaling or downscaling, overall or horizontal / vertical.
    - Containing / being contained / being inside or outside of a perimeter. Corners of perimeters.
    - Drawing lines, connecting points, orthogonal projections.
    - Copying, repeating objects.
* **Numbers and Counting priors**:
  + Many ARC problems involve counting or sorting objects and/or comparing numbers:
    - e.g. which shape or symbol appears most / least / same number of times?
    - Which object is the largest / smallest?
    - Which objects are the same size / color?
    - Or repeating a pattern a number of times (possibly depending on other properties).
  + Addition and subtraction is also possible, although all quantities featured are smaller than (say) 10.
* **Goal-directedness prior**:
  + Many of the input/output grids might be effectively modeled as being the starting and end states of a process that involves intentionality
  + e.g. A problem might combines the concepts of "line extrapolation", "turning upon hitting an obstacle", and "efficiently reaching a goal"
""".lstrip()

combinations_v0_OLD="""
These transformations may depend on each other, so that (for instance) only objects of the same shape as others in the input grid are reflected in the output grid.  
Making use of this *compositionality* is a key part of success in creating good solutions to a problem.
""".lstrip()

# Taking Gemini's advice
description="""
## Core Knowledge
Solving ARC problems requires understanding and applying Core Knowledge concepts relating to spatial reasoning, object manipulation, and basic mathematical principles. These concepts include:

* **Object cohesion**:
  + Ability to parse grids :
    - identifying distinct objects within the grid based on properties like:
      + For instance: color continuity, spatial contiguity, repeated patterns, or symmetries
    - segmenting the grid into zones or partitions, which can be treated as sub-grids
      + For instance: dividing a grid with delineated quadrants into separate, potentially inter-related sub-grids                   
* **Object persistence**:
  + Objects are assumed to persist despite the presence of noise or occlusion by other objects
    - For example, if a square is partially covered by a triangle, the solver should still recognize the underlying square
    - While generally true, there are cases where objects might disappear or transform significantly
  + In many cases, objects from the input grid persist on the output grid, but in a transformed form but in a transformed form (e.g., rotated, scaled, or recolored)
* **Object influence via contact**: 
  + Many problems feature physical contact between objects
    - For instance: one object being translated until it is in contact with another
    - Other examples: a line extending until it touches another shape; objects snapping to a grid; or an object being 'pushed' by another
* **Basic Geometry and Topology priors**:
  + Geometric and topological reasoning is crucial. Commonly encountered concepts include:
    - Shapes: Lines, rectangles and simple shapes;  Other objects that occur are likely to have simple motifs
    - Transformations: rotation, translation, mirroring, flipping, scaling (overall or horizontal/vertical)
    - Relationships: Containing/contained, inside/outside perimeter, corners, parallel lines, topological connectedness, set relationships (inclusion, intersection, disjointness).
    - Actions: Drawing lines, connecting points, orthogonal projections, copying, repeating objects
    - Self-similarity via symmetries such as rotations and mirroring
* **Numbers and Counting priors**:
  + Many ARC problems involve counting or sorting objects and/or comparing numbers, for instance:
    - Which shape or symbol appears most / least / same number of times?
    - Which object is the largest / smallest?
    - Which objects are the same size / color?
  + Similarly actions being taken might depend on counting and/or comparing numbers
    - For example: Repeating a single shape a number of times depending on the number of different shapes present
  + Simple arithmetic operations (addition, subtraction, multiplication, division), although all quantities featured will be small integers less than (say) 10
* **Goal-directedness prior**:
  + Many ARC problems can be interpreted as depicting a sequence of actions with a specific goal
  + For instance: 
    - A problem might combines the concepts of "line extrapolation", "turning upon hitting an obstacle", and "efficiently reaching a goal"
    - Arranging objects to fill a container or constructing a symmetrical pattern
  + Some ARC problems might imply a need for planning or simulating steps towards a solution
* **Compositionality**:
  + Successfully solving ARC problems often requires chaining the above concepts together
    - For instance: First identifying simply connected components (cohesion), then counting them (numerical), and finally replicating the largest component multiple times side-by-side (geometry)
    - For instance: First grouping shapes by color (cohesion and color), sorting them by size (numerical), recoloring the most frequent (numerical and color), and reflecting it across a vertical axis (geometry and symmetry)
""".lstrip()


