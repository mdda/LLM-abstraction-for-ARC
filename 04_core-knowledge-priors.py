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

# ## Priors from the [ARC paper](https://arxiv.org/abs/1911.01547)
# #### III.1.2 Core Knowledge priors = page=46.62
#
# * Idea : This has to be comprehensive, since Chollet claims it is the basis of ARC itself

# * a. **Objectness priors**:
#   + *Object cohesion:* Ability to parse grids into “objects” based on continuity criteria including:
#     - color continuity or spatial contiguity (figure 5)
#       + _Figure 5: Left, objects defined by spatial contiguity. Right, objects defined by color continuity._
#     - ability to parse grids into zones, partitions.
#   + *Object persistence:* Objects are assumed to persist despite :
#     - the presence of noise (figure 6)
#       + _Figure 6: A denoising task._
#     - occlusion by other objects.
#     - In many cases (but not all) objects from the input persist on the output grid, often in a transformed form.
#       + Common geometric transformations of objects are covered in category (d), “basic geometry and topology priors”.
#   + *Object influence via contact:* Many tasks feature physical contact between objects
#     - e.g. one object being translated until it is in contact with another (figure 7),
#       + _Figure 7: The red object “moves” towards the blue object until “contact”._
#     - or a line “growing” until it “rebounds” against another object (figure 8).
#       + _Figure 8: A task where the implicit goal is to extrapolate a diagonal line that “rebounds” upon contact with a red obstacle._
# * b. **Goal-directedness prior**:
#   + While ARC does not feature the concept of time, many of the input/output grids can be effectively modeled by humans as being the starting and end states of a process that involves intentionality (e.g. figure 9).
#     - _Figure 9: A task that combines the concepts of “line extrapolation”, “turning on obstacle”, and “efficiently reaching a goal” (the actual task has more demonstration pairs than these three)._
#   + As such, the goal-directedness prior may not be strictly necessary to solve ARC, but it is likely to be useful.
# * c. **Numbers and Counting priors**:
#   + Many ARC tasks involve counting or sorting objects (e.g. sorting by size), comparing numbers
#     - (e.g. which shape or symbol appears the most (e.g. figure 10)?
#       + _Figure 10: A task where the implicit goal is to count unique objects and select the object that appears the most times (the actual task has more demonstration pairs than these three)._
#     - The least?
#     - The same number of times?
#     - Which is the largest object?
#     - The smallest?
#     - Which objects are the same size?),
#   + or repeating a pattern for a fixed number of time.
#   + The notions of addition and subtraction are also featured (as they are part of the Core Knowledge number system as per [85]).
#   + All quantities featured in ARC are smaller than approximately 10.
# * d. **Basic Geometry and Topology priors**:
#   + ARC tasks feature a range of elementary geometry and topology concepts, in particular:
#     - Lines, rectangular shapes (regular shapes are more likely to appear than complex shapes).
#     - Symmetries (e.g. figure 11), rotations, translations.
#       * _Figure 11: Drawing the symmetrized version of a shape around a marker. Many tasks involve some form of symmetry._
#     - Shape upscaling or downscaling, elastic distortions.
#     - Containing / being contained / being inside or outside of a perimeter.
#     - Drawing lines, connecting points, orthogonal projections.
#     - Copying, repeating objects.
#   

# ## LLM-ready version

# %load_ext autoreload
# %autoreload 2

import arc_mdda.core_knowledge

print(arc_mdda.core_knowledge.prelude)

# +
#print(arc_mdda.core_knowledge.output_format)
# -

print(arc_mdda.core_knowledge.description)

# +
#print(arc_mdda.core_knowledge.combinations)
# -


