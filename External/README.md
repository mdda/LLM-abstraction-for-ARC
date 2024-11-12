# LLM-abstraction-for-ARC / External

## Get external libraries/data

(These instructions are also in this repo's top-level `./README.md`)

```bash
git clone git@github.com:fchollet/ARC-AGI.git
# Data is in ./External/ARC-AGI/data/{training,evaluation} (400 each)

git clone git@github.com:mdda/arc-dsl-llm.git
# Repo is in ./External/arc-dsl-llm/*
#   NB: it includes a sneaky internal link to arc_dsl to 'modularise' it
```