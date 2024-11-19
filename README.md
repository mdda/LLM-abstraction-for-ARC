# LLM-abstraction-for-ARC

This repo is a copy of the relevant code for the dataset paper 
["Capturing Sparks of Abstraction for the ARC Challenge"](https://arxiv.org/abs/2411.11206).

* Quick-start : Have a look at the Jupyter Notebooks (pre-rendered) in `./notebooks`

## Assets made available (Apache 2.0 license)

* The `arc_mdda` module is ARC-related code in modularised form, mostly built by first testing within notebooks
* The Gemini-Flash-002 generated 'Sparks of Abstraction' dataset is downloadable via [this link](https://drive.google.com/file/d/1o-QT_erDAT5Ns6WqWosr8sqy6a_HLcpX/view?usp=sharing)
* The instructions below include the incorporation of the related repo [`arc-dsl-llm`](https://github.com/mdda/arc-dsl-llm)

## Get external libraries/data

```bash
mkdir -p ./External

pushd External
git clone git@github.com:fchollet/ARC-AGI.git
# Data is in ./External/ARC-AGI/data/{training,evaluation} (400 each)
popd

pushd External
git clone git@github.com:mdda/arc-dsl-llm.git
# Repo is in ./External/arc-dsl-llm/*
#   NB: it includes a sneaky internal link to arc_dsl to 'modularise' it
popd
```

## Using the Gemini-LLM

The Gemini-Flash-002 model is used via `arc_mdda/models/gemini.py`, 
and will use (by default) the VertexAI credentials you provide in `./key-vertexai-iam.json`

```bash
export GOOGLE_APPLICATION_CREDENTIALS="./key-vertexai-iam.json"
```

The code also allows for usage of the $FREE Gemini API 
(for which you'll need to add a `free=True` flag to the `get_model()` calls).


## Library installation

```bash
uv pip install jupytext requests frozenlist                 # Basics
uv pip install vertexai google-generativeai omegaconf       # LLM access
uv pip install llvmlite umap-learn hdbscan                  # visualisation
uv pip install matplotlib pandas datashader bokeh holoviews # visualisation
```


## Examining / Running the notebooks

* NB: To just have a look at the notebook outputs, see : `./notebooks/*.ipynb` (as expected)

`jupytext` has been used within JupyterLab for the notebooks : This means that the actual saved-to-github 
code is the the `.py` files in the main directory, which should be run in JupyterLab (say) using the 
`jupytext` plugin, and choosing `Open as Notebook` on the `.py` file.

The local notebook contents is stored to `cache-notebooks`, and not checked into the repo.  i.e. the following was done:
```bash
jupytext --set-formats cache-notebooks//ipynb,py XYZ.py
```

## Citing this work

If you find this helpful in your research, please consider citing: 

```bibtex
@misc{andrews2024capturingsparksabstractionarc,
      title={Capturing Sparks of Abstraction for the ARC Challenge}, 
      author={Martin Andrews},
      year={2024},
      eprint={2411.11206},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2411.11206}, 
}
```


### Acknowledgements

Support for this research was provided by the Google AI/ML Developer Programs team,
including access to the Gemini models and GPUs on Google Cloud Platform.