Use load_model.py with different model names to get test accuracies.

Enumeration of layers is broken because I always worked with layers
starting with the 2nd one, but enumerated them from 0.
This was unacceptable for the thesis, so there I use proper enumeration,
and try to adapt the code appropriately.

'params' folder includes all parameter configurations that I tried.

'named_params' folder includes parameter configurations for models used in the thesis.

Pre-trained models are in the v0.0.1 release. Put them into the 'pretrained_models' folder.

collect_activations.py was a huge file for a whole bunch of experiments, like GANs, MWDs, assessing binary separability of activations, and so on.
I extracted MWD stuff to calculate_wasser_dists.py, and grad+MWD stuff to correlate_wasser_dists_gradients.py

Approximate list of packages needed for the repo:
```
 conda install click
 conda install -c conda-forge matplotlib
 conda install imageio
 conda install scikit-image
 conda install scikit-learn
 conda install pandas
 conda install -c conda-forge fuzzywuzzy
 conda install -c conda-forge tensorboardx
 conda install kornia
 conda install graphviz
 conda install python-graphviz
 conda install sortedcontainers
```

Repo also includes 3 conda environments exported to .yml files. 
One is for GANs and includes keras. 
Other two are for everything else, conda_env_prune.yml is a newer version that should work,
but in case it doesn't conda_env_main_old.yml can be tried.