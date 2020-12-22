load_model with different model names to get test accuracies

enumeration of layers is broken because I always worked with layers
starting with the 2nd one, but enumerated them from 0.
This was unacceptable for the thesis, so there I use proper enumeration,
and try to adapt the code appropriately.

'params' folder includes all parameter configurations that I tried.

'named_params' folder includes parameter configurations for models used in the thesis.

Pre-trained models are in the v0.0.1 release. Put them into the 'pretrained_models' folder.