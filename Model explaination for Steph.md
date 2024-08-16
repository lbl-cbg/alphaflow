Model explaination for Steph

The main model is defined in the new_model.py
The diffusion part is defined in the diffusion.py

The diffusion part is directly applied to the batch and is inside the same pylighting module but defined as two pytorch model
In the SAXS intergreted branch I merged them together.

Those two are the most important files and I modified the train.py and predict.py and other files to remove unnecessary code from AlphaFlow.