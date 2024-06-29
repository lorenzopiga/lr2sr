# lr2sr
A modified trained super-resolution model for N-body cosmological simulation.

The model super-resolves the output of N-body cosmological simulation. It generates 512 times more tracer particles and produces their full phase-space (displacment+velocity) distribution.

For more detailed description, check the work [Paper-II](https://doi.org/10.1093/mnras/stab2113).

## Model

`G_z0.pt` and `G_z2.pt` are the trained SR models to super-resolve the N-body cosmological simulation at z=2 and z=0 separately. They are the models used in [Paper-II](https://doi.org/10.1093/mnras/stab2113)

For details of the model archetecture and training, check repository [map2map](https://github.com/eelregit/map2map), also [Paper-I](https://www.pnas.org/content/118/19/e2022038118) and [Paper-II](https://doi.org/10.1093/mnras/stab2113).


## Usage

We focus on the production of Lambda-CDM SR simulations from the DEMNUni suite. The goal has been to understand the required RAM necessary to handle the SR versions in the current public code and how to modify it for large LR simulations as the DEMNUni are. The idea is to save to disk the different SR chunks coming from the application of the GAN model and save them rather than building the full image of the LR sims, which would require an incredibly large RAM. 





