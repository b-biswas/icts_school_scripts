# ICTS BigDataCosmo School

This repository summarizes the work done by group [P5](https://www.icts.res.in/sites/default/files/seminar%20doc%20files/Project%205.pdf)  at ICTS BigDataCosmo School:

### Members:
Animesh Sah, Anoma Ganguly, Biswajit Biswas, Purba Mukherjee, Sankarshana Srinivasan & Souvik Jana\
### Mentors:
Elisabeth Krause & Supranta S. Boruah

## Outline
During the project, the group undertook the following tasks:\
**Step 1**
Generate data vectors
[Install cocoa](https://docs.google.com/document/d/1n7iJJnyID-e7expuR-ebINyCVnE-cKrZh8D-L2wjulk/edit)
The cocoa folder in this repository contains the scripts to generate data vectors (dv) for some randomly chosen set of cosmological parameters close to the fiducial. 

**Step 2**
Preprocess the data: 
We first remove the dv with very large chi^2 compared to the fiducial (these would most likely correspond to the unphysical models) 
Then we normalize the remaining data points (for eg. between 0 and 1) so that they can be fed to train a Neural Network. 

**Step 3**
Train the emulator on generated data vectors. 
We want to run an MCMC chain to contain cosmology with the dv computed in DES Y3. 
Computing the high dimensional integrals at each MCMC step can be computationally very expensive.
So we train an emulator to approximate this computation at each MCMC step. 

**Step 4**
Run MCMC chains. 
Putting everything together, we now need to run an MCMC chain using the emulator to constrain cosmology. 

