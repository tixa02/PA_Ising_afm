# PA_Ising_afm
The programs PAafm_ising are introduced in the scientific project #1346. This programm implements population annealing for AntiFerroMagnet Ising model.

To compile the programs, use Nvidia CUDA compiler (nvcc). For example: 

nvcc -o PA_Ising_afm PAafm_ising.cu

Optionally, the flags specifying GPU architecture of a particular device can be added. For example:

nvcc -arch=sm_35 -o PA_Ising_afm PAafm_ising.cu
