## LEGO: LLM-based Evaluation and Guided Optimization for Adaptive Algorithm Design

### Contact

We welcome academic and business collaborations with funding support. For more details, please contact us via email at xuhua@tsinghua.edu.cn.

### Overview

This release contains the core processes of the **LEGO** framework, as described in the paper *LEGO: LLM-based Evaluation and Guided Optimization for Adaptive Algorithm Design*. The provided code implements the main components of the approach, covering main streamline, optimization, evaluation. 

Here is a brief overview of the provided files. More detailed documentation is available within each file:

- **fake_generate/**: This directory contains codes to generate fake problem instances of 4 types: IS, MKS, SC, MVC.
- **generate_algorithms/**: This directory contains codes as examples of generated complete algorithms.

- **calc/**: This directory is an outside repository, containing codes to calculate hypervolume effectively.(using hbda algorithm)
- **optimize/**: This directory contains codes for hyper-parameters searching.
- **run_eval/**: This directory contains codes for running all algorithm combinations and evaluating their performances.
- **main_streamline/**: This directory contains codes as the main streamline, including all layers' structure, components library and the codes to run a complete algorithm on a particular instance.
- **Model/**, **Dataset/**, **logs/**: These directories have nothing but will be used when main streamline or evaluation working.

### Requirements

The required environment is specified in the provided TXT file `requirements.txt`.
