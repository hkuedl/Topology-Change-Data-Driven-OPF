# Topology-Change-Data-Driven-OPF
Codes for "A Two-Stage Approach for Topology Change-Aware Data-Driven OPF"
> This work proposed a two-stage approach for topology change-aware data-driven OPF. It consists of: 1) generating data-driven models using a topology transfer framework; and 2) ensembling well-trained models. In Stage 1, GPR is employed to capture the nonlinear correlation between the new and predicted OPF data. The new data is obtained by solving the OPF problem using traditional optimization solvers under the new topology; the predicted data is obtained by inputting the same power demand into the data-driven OPF model trained on one of the historical datasets. This framework allows us to obtain sample-efficient topology transfer models. In Stage 2, a dynamic ensemble learning strategy is developed, where the weights and the topology transfer models that need to be ensembled are dynamically determined. This strategy allows us to avoid obtaining biased OPF solutions from sub-models.

Authors: Yixiong Jia, Xian Wu, Zhifang Yang, and Yi Wang.

## Requirements
Python version: 3.8.10

The must-have packages can be installed by running
```
pip install requirements.txt
```

### Data
All the clean data for experiments are saved in ```Data/GFC12```. 

The row data can be found in ```Data/GFC12 row```.

You can also find the code for processing the data in this fold.

### Reproduction
If you want to run the proposed approach, you can run ```test.ipynb```.

## Citation
```
@article{jia2024two,
  title={A Two-Stage Approach for Topology Change-Aware Data-Driven OPF},
  author={Jia, Yixiong and Wu, Xian and Yang, Zhifang and Wang, Yi},
  journal={IEEE Transactions on Power Systems},
  year={2024},
  publisher={IEEE}
}
