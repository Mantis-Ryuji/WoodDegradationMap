# Numerical implementation notes

These notes clarify mixed precision, training loss, and inertia based on the source code of ChemoMAE 0.2.2 and its callers.
See the [experiment protocol](design/experiment_protocol.md) for settings and each run's artifacts for the execution environment and measured values.

<a id="training-precision"></a>

## 1. Mixed precision and training-loss records

**Arithmetic precision.**

Training uses CUDA AMP with FP16 autocast and a dynamic GradScaler. Model parameters and clean targets remain in FP32, and additional perturbations are computed in FP32 with autocast disabled. Encoder and decoder forward passes and the loss call are within the autocast context, but not every operation is performed in FP16.

The installed ChemoMAE 0.2.2 loss squares the difference between reconstruction and target, selects the elements indicated by the target mask, and averages them. The target is not cast to FP16. Subtraction between an FP16 reconstruction and an FP32 target promotes the result to FP32, so subtraction, squaring, and averaging follow an FP32 arithmetic path. This description is based on reading the caller and library source, not on measuring each tensor's dtype at runtime. Computing loss in FP32 does not remove the effects of numerical precision in FP16 forward operations, backpropagation, or the training trajectory. GradScaler scales the loss for backpropagation; the loss saved to the history is the value before scaling.

**History aggregation and interpretation.**

The recorded training loss is the sum of batch MSE values computed during parameter updates divided by the number of batches in the epoch. Except for the final partial batch, the batch size and number of target channels are fixed within a condition. This is neither a reevaluation of all training pixels using fixed final weights nor a loss on unseen specimens. AMP scale and the number of steps with skipped updates are recorded separately in the history.

## 2. Cluster-center and inertia records

If the sum of representations in a nonempty cluster becomes zero during center updates, the usual unit-center update is undefined.
The library's division protection alone does not establish a valid center. The project checks final centers for nonfinite values and zero or very small norms, recording an unsuccessful fit when a center is invalid.

The reference implementation's `reference_inertia` is a recorded value computed using assignments before the final center update. It is distinguished from `final_center_inertia`, which is recomputed from the saved final centers.
Distinguish the objective, mean cosine dissimilarity, from a value recorded at a particular point in the iterations.
See the [clustering specification](design/experiment_protocol.md#augmentation-clustering) for initialization, stopping criteria, and epsilon.

## 3. Implementation and execution-environment references

- [Model, extraction, and schedule](../src/wood_degradation_map/experiments/neural.py), [training](../src/wood_degradation_map/experiments/training.py), [clustering](../src/wood_degradation_map/experiments/clustering.py).
- [Configuration](../src/wood_degradation_map/experiments/config.py), [run-record conventions](experiment_runbook.md#artifact-records).
- Check each run's records for the GPU model, actual driver/CUDA/package versions, source hashes, and completion status. Dependency specifications alone do not establish that every run used the same environment.
