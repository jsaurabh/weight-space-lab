import inspect
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.gpu_worker import Worker
print(inspect.getsource(GPUModelRunner.reload_weights))
print(inspect.getsource(Worker.compute_weight_checksums))
