import inspect
from vllm.model_executor.model_loader.reload.inplace import reload_weights
print(inspect.getsource(reload_weights))
