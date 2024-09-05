from src.IGRtrain import IGRTrainer
from src.IGRsample import IGRSampler
from src.config import IGRConfig
import torch
import gc

# No longer used due to GPU memory issue

class IGRiterator:
    def __init__(self, config_path, feedback_content):
        self.config_path = config_path
        self.feedback_content = feedback_content
        self.config = IGRConfig()

    def run(self):
        for _ in range(self.config.MAX_ITER):
            
            # Sample & Save | Convert to subprocess.run() to avoid GPU memory release issue
            sampler = IGRSampler(self.feedback_content, slm_model=self.config.BASE_MODEL, blm_model=self.config.HELPER_MODEL)
            sampler.process_prompts(self.config.current_iteration)
            
            # Train & Save | debugging | Convert to subprocess.run() to avoid GPU memory release issue
            trainer = IGRTrainer(self.config)
            trainer.run(self.config_path, self.feedback_content)
            
            self.config.next_iter()

    def clear_gpu_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def print_gpu_memory_usage(self):
        if torch.cuda.is_available():
            print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")