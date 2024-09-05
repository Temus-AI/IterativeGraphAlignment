# HF Token | API Keys and others 
import os 
HF_TOKEN = "hf_****"
ANTHROPIC_API_KEY = "sk-ant-****"
OPENAI_API_KEY = "sk-proj-****"
COHERE_API_KEY = "****"
OPENROUTER_API_KEY = "sk-or-****"


def set_env():
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    os.environ["COHERE_API_KEY"] = COHERE_API_KEY
    os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY
    # HF Login
    from huggingface_hub import login
    login(os.getenv("HF_TOKEN"))

from dataclasses import dataclass, field
from typing import List

@dataclass
class IGRConfig:
    _BASE_MODEL: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    HELPER_MODEL: List[str] = field(default_factory=lambda: ["hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4", "Qwen/Qwen2-72B-Instruct-GPTQ-Int4", "microsoft/Phi-3-medium-4k-instruct", "mistralai/Mistral-Nemo-Instruct-2407"])
    MAX_ITER: int = 4
    HF_USERNAME: str = "Ksgk-fy"
    ID: str = ""
    HF_TOKEN: str = os.getenv("HF_TOKEN")
    current_iteration: int = 1
    set_env()
    
    @property
    def model_str(self):
        return self._BASE_MODEL.split("/")[-1]
    
    @property
    def adaptor_output_dir(self):
        return f"IGR{self.ID}-Adaptor-{self.model_str}-{self.current_iteration}"
    
    @property 
    def sft_adaptor_output_dir(self):
        return f"SFT{self.ID}-Adaptor-{self.model_str}"
    
    @property 
    def star_adaptor_output_dir(self):
        return f"STAR{self.ID}-Adaptor-{self.model_str}-{self.current_iteration}"
    
    @property 
    def adaptor_id(self):
        return f"{self.HF_USERNAME}/IGR{self.ID}-Adaptor-{self.model_str}-{self.current_iteration}"
    
    @property
    def sft_adaptor_id(self):
        return f"{self.HF_USERNAME}/SFT{self.ID}-Adaptor-{self.model_str}"
    
    @property
    def star_adaptor_id(self):
        return f"{self.HF_USERNAME}/STAR{self.ID}-Adaptor-{self.model_str}-{self.current_iteration}"
    
    @property
    def model_id(self):
        return f"{self.HF_USERNAME}/IGR{self.ID}-Model-{self.model_str}-{self.current_iteration}"
    
    @model_id.setter
    def model_id(self, new_id):
        self._model_id = new_id
    
    def reset_model_id(self, new_id):
        self.model_id = new_id
    
    @property 
    def sft_model_id(self):
        return f"{self.HF_USERNAME}/SFT{self.ID}-Model-{self.model_str}"
    
    @property
    def star_model_id(self):
        return f"{self.HF_USERNAME}/STAR{self.ID}-Model-{self.model_str}-{self.current_iteration}"
    
    def next_iter(self):
        print(f"-------- Moving to the next iteration: {self.current_iteration + 1} --------")
        self.current_iteration += 1
        
    @property
    def BASE_MODEL(self):
        if self.current_iteration == 1:
            return self._BASE_MODEL
        else:
            # check if such model exists on huggingface 
            from huggingface_hub import list_models
            
            model_name = f"{self.HF_USERNAME}/IGR{self.ID}-Model-{self.model_str}-{self.current_iteration-1}"
            available_models = list_models(author=self.HF_USERNAME)
            
            if model_name not in [model.modelId for model in available_models]:
                raise ValueError(f"Model {model_name} not found on Hugging Face. Previous IGR iteration fails to save the model")
                # print(f"Warning: Model {model_name} not found on Hugging Face. Using the original BASE_MODEL.")
                # return self._BASE_MODEL
            
            return f"{self.HF_USERNAME}/IGR{self.ID}-Model-{self.model_str}-{self.current_iteration-1}"
        
    @property 
    def BASE_STAR_MODEL(self):
        if self.current_iteration == 1:
            return self._BASE_MODEL
        else:
            # check if such model exists on huggingface 
            from huggingface_hub import list_models
            
            model_name = f"{self.HF_USERNAME}/STAR{self.ID}-Model-{self.model_str}-{self.current_iteration-1}"
            available_models = list_models(author=self.HF_USERNAME)
            
            if model_name not in [model.modelId for model in available_models]:
                raise ValueError(f"Model {model_name} not found on Hugging Face. Previous IGR iteration fails to save the model")
                # print(f"Warning: Model {model_name} not found on Hugging Face. Using the original BASE_MODEL.")
                # return self._BASE_MODEL
            
            return f"{self.HF_USERNAME}/STAR{self.ID}-Model-{self.model_str}-{self.current_iteration-1}"
        
    @property 
    def BASE_SFT_MODEL(self):
        return self.sft_model_id