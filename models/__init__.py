import os

from transformers import AutoTokenizer, AutoModelForCausalLM

from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.auto.modeling_auto import MODEL_MAPPING, MODEL_FOR_CAUSAL_LM_MAPPING

from .llada.model_llada import LLaDAConfig, LLaDAModel, LLaDAForCausalLM

CONFIG_MAPPING.register('llada', LLaDAConfig, True)
MODEL_MAPPING.register(LLaDAConfig, LLaDAModel, True)
MODEL_FOR_CAUSAL_LM_MAPPING.register(LLaDAConfig, LLaDAForCausalLM, True)