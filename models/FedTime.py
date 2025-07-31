import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding
import math


class Model(nn.Module):
    """
    FedTime: A Federated Large Language Model for Long-Term Time Series Forecasting
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = 'long_term_forecast'
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.d_model = configs.d_model
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        self.use_peft = configs.use_peft
        self.peft_method = configs.peft_method
        self.use_dpo = configs.use_dpo
        self.revin = configs.revin
        
        # Calculate number of patches
        if configs.padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
            self.patch_num = int((self.seq_len + self.stride - self.patch_len) / self.stride + 1)
        else:
            self.padding_patch_layer = None
            self.patch_num = int((self.seq_len - self.patch_len) / self.stride + 1)
        
        # RevIN normalization
        if self.revin:
            self.revin_layer = RevIN(self.enc_in)
        
        # Patch embedding
        self.patch_embedding = PatchEmbedding(
            self.patch_len, self.patch_num, self.d_model, 
            configs.patch_len, configs.stride
        )
        
        # LLaMA backbone
        self.llm_config = LlamaConfig(
            vocab_size=1000,  # Dummy vocab size
            hidden_size=configs.llm_dim,
            intermediate_size=configs.llm_dim * 4,
            num_hidden_layers=configs.llm_layers,
            num_attention_heads=configs.llm_dim // 64,
            max_position_embeddings=self.patch_num + 10,
            rms_norm_eps=1e-6,
            tie_word_embeddings=False,
        )
        
        self.llm_model = LlamaModel(self.llm_config)
        
        # PEFT setup
        if self.use_peft:
            if self.peft_method == 'lora' or self.peft_method == 'qlora':
                peft_config = LoraConfig(
                    task_type=TaskType.FEATURE_EXTRACTION,
                    inference_mode=False,
                    r=configs.lora_r,
                    lora_alpha=configs.lora_alpha,
                    lora_dropout=configs.lora_dropout,
                    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", 
                                   "gate_proj", "up_proj", "down_proj"]
                )
                self.llm_model = get_peft_model(self.llm_model, peft_config)
        
        # Input projection to LLM dimension
        self.input_projection = nn.Linear(self.d_model, configs.llm_dim)
        
        # Output projection layers
        self.output_projection = nn.Linear(configs.llm_dim, self.d_model)
        self.prediction_head = nn.Linear(self.d_model * self.patch_num, self.pred_len)
        
        # Channel independence
        self.individual = configs.individual
        if self.individual:
            self.prediction_heads = nn.ModuleList([
                nn.Linear(self.d_model * self.patch_num, self.pred_len) 
                for _ in range(self.enc_in)
            ])
        
        self.dropout = nn.Dropout(configs.dropout)
        
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        """
        Forward pass for FedTime model
        
        Args:
            x_enc: Input time series [B, L, D]
            
        Returns:
            Output predictions [B, T, D]
        """
        # Handle channel independence
        if self.individual:
            return self._forward_individual(x_enc)
        else:
            return self._forward_shared(x_enc)
    
    def _forward_individual(self, x_enc):
        """Forward pass with individual channel processing"""
        B, L, D = x_enc.shape
        outputs = []
        
        for i in range(D):
            # Process each channel independently
            x_channel = x_enc[:, :, i:i+1]  # [B, L, 1]
            
            # RevIN normalization
            if self.revin:
                x_channel = self.revin_layer(x_channel, 'norm')
            
            # Patching and embedding
            x_patches = self._create_patches(x_channel)  # [B, N, P]
            x_embed = self.patch_embedding(x_patches)    # [B, N, d_model]
            
            # Project to LLM dimension
            x_llm = self.input_projection(x_embed)       # [B, N, llm_dim]
            
            # LLM processing
            llm_output = self.llm_model(inputs_embeds=x_llm)
            hidden_states = llm_output.last_hidden_state # [B, N, llm_dim]
            
            # Project back to model dimension
            x_out = self.output_projection(hidden_states) # [B, N, d_model]
            x_out = self.dropout(x_out)
            
            # Flatten and predict
            x_flat = x_out.reshape(B, -1)                # [B, N*d_model]
            pred = self.prediction_heads[i](x_flat)      # [B, pred_len]
            
            # RevIN denormalization
            if self.revin:
                pred = pred.unsqueeze(-1)                # [B, pred_len, 1]
                pred = self.revin_layer(pred, 'denorm')
                pred = pred.squeeze(-1)                  # [B, pred_len]
            
            outputs.append(pred)
        
        # Stack channel outputs
        output = torch.stack(outputs, dim=-1)            # [B, pred_len, D]
        return output
    
    def _forward_shared(self, x_enc):
        """Forward pass with shared processing across channels"""
        B, L, D = x_enc.shape
        
        # RevIN normalization
        if self.revin:
            x_enc = self.revin_layer(x_enc, 'norm')
        
        # Process all channels together
        x_patches = self._create_patches(x_enc)          # [B*D, N, P]
        x_embed = self.patch_embedding(x_patches)        # [B*D, N, d_model]
        
        # Project to LLM dimension
        x_llm = self.input_projection(x_embed)           # [B*D, N, llm_dim]
        
        # LLM processing
        llm_output = self.llm_model(inputs_embeds=x_llm)
        hidden_states = llm_output.last_hidden_state     # [B*D, N, llm_dim]
        
        # Project back to model dimension
        x_out = self.output_projection(hidden_states)    # [B*D, N, d_model]
        x_out = self.dropout(x_out)
        
        # Reshape and predict
        x_flat = x_out.reshape(B * D, -1)                # [B*D, N*d_model]
        pred = self.prediction_head(x_flat)              # [B*D, pred_len]
        pred = pred.reshape(B, D, self.pred_len)         # [B, D, pred_len]
        pred = pred.transpose(1, 2)                      # [B, pred_len, D]
        
        # RevIN denormalization
        if self.revin:
            pred = self.revin_layer(pred, 'denorm')
        
        return pred
    
    def _create_patches(self, x):
        """Create patches from time series data"""
        B, L, D = x.shape
        
        # Padding if needed
        if self.padding_patch_layer is not None:
            x = self.padding_patch_layer(x.permute(0, 2, 1)).permute(0, 2, 1)
        
        # Reshape for channel independence processing
        x = x.reshape(B * D, L)  # [B*D, L]
        
        # Create patches
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        patches = patches.transpose(1, 2)  # [B*D, N, P]
        
        return patches


class RevIN(nn.Module):
    """
    Reversible Instance Normalization for time series forecasting
    """
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode='norm'):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x


class DPOLoss(nn.Module):
    """
    Direct Preference Optimization Loss for model alignment
    """
    def __init__(self, beta=0.1):
        super(DPOLoss, self).__init__()
        self.beta = beta
        
    def forward(self, policy_chosen_logps, policy_rejected_logps, 
                reference_chosen_logps, reference_rejected_logps):
        """
        Compute DPO loss
        
        Args:
            policy_chosen_logps: Log probabilities of chosen responses from policy model
            policy_rejected_logps: Log probabilities of rejected responses from policy model
            reference_chosen_logps: Log probabilities of chosen responses from reference model
            reference_rejected_logps: Log probabilities of rejected responses from reference model
        """
        policy_logratios = policy_chosen_logps - policy_rejected_logps
        reference_logratios = reference_chosen_logps - reference_rejected_logps
        
        logits = self.beta * (policy_logratios - reference_logratios)
        loss = -F.logsigmoid(logits).mean()
        
        return loss
