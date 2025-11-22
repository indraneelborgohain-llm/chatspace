from training.data_loader import train_loader,val_loader
from architecture.gptoss import Transformer,ModelConfig
import torch
from inference import generate_text
from training.trainer import trainer
device= "cuda:0"
context="Once upon a day"
model= Transformer(ModelConfig(
    num_attention_heads=8
    ,num_key_value_heads=4,
    num_experts=4,
    experts_per_token=1,
    num_hidden_layers=12,
    hidden_size =1024,
    intermediate_size = 1024
    
    ),device)
print(sum([p.numel() for p in model.parameters()])/1000000,"M parameaters")
torch.save(model.state_dict(),"model/gptoss.pt")
tl,vl,ts= trainer(model,train_loader,val_loader,device)
generate_text(model,context)
