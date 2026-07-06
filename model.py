# model.py
import torch
import torch.nn as nn
from torch_geometric.nn import TGNMemory
from torch_geometric.nn.models.tgn import IdentityMessage, LastAggregator

class TimeEncoder(nn.Module):
    """Encodes continuous time intervals into dense vector spaces (Time2Vec style)"""
    def __init__(self, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.lin = nn.Linear(1, out_channels)

    def forward(self, t):
        # t shape: (batch_size) -> (batch_size, 1)
        return torch.cos(self.lin(t.unsqueeze(-1).float()))

class TGNLinkPredictor(nn.Module):
    def __init__(self, num_nodes, edge_feat_dim, memory_dim=172, time_dim=100):
        super().__init__()
        
        # 1. Continuous Time Encoding Layer
        self.time_encoder = TimeEncoder(time_dim)
        
        # 2. Main Memory Core Module (Paper Identity Message + Last Aggregator)
        self.memory = TGNMemory(
            num_nodes=num_nodes,
            raw_msg_dim=edge_feat_dim,
            memory_dim=memory_dim,
            time_dim=time_dim,
            message_module=IdentityMessage(
                raw_msg_dim=edge_feat_dim, 
                memory_dim=memory_dim, 
                time_dim=time_dim
            ),
            aggregator_module=LastAggregator(),
        )
        
        # 3. Task Decoder (Predicts edge probabilities from concatenated state vectors)
        self.decoder = nn.Sequential(
            nn.Linear(memory_dim * 2 + time_dim, 100),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(100, 1)
        )

    def update_memory(self, src, dst, t, msg):
        self.memory.update_state(src, dst, t, msg)

    def forward(self, src, dst, t):
        # Read latest persistent dynamic memory tracking arrays
        memory, last_update = self.memory(torch.arange(self.memory.num_nodes, device=src.device))
        
        # Compute elapsed delta timestamps to protect against memory staleness
        delta_t_src = t - last_update[src]
        time_emb = self.time_encoder(delta_t_src)
        
        # Combine structural context snapshots for evaluation representation
        src_emb = memory[src]
        dst_emb = memory[dst]
        
        edge_representation = torch.cat([src_emb, dst_emb, time_emb], dim=-1)
        return self.decoder(edge_representation).squeeze(-1)