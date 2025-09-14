import torch
import torch.nn as nn
import torch.nn.functional as F

class SGNS(nn.Module):
    """
    Skip-gram with Negative Sampling:
    loss = -log sigma(v_c·v_o) - sum log sigma(-v_c·v_neg)
    """
    def __init__(self, vocab_size:int, emb_dim:int=128):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, emb_dim)
        self.out_embed = nn.Embedding(vocab_size, emb_dim)
        nn.init.uniform_(self.in_embed.weight, -0.5/emb_dim, 0.5/emb_dim)
        nn.init.zeros_(self.out_embed.weight)

    def forward(self, center, context, negatives):
        # center: [B], context: [B], negatives: [B, K]
        v_c = self.in_embed(center)                  # [B, D]
        v_o = self.out_embed(context)                # [B, D]
        pos_score = (v_c * v_o).sum(dim=1)           # [B]
        pos_loss = F.logsigmoid(pos_score)           # [B]

        neg_vecs = self.out_embed(negatives)         # [B, K, D]
        neg_score = torch.bmm(neg_vecs, v_c.unsqueeze(2)).squeeze(2)  # [B, K]
        neg_loss = F.logsigmoid(-neg_score).sum(dim=1)                # [B]

        loss = -(pos_loss + neg_loss).mean()
        return loss

    @torch.no_grad()
    def get_input_vectors(self):
        return self.in_embed.weight.detach().cpu()
