# initial_value = 0
# constant = 5
#
# for i in range(8):
#     initial_value = initial_value + constant
# print(initial_value)
import torch
import torch.nn as nn
x = nn.Parameter(torch.zeros(2, 32, 1024, 20, 20))
print(x.shape)
def forward(self, x):
    """
    Args:
        x (tuple?)

    """
    print(x.shape)
    rgb_fea = x[0]  # rgb_fea (tensor): dim:(B, C, H, W)
    ir_fea = x[1]  # ir_fea (tensor): dim:(B, C, H, W)
    assert rgb_fea.shape[0] == ir_fea.shape[0]
    bs, c, h, w = rgb_fea.shape

    # -------------------------------------------------------------------------
    # AvgPooling
    # -------------------------------------------------------------------------
    # AvgPooling for reduce the dimension due to expensive computation
    rgb_fea = self.avgpool(rgb_fea)
    ir_fea = self.avgpool(ir_fea)  # dim:(B, C, 8, 8)

    # -------------------------------------------------------------------------
    # Transformer
    # -------------------------------------------------------------------------
    # pad token embeddings along number of tokens dimension
    rgb_fea_flat = rgb_fea.view(bs, c, -1)  # flatten the feature b,c n
    ir_fea_flat = ir_fea.view(bs, c, -1)  # flatten the feature b, c n
    token_embeddings = torch.cat([rgb_fea_flat, ir_fea_flat], dim=2)  # concat
    token_embeddings = token_embeddings.permute(0, 2,
                                                1).contiguous()  # dim:(B, 2*H*W, C) .contiguous()方法在底层开辟新内存，在内存上tensor是连续的
    x = self.drop(self.pos_emb + token_embeddings)  # sum positional embedding and token    dim:(B, 2n, C)

    for i in range(self.n_layers - 1):
        x = torch.cat([self.fusion[i](x), ir_fea_flat], dim=2)
    x = self.fusion[-1](x)

    # decoder head
    x = self.ln_f(x)  # dim:(B, H*W, C)
    x = x.view(bs, self.vert_anchors, self.horz_anchors, self.n_embd)
    x = x.permute(0, 3, 1, 2)  # dim:(B, C, H, W)

    # -------------------------------------------------------------------------
    # Interpolate (or Upsample)
    # -------------------------------------------------------------------------
    rgb_fea_out = F.interpolate(x, size=([h, w]), mode='bilinear')
    print(rgb_fea_out.shape)
    return rgb_fea_out