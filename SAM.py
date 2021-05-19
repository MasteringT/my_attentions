class SAM(nn.Module):
    def __init__(self, dim, heads=4, dim_head=None):
        super(SAM, self).__init__()
        self.dim_head = (int(dim / heads)) if dim_head is None else dim_head
        _dim = self.dim_head * heads #64
        self.heads = heads
        self.dim = dim
        self.to_qvk_h = nn.Linear(dim, _dim * 3, bias=False)
        self.to_qvk_m = nn.Linear(dim, _dim * 3, bias=False)
        self.w_z = nn.Linear(self.dim_head*2, dim, bias=False)
        self.w_m = nn.Linear(dim*2, dim*3, bias=False)
        self.scale_factor = self.dim_head ** -0.5

    def forward(self, h,m):
        w  = h.shape[-2]
        h = rearrange(h, 'b c w h -> b (w h) c', c = self.dim)
        m = rearrange(m, 'b c w h -> b (w h) c', c = self.dim)

        qkv_h = self.to_qvk_h(h)  # [batch, tokens, dim*3*heads ]
        qh, kh, vh = tuple(rearrange(qkv_h, 'b t (d k h ) -> k b h t d ', k=3, h=self.heads))
        ah_ = torch.einsum('... i d , ... j d -> ... i j', qh, kh) * self.scale_factor
        ah = torch.softmax(ah_, dim=-1)
        zh = torch.einsum('... i j , ... j d -> ... i d', ah, vh)

        qkv_m = self.to_qvk_m(m)
        qm, km, vm = tuple(rearrange(qkv_m, 'b t (d k h ) -> k b h t d ', k=3, h=self.heads))
        am_ = torch.einsum('... i d , ... j d -> ... i j', qh, km) * self.scale_factor
        am = torch.softmax(am_, dim=-1)
        zm = torch.einsum('... i j , ... j d -> ... i d', am, vm)
        z = self.w_z(torch.cat((zm,zh),axis = -1))
        z = torch.mean(z,dim = 1)
        o = torch.cat((z,h),axis = -1)
        mo,mq,mi = torch.split(self.w_m(o), self.dim, dim=-1)
        m_next = torch.tanh(mq)*torch.sigmoid(mi)+(1-torch.sigmoid(mi))*m
        h_ = torch.sigmoid(mo)*m_next

        h_ = rearrange(h_, 'b (w h) c -> b c w h', c = self.dim,w = w)
        m_next = rearrange(m_next, 'b (w h) c -> b c w h', c = self.dim,w  =w)

        return h_,m_next
