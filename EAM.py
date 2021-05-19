class EAM(nn.Module):
    def __init__(self, d_model,S = 8):
        super(EAM, self).__init__()
        self.dim = d_model
        self.w_z = nn.Linear(self.dim*2, self.dim, bias=False)
        self.w_m = nn.Linear(self.dim*2,self.dim*3, bias=False)
        ####
        self.mk_h=nn.Linear(d_model,S,bias=False)
        self.mv_h=nn.Linear(S,d_model,bias=False)
        self.mk_m=nn.Linear(d_model,S,bias=False)
        self.mv_m=nn.Linear(S,d_model,bias=False)
        self.softmax=nn.Softmax(dim=1)
        self.init_weights()

    def forward(self, h,m):
        w  = h.shape[-2]
        h = rearrange(h, 'b c w h -> b (w h) c', c = self.dim)
        m = rearrange(m, 'b c w h -> b (w h) c', c = self.dim)

        #
        attn=self.mk_h(h) #bs,n,S
        attn=self.softmax(attn) #bs,n,S
        attn=attn/torch.sum(attn,dim=2,keepdim=True) #bs,n,S
        zh=self.mv_h(attn) #bs,n,d_model

        attn=self.mk_m(m) #bs,n,S
        attn=self.softmax(attn) #bs,n,S
        attn=attn/torch.sum(attn,dim=2,keepdim=True) #bs,n,S
        zm=self.mv_m(attn) #bs,n,d_model
        z = self.w_z(torch.cat((zm,zh),axis = -1))
        z = torch.mean(z,dim = 1)  #multi-head
        o = torch.cat((z,h),axis = -1)
        mo,mq,mi = torch.split(self.w_m(o), self.dim, dim=-1)
        m_next = torch.tanh(mq)*torch.sigmoid(mi)+(1-torch.sigmoid(mi))*m
        h_ = torch.sigmoid(mo)*m_next

        h_ = rearrange(h_, 'b (w h) c -> b c w h', c = self.dim,w = w)
        m_next = rearrange(m_next, 'b (w h) c -> b c w h', c = self.dim,w  =w)
        return h_,m_next

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
