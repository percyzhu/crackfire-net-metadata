import torch
from torch import nn
from torch.nn import functional as F

def mlp(a,b): return nn.Sequential(nn.Linear(a,32),nn.SiLU(),nn.Linear(32,b))

class Model(nn.Module):
    def __init__(self,kind):
        super().__init__(); self.kind=kind
        self.node=mlp(14,32); self.beam=mlp(3,32); self.fire=nn.GRU(3,32,batch_first=True)
        self.messages=nn.ModuleList([mlp(68,32) for _ in range(2)]) if kind!='deepsets' else nn.ModuleList()
        self.updates=nn.ModuleList([mlp(64,32) for _ in range(2)]) if kind!='deepsets' else nn.ModuleList()
        self.to_global=nn.ModuleList([mlp(64,32) for _ in range(2)]) if kind=='virtual_gnn' else nn.ModuleList()
        self.global_update=nn.ModuleList([mlp(64,32) for _ in range(2)]) if kind=='virtual_gnn' else nn.ModuleList()
        self.to_crack=nn.ModuleList([mlp(64,32) for _ in range(2)]) if kind=='virtual_gnn' else nn.ModuleList()
        self.query=mlp(34,32); self.decode=mlp(129,4)
        self.register_buffer('z',torch.linspace(1/60,59/60,31))

    def forward(self,nodes,mask,edges,fire,beam):
        B,N,_=nodes.shape; h=self.node(nodes)*mask[...,None]; g=self.beam(beam)
        count=mask.sum(1).clamp_min(1)
        em=mask[:,:,None]&mask[:,None,:]&~torch.eye(N,device=h.device,dtype=torch.bool)[None]
        for k,(msg,upd) in enumerate(zip(self.messages,self.updates)):
            rec=h[:,:,None].expand(-1,-1,N,-1); send=h[:,None].expand(-1,N,-1,-1)
            agg=(msg(torch.cat([rec,send,edges],-1))*em[...,None]).sum(2)/em.sum(2).clamp_min(1)[...,None]
            h=(h+upd(torch.cat([h,agg],-1)))*mask[...,None]
            if self.kind=='virtual_gnn':
                gm=(self.to_global[k](torch.cat([h,g[:,None].expand(-1,N,-1)],-1))*mask[...,None]).sum(1)/count[:,None]
                g=g+self.global_update[k](torch.cat([g,gm],-1))
                h=(h+self.to_crack[k](torch.cat([h,g[:,None].expand(-1,N,-1)],-1)))*mask[...,None]
        pool=h.sum(1)/count[:,None]; temporal=self.fire(fire)[0]
        q=self.z[None,:,None].expand(B,-1,N)
        query=self.query(torch.cat([h[:,None].expand(-1,31,-1,-1),q[...,None],(q-nodes[:,None,:,4])[...,None]],-1))
        query=(query*mask[:,None,:,None]).sum(2)/count[:,None,None]
        T=fire.shape[1]
        return self.decode(torch.cat([query[:,None].expand(-1,T,-1,-1),temporal[:,:,None].expand(-1,-1,31,-1),
            pool[:,None,None].expand(-1,T,31,-1),g[:,None,None].expand(-1,T,31,-1),count[:,None,None,None].expand(-1,T,31,1)/15],-1))

class SpatialModel(Model):
 def __init__(self,kind,norm):
  is_set=kind=='local_set';super().__init__('deepsets' if is_set else 'gnn');self.variant=kind;self.local=kind.startswith('local')
  if not is_set:self.updates=nn.ModuleList([nn.Sequential(nn.Linear(64,129),nn.SiLU(),nn.Linear(129,32)) for _ in range(2)])
  else:self.node=nn.Sequential(nn.Linear(14,705),nn.SiLU(),nn.Linear(705,32))
  if self.local:self.query=mlp(40,32)
  self.register_buffer('beam_mean',torch.tensor(norm['beam_mean'],dtype=torch.float32));self.register_buffer('beam_std',torch.tensor(norm['beam_std'],dtype=torch.float32))

 def forward(self,nodes,mask,edges,fire,beam,query_z=None):
  qz=self.z if query_z is None else torch.as_tensor(query_z,device=nodes.device,dtype=nodes.dtype)
  if qz.ndim!=1 or len(qz)==0 or not torch.isfinite(qz).all() or (qz<0).any() or (qz>1).any():raise ValueError("query_z must be a finite 1D relative coordinate array in [0,1]")
  J=len(qz)
  B,N,_=nodes.shape;h=self.node(nodes)*mask[...,None];g=self.beam(beam);count=mask.sum(1).clamp_min(1)
  em=mask[:,:,None]&mask[:,None,:]&~torch.eye(N,device=h.device,dtype=torch.bool)[None]
  for msg,upd in zip(self.messages,self.updates):
   rec=h[:,:,None].expand(-1,-1,N,-1);send=h[:,None].expand(-1,N,-1,-1)
   agg=(msg(torch.cat([rec,send,edges],-1))*em[...,None]).sum(2)/em.sum(2).clamp_min(1)[...,None]
   h=(h+upd(torch.cat([h,agg],-1)))*mask[...,None]
  pool=h.sum(1)/count[:,None];temporal=self.fire(fire)[0];q=qz[None,:,None].expand(B,-1,N);delta=q-nodes[:,None,:,4]
  parts=[h[:,None].expand(-1,J,-1,-1),q[...,None],delta[...,None]]
  if self.local:
   L=(beam*self.beam_std+self.beam_mean)[:,0,None,None]
   axial=delta*L;half=nodes[:,None,:,7]*L/2;gap=(abs(axial)-half).clamp_min(0)
   parts += [torch.stack([gap/.1,torch.sign(axial)*gap/.1,(abs(axial)<=half).to(nodes.dtype),torch.exp(-gap/.02),torch.exp(-gap/.05),torch.exp(-gap/.15)],-1)]
  query=(self.query(torch.cat(parts,-1))*mask[:,None,:,None]).sum(2)/count[:,None,None];T=fire.shape[1]
  cat=torch.cat([query[:,None].expand(-1,T,-1,-1),temporal[:,:,None].expand(-1,-1,J,-1),pool[:,None,None].expand(-1,T,J,-1),g[:,None,None].expand(-1,T,J,-1),count[:,None,None,None].expand(-1,T,J,1)/15],-1)
  return F.softplus(self.decode(cat))
