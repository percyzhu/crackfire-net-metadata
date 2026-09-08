import numpy as np
from matplotlib.path import Path as MplPath
from matplotlib.patches import Polygon,FancyArrowPatch,PathPatch
INK="#24333c"
MUTED="#596970"

def arrow(ax, p, q, color=MUTED, lw=.85, style='-|>', scale=8, **kw):
    a = FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=scale, color=color, linewidth=lw, **kw)
    ax.add_patch(a)
    return a

def project(p):
    p = np.asarray(p)
    return np.stack([p[..., 2] + .55*p[..., 0], p[..., 1]-.42*p[..., 0]+.18*p[..., 2]], axis=-1)

def face(ax, points, fc, ec='none', lw=.65, zo=1):
    p = Polygon(project(points), closed=True, facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zo)
    ax.add_patch(p)
    return p

def surface(ax, domain, holes, mapfn, color, zo):
    aa = sorted(set([*domain[0]] + [v for h in holes for v in h[0]]))
    bb = sorted(set([*domain[1]] + [v for h in holes for v in h[1]]))
    paths=[]
    for a0, a1 in zip(aa[:-1], aa[1:]):
        for b0, b1 in zip(bb[:-1], bb[1:]):
            ac, bc = (a0+a1)/2, (b0+b1)/2
            if any(h[0][0]<ac<h[0][1] and h[1][0]<bc<h[1][1] for h in holes): continue
            p=project([mapfn(a0,b0),mapfn(a1,b0),mapfn(a1,b1),mapfn(a0,b1)])
            paths.append(MplPath(np.vstack([p,p[0]]),[MplPath.MOVETO]+[MplPath.LINETO]*3+[MplPath.CLOSEPOLY]))
    ax.add_patch(PathPatch(MplPath.make_compound_path(*paths),facecolor=color,edgecolor='none',zorder=zo))
    a0, a1 = domain[0]; b0, b1 = domain[1]
    face(ax, [mapfn(a0,b0),mapfn(a1,b0),mapfn(a1,b1),mapfn(a0,b1)], 'none', INK, .7, zo+1)

def dim(ax, p, q, text, offset=(0,0), textoffset=(0,0), rotation=0):
    p, q, v = np.asarray(p), np.asarray(q), np.asarray(offset)
    for pt in [p,q]: ax.plot([pt[0],pt[0]+v[0]],[pt[1],pt[1]+v[1]],color=MUTED,lw=.5)
    arrow(ax,p+v,q+v,style='<->',scale=6,lw=.65)
    ax.text(*((p+q)/2+v+np.asarray(textoffset)),text,fontsize=8,color=INK,ha='center',va='center',
            rotation=rotation,bbox={'facecolor':'white','edgecolor':'none','pad':.5})
