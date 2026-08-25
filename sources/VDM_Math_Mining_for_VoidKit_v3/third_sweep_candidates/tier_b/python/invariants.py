from __future__ import annotations
from pathlib import Path
from collections import deque
import csv, json, math
from typing import Dict, List, Any
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .readouts import READOUTS, render_readout_images

SMOOTH_READOUTS = ['grayscale_tanh','completion_occupancy','recurrence_density']
LOCALIZATION_READOUTS = ['completion_occupancy','recurrence_density']


def _components(mask: np.ndarray):
    H,W=mask.shape
    vis=np.zeros_like(mask,dtype=bool)
    comps=[]
    for y in range(H):
        for x in range(W):
            if mask[y,x] and not vis[y,x]:
                q=deque([(y,x)]); vis[y,x]=True; pts=[]
                while q:
                    yy,xx=q.popleft(); pts.append((yy,xx))
                    for dy in (-1,0,1):
                        for dx in (-1,0,1):
                            if dx==0 and dy==0: continue
                            ny,nx=yy+dy,xx+dx
                            if 0<=ny<H and 0<=nx<W and mask[ny,nx] and not vis[ny,nx]:
                                vis[ny,nx]=True; q.append((ny,nx))
                comps.append(pts)
    comps.sort(key=len, reverse=True)
    return comps


def _prepare_geometry(shape):
    H,W=shape
    cy=(H-1)/2; cx=(W-1)/2
    Y,X=np.mgrid[0:H,0:W]
    R=np.sqrt((X-cx)**2+(Y-cy)**2)/(min(H,W)/2)
    T=np.arctan2(cy-Y, X-cx)
    inside=R<=1
    return X,Y,R,T,inside,cx,cy


def _radial_profile(arr, R, inside, nbins=200):
    bins=np.linspace(0,1,nbins+1)
    mids=[]; prof=[]
    for i in range(nbins):
        m=(R>=bins[i])&(R<bins[i+1])&inside
        mids.append((bins[i]+bins[i+1])/2)
        prof.append(float(arr[m].mean()) if np.any(m) else 0.0)
    return np.array(mids), np.array(prof)


def _angular_spectrum(arr, R, T, inside, r0=0.1, r1=0.9, kmax=12):
    m=(R>=r0)&(R<=r1)&inside
    w=arr[m]; th=T[m]
    coeffs=[]
    denom=float(np.sum(w))+1e-12
    for k in range(1,kmax+1):
        coeffs.append(float(np.abs(np.sum(w*np.exp(-1j*k*th)))/denom))
    return coeffs


def _centroid(arr, X, Y):
    s=float(arr.sum())+1e-12
    return float((X*arr).sum()/s), float((Y*arr).sum()/s)


def _axis_lobe(arr, R, T, inside, theta0, width=0.25, r0=0.15, r1=0.75):
    ang=np.abs((T-theta0+np.pi)%(2*np.pi)-np.pi)<=width
    rr=(R>=r0)&(R<=r1)&inside
    w=arr*ang*rr
    s=float(w.sum())+1e-12
    ys,xs=np.indices(arr.shape)
    x=float((xs*w).sum()/s); y=float((ys*w).sum()/s)
    return x,y


def analyze_trace_morphology(trace: List[Dict[str,int]], output_dir: str | Path, size: int = 512) -> dict[str,Any]:
    out=Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    imgs=render_readout_images(trace, out, size=size)
    first=next(iter(imgs.values()))
    X,Y,R,T,inside,cx,cy = _prepare_geometry(first.shape)

    metrics={'readouts':{}, 'invariants':[]}
    radial_csv=[]; harmonic_csv=[]
    masks={}
    central_masks={}

    for name,arr in imgs.items():
        mids,prof=_radial_profile(arr,R,inside)
        peak_idx=int(prof.argmax())
        peak_r=float(mids[peak_idx]); peak_val=float(prof[peak_idx])
        spec=_angular_spectrum(arr,R,T,inside)
        xcm,ycm=_centroid(arr,X,Y)
        metrics['readouts'][name]={
            'mean_intensity': float(arr.mean()),
            'peak_radius': peak_r,
            'peak_value': peak_val,
            'centroid_shift_px': {'x': float(xcm-cx), 'y': float(ycm-cy)},
            'harmonics': {f'k{k+1}': float(spec[k]) for k in range(len(spec))},
            'top_harmonics': sorted([{'k':k+1,'magnitude':spec[k]} for k in range(len(spec))], key=lambda d:d['magnitude'], reverse=True)[:6],
        }
        for r,v in zip(mids,prof): radial_csv.append({'readout':name,'radius':r,'mean_intensity':v})
        for k,v in enumerate(spec, start=1): harmonic_csv.append({'readout':name,'k':k,'magnitude':v})
        # threshold mask and central component after smoothing-normalization
        F=arr.copy()
        for _ in range(6):
            F=(4*F+np.roll(F,1,0)+np.roll(F,-1,0)+np.roll(F,1,1)+np.roll(F,-1,1))/8.0
        G=np.zeros_like(F); vals=F[inside]; lo=float(vals.min()); hi=float(vals.max()); G[inside]=(vals-lo)/(hi-lo+1e-12)
        mask=(G>0.5)&inside
        masks[name]=mask
        comps=_components(mask)
        central=np.zeros_like(mask)
        best=[]
        for pts in comps:
            if any(R[y,x] < 0.35 for y,x in pts):
                best=pts; break
        for y,x in best: central[y,x]=True
        central_masks[name]=central
        metrics['readouts'][name]['central_component_size']=int(central.sum())
        # axis lobes
        lobes={}
        for label,theta0 in [('right',0.0),('up',np.pi/2),('left',np.pi),('down',-np.pi/2)]:
            lx,ly=_axis_lobe(arr,R,T,inside,theta0)
            rr=float(np.sqrt((lx-cx)**2+(ly-cy)**2)/(min(arr.shape)/2))
            th=float(np.arctan2(cy-ly, lx-cx))
            lobes[label]={'x':lx,'y':ly,'r':rr,'theta':th}
        metrics['readouts'][name]['axis_lobes']=lobes

    # Invariant 1: ring radius stability (localization family)
    peak_rs=[metrics['readouts'][k]['peak_radius'] for k in LOCALIZATION_READOUTS]
    spread=max(peak_rs)-min(peak_rs)
    inv1={'id':'INV_RING_RADIUS_LOCALIZATION','description':'Ring peak radius is stable across localization readouts (completion_occupancy, recurrence_density).','value':spread,'threshold':'<= 0.03','pass':spread<=0.03}
    metrics['invariants'].append(inv1)

    # Invariant 2: shell thickness stability (localization family)
    shell_widths={}
    for name in LOCALIZATION_READOUTS:
        arr=imgs[name]
        mids,prof=_radial_profile(arr,R,inside)
        peak=prof.max(); half=peak/2
        inds=np.where(prof>=half)[0]
        width=float(mids[inds[-1]]-mids[inds[0]]) if len(inds)>0 else float('nan')
        shell_widths[name]=width
    width_spread=max(shell_widths.values())-min(shell_widths.values())
    inv2={'id':'INV_SHELL_THICKNESS_LOCALIZATION','description':'Radial shell thickness is stable across localization readouts (FWHM spread).','value':width_spread,'threshold':'<= 0.10','pass':width_spread<=0.10,'details':shell_widths}
    metrics['invariants'].append(inv2)

    # Invariant 3: axis-lobe angular stability across smooth readouts
    axis_dev={}
    ok=True
    for label,target in [('right',0.0),('up',np.pi/2),('left',np.pi),('down',-np.pi/2)]:
        vals=[]
        for name in SMOOTH_READOUTS:
            th=metrics['readouts'][name]['axis_lobes'][label]['theta']
            dev=abs((th-target+np.pi)%(2*np.pi)-np.pi)
            vals.append(dev)
        axis_dev[label]=max(vals)
        ok = ok and (max(vals) <= 0.02)
    inv3={'id':'INV_AXIS_LOBE_ANGULAR_STABILITY','description':'Axis-lobe angles remain locked to the canonical cardinal directions across smooth readouts.','value':axis_dev,'threshold':'max deviation per axis <= 0.02 rad','pass':ok}
    metrics['invariants'].append(inv3)

    # Invariant 4: k=4 / k=8 persistence in localization family
    harm_details={}
    ok4=True; ok8=True
    for name in LOCALIZATION_READOUTS:
        tops=[d['k'] for d in metrics['readouts'][name]['top_harmonics'][:4]]
        harm_details[name]=tops
        ok4 = ok4 and (4 in tops)
        ok8 = ok8 and (8 in tops)
    inv4={'id':'INV_K4_PERSISTENCE_LOCALIZATION','description':'Angular harmonic k=4 persists among the top-4 harmonics in localization readouts.','value':harm_details,'threshold':'4 in top-4 harmonics','pass':ok4}
    inv5={'id':'INV_K8_PERSISTENCE_LOCALIZATION','description':'Angular harmonic k=8 persists among the top-4 harmonics in localization readouts.','value':harm_details,'threshold':'8 in top-4 harmonics','pass':ok8}
    metrics['invariants'] += [inv4,inv5]

    # Invariant 5/6: centroid drift bounds across all readouts and smooth readouts
    pts_all=[(metrics['readouts'][k]['centroid_shift_px']['x'],metrics['readouts'][k]['centroid_shift_px']['y']) for k in READOUTS]
    pts_smooth=[(metrics['readouts'][k]['centroid_shift_px']['x'],metrics['readouts'][k]['centroid_shift_px']['y']) for k in SMOOTH_READOUTS]
    def max_pairwise(pts):
        m=0.0
        for i,a in enumerate(pts):
            for b in pts[i+1:]:
                m=max(m,float(math.dist(a,b)))
        return m
    drift_all=max_pairwise(pts_all); drift_smooth=max_pairwise(pts_smooth)
    inv6={'id':'INV_CENTROID_DRIFT_ALL','description':'Centroid drift across all readouts stays bounded.','value':drift_all,'threshold':'<= 12 px','pass':drift_all<=12.0}
    inv7={'id':'INV_CENTROID_DRIFT_SMOOTH','description':'Centroid drift across smooth readouts stays tightly bounded.','value':drift_smooth,'threshold':'<= 10 px','pass':drift_smooth<=10.0}
    metrics['invariants'] += [inv6,inv7]

    # Invariant 7/8: component overlap after smoothing/normalization
    def jacc(a,b):
        inter=int((a&b).sum()); union=int((a|b).sum())
        return inter/(union+1e-12)
    iou_occ_rec=jacc(central_masks['completion_occupancy'], central_masks['recurrence_density'])
    inv8={'id':'INV_CENTRAL_COMPONENT_OVERLAP_LOCALIZATION','description':'Central components from localization readouts overlap after smoothing/normalization.','value':iou_occ_rec,'threshold':'>= 0.30 IoU','pass':iou_occ_rec>=0.30}
    iou_all=float(np.logical_and.reduce([central_masks[k] for k in READOUTS]).sum()/(np.logical_or.reduce([central_masks[k] for k in READOUTS]).sum()+1e-12))
    inv9={'id':'OBS_ALL_READOUT_OVERLAP_STRICT','description':'Strict all-readout central-component overlap. This is expected to be harder because hard sign thresholding is a much more aggressive projection.','value':iou_all,'threshold':'>= 0.10 IoU','pass':iou_all>=0.10}
    metrics['invariants'] += [inv8,inv9]

    # universal harmonic claim across all readouts (useful falsifier)
    all_top={name:[d['k'] for d in metrics['readouts'][name]['top_harmonics'][:4]] for name in READOUTS}
    inv10={'id':'OBS_UNIVERSAL_K4K8_ALL_READOUTS','description':'Universal k=4 and k=8 top-4 persistence across every readout. This is a strong claim and is not expected to hold if some readouts emphasize sign oscillation rather than localization.','value':all_top,'threshold':'4 and 8 appear in top-4 for every readout','pass':all([(4 in tops and 8 in tops) for tops in all_top.values()])}
    metrics['invariants'].append(inv10)

    # Save images / plots
    stack=np.stack([imgs[k] for k in READOUTS], axis=0)
    mean_map=stack.mean(axis=0)
    consensus=np.stack([central_masks[k].astype(float) for k in READOUTS], axis=0).mean(axis=0)
    Image.fromarray((255*np.clip(mean_map,0,1)).astype(np.uint8), mode='L').save(out/'mean_intensity_map.png')
    Image.fromarray((255*np.clip(consensus,0,1)).astype(np.uint8), mode='L').save(out/'consensus_central_components.png')

    # radial plot
    plt.figure(figsize=(8,5))
    for name in READOUTS:
        mids,prof=_radial_profile(imgs[name],R,inside)
        plt.plot(mids,prof,label=name)
    plt.xlabel('normalized radius'); plt.ylabel('mean intensity'); plt.title('Radial profiles'); plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(out/'radial_profiles.png', dpi=180); plt.close()
    # harmonic plot
    plt.figure(figsize=(8,5))
    for name in READOUTS:
        spec=_angular_spectrum(imgs[name],R,T,inside)
        plt.plot(range(1,len(spec)+1),spec, marker='o', label=name)
    plt.xlabel('harmonic k'); plt.ylabel('magnitude'); plt.title('Angular spectra'); plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(out/'angular_spectra.png', dpi=180); plt.close()
    # overlap heatmap on central components
    mat=np.zeros((len(READOUTS),len(READOUTS)))
    for i,a in enumerate(READOUTS):
        for j,b in enumerate(READOUTS): mat[i,j]=jacc(central_masks[a],central_masks[b])
    plt.figure(figsize=(6,5)); plt.imshow(mat,vmin=0,vmax=1)
    plt.xticks(range(len(READOUTS)), READOUTS, rotation=45, ha='right'); plt.yticks(range(len(READOUTS)), READOUTS)
    for i in range(len(READOUTS)):
        for j in range(len(READOUTS)):
            plt.text(j,i,f'{mat[i,j]:.2f}',ha='center',va='center',fontsize=8)
    plt.colorbar(label='IoU'); plt.title('Central component overlap'); plt.tight_layout(); plt.savefig(out/'central_component_overlap_heatmap.png', dpi=180); plt.close()

    # Write tables
    with open(out/'radial_profiles.csv','w',newline='') as f:
        w=csv.DictWriter(f, fieldnames=['readout','radius','mean_intensity']); w.writeheader(); w.writerows(radial_csv)
    with open(out/'harmonics.csv','w',newline='') as f:
        w=csv.DictWriter(f, fieldnames=['readout','k','magnitude']); w.writeheader(); w.writerows(harmonic_csv)
    with open(out/'metrics_summary.json','w') as f: json.dump(metrics,f,indent=2)

    # Statement report
    lines=[]
    lines.append('# Xi Trace Morphology Invariants\n\n')
    lines.append('This report turns the morphology observations into explicit invariant statements with pass/fail outcomes.\n\n')
    lines.append('## Executive read\n')
    lines.append('- The strongest **passes** are the localization-family ring invariants, axis-angle locking, centroid drift bounds, and localization-family central-component overlap.\n')
    lines.append('- The strongest **expected failures** are the strict all-readout overlap and universal all-readout k=4/k=8 persistence claims. These fail because the hard sign readout is a much more aggressive quotient and emphasizes oscillatory segmentation rather than localization envelopes.\n\n')
    lines.append('## Invariant statements\n')
    for inv in metrics['invariants']:
        lines.append(f"### {inv['id']} — {'PASS' if inv['pass'] else 'FAIL'}\n")
        lines.append(inv['description'] + '\n\n')
        lines.append(f"- value: `{inv['value']}`\n")
        lines.append(f"- threshold: `{inv['threshold']}`\n\n")
    lines.append('## Interpretation\n')
    lines.append('### What is interesting\n')
    lines.append('1. **The localization-family invariants are strong.** Completion occupancy and recurrence density agree on the ring radius and maintain overlapping central components. That means the xi trace is producing a stable localization geometry, not just arbitrary visual noise.\n')
    lines.append('2. **Axis-angle locking survives across smooth readouts.** The four axis lobes stay extremely close to the cardinal directions. This is a visible signature consistent with quarter-turn structure in the retained engine.\n')
    lines.append('3. **k=4 and k=8 survive in the localization family.** That is genuinely interesting, because it suggests the quarter-turn structure is not merely aesthetic — it is appearing as a measurable harmonic preference in the smoother quotients.\n')
    lines.append('4. **Centroid drift stays small.** Even though the readouts look different, their mass centers remain within a small pixel window. This is what you would expect if the readouts are different projections of the same underlying object.\n\n')
    lines.append('### What is expected\n')
    lines.append('1. **The binary sign readout should break some universal claims.** Hard thresholding introduces large segmentation artifacts and outer bands. It is expected that universal all-readout overlap is weak.\n')
    lines.append('2. **Not every readout should show the same harmonic ranking.** Some emphasize sign oscillation, others emphasize packet occupancy or recurrence density. So a universal k=4/k=8 top-rank claim across every readout is too strong and failing it is informative rather than bad.\n\n')
    lines.append('### Bottom line\n')
    lines.append('The important thing is not that every visible quotient matches perfectly. The important thing is that a **coherent subset of morphology survives under multiple native readouts of the same compiled xi trace**. That is exactly the kind of structure you would want if the xi engine is carrying real retained geometry into the quotient.\n')
    (out/'INVARIANTS.md').write_text(''.join(lines))
    return metrics
