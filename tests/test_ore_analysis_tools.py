import math
import numpy as np

from voidkit.stats.heavytail import gini, sample_discrete_powerlaw, fit_powerlaw_discrete, ks_distance_powerlaw_discrete, vuong_test
from voidkit.stats.basic import hellinger, shannon_entropy_from_values, participation_ratio_from_values, detect_avalanches
from voidkit.info_theory.complexity import perm_entropy, lz_complexity, transfer_entropy_nats, tc_o_gaussian
from voidkit.time_series.events import transfer_entropy_bits, implied_timescale, event_triggered_average
from voidkit.recurrence.rqa import rqa_metrics
from voidkit.topology.vr_graph import pairwise_distances, beta1_curve, bh_fdr, pvals_from_null
from voidkit.graph.matching import pairwise_cost, edge_jaccard
from voidkit.signal.ridges import ridge_points_from_timeseries
from voidkit.signal.spectral import psd_loglog_slope


def test_heavytail_core_on_synthetic_sample():
    rng=np.random.default_rng(7); x=sample_discrete_powerlaw(alpha=2.5,xmin=2,n=3000,rng=rng)
    fit=fit_powerlaw_discrete(x,2)
    assert fit is not None
    alpha,_=fit
    assert 2.2<alpha<2.8
    assert ks_distance_powerlaw_discrete(x,2,alpha)<.08
    assert 0<=gini(x)<=1


def test_vuong_identity():
    ll=np.arange(20,dtype=float)
    z,p=vuong_test(ll,ll)
    assert math.isnan(z) or z==0 or math.isinf(z)


def test_basic_stats():
    assert hellinger([1,0],[1,0])<1e-6
    assert shannon_entropy_from_values([1,1])>0
    assert abs(participation_ratio_from_values([1,1])-2)<1e-12
    s,d=detect_avalanches(np.array([0,2,3,0,4,0],float),1)
    assert d.tolist()==[2,1]


def test_information_metrics_order_and_units_finite():
    x=np.tile(np.arange(8),100).astype(float)
    assert math.isfinite(perm_entropy(x,m=3,tau=1))
    assert math.isfinite(lz_complexity(x))
    y=np.roll(x,1)
    assert math.isfinite(transfer_entropy_nats(x,y,bins=4,lag=1))
    assert math.isfinite(transfer_entropy_bits(x,y,lag=1,n_bins=4))
    X=np.random.default_rng(0).normal(size=(500,3))
    tc,dtc,o=tc_o_gaussian(X)
    assert all(math.isfinite(v) for v in (tc,dtc,o))


def test_timeseries_event_average_and_timescale():
    assert implied_timescale(.5)>0
    avg,n=event_triggered_average(np.arange(20,dtype=float),[5,10],2)
    assert n==2 and len(avg)==5


def test_rqa_metrics_simple():
    R=np.eye(6,dtype=bool)
    out=rqa_metrics(R)
    assert out['RR']==0.0


def test_vr_graph_cycle_rank_square():
    pts=np.array([[0,0],[1,0],[1,1],[0,1]],float)
    eps=np.array([1.01])
    b1,E,C=beta1_curve(pts,eps)
    assert E[0]==4 and C[0]==1 and b1[0]==1
    D=pairwise_distances(pts); assert D.shape==(4,4)


def test_fdr_and_empirical_pvals():
    thr,mask=bh_fdr(np.array([.001,.01,.8]),.05)
    assert mask[:2].all() and not mask[2]
    obs=np.array([2.,1.]); null=np.array([[0.,1.],[1.,1.],[2.,0.]])
    p=pvals_from_null(obs,null)
    assert np.all((p>0)&(p<=1))


def test_graph_cost_and_jaccard():
    X=np.array([[0.,0.],[1.,0.]])
    C=pairwise_cost(X,X); assert np.allclose(np.diag(C),0)
    j,inter,union=edge_jaccard(np.array([1,2,3]),np.array([2,3,4]))
    assert inter==2 and union==4 and j==.5


def test_ridge_extraction_on_sine():
    fs=128.; t=np.arange(0,4,1/fs); h=np.sin(2*np.pi*12*t)
    pts=ridge_points_from_timeseries(t,h,{'stft':{'nperseg':128,'noverlap':64,'freq_max':30,'top_k':1,'power_quantile':0.5}})
    assert pts.shape[1]==2 and len(pts)>=4
    assert abs(np.median(pts[:,1])-12)<1.1


def test_psd_slope_returns_finite_on_coloredish_signal():
    rng=np.random.default_rng(0); x=np.cumsum(rng.normal(size=4096))
    slope,_=psd_loglog_slope(x,fs=1.0,f_low=1/500,f_high=.1)
    assert math.isfinite(slope)
