import math
import numpy as np
import scipy.sparse as sp

from voidkit.pde.implicit_rd import diffusion_CN_step_periodic, strang_step, dg_rd_step_with_stats, discrete_lyapunov_Lh
from voidkit.wave.noether import verlet_step_with_half, discrete_energy, discrete_momentum
from voidkit.graph.spectral import stationary_distribution, spectral_embedding, participation_coefficient, gridify
from voidkit.info_theory.gaussian import mi_gaussian_xy, gaussian_tc_dtc_o
from voidkit.info_theory.divergence import jensen_shannon_divergence


def test_cn_diffusion_preserves_constant():
    x=np.ones(64); y=diffusion_CN_step_periodic(x,.2,1.,.5)
    assert np.max(np.abs(y-x))<1e-14


def test_exact_reaction_strang_D0_matches_exact_flow():
    x=np.array([.1,.2,.3]); y=strang_step(x,.3,1.,0.,.2,.1)
    from voidkit.dynamics.logistic import exact_step
    assert np.allclose(y, exact_step(x,.2,.1,.3), atol=1e-14, rtol=0)


def test_newton_dg_stats_converge_small_case():
    x=np.linspace(.01,.1,16)
    y,st=dg_rd_step_with_stats(x,.01,1.,.1,.2,.1)
    assert st['converged'] and np.all(np.isfinite(y))


def test_noether_core_finite():
    N=64; dx=2*np.pi/N; x=np.arange(N)*dx
    phi=np.sin(x); pi=np.cos(x)*.1
    p1,ph,pn=verlet_step_with_half(phi,pi,.01,dx,1.,.2)
    E=discrete_energy(phi,p1,ph,dx,1.,.2); P=discrete_momentum(phi,p1,ph,dx)
    assert math.isfinite(E) and math.isfinite(P)


def test_stationary_distribution_and_spectral_graph_tools():
    A=sp.csr_matrix(np.array([[0,1,0],[1,0,1],[0,1,0]],float))
    pi=stationary_distribution(A,teleport=1e-6)
    assert np.isclose(pi.sum(),1) and np.all(pi>=0)
    vals,coords=spectral_embedding(A,n_components=1,seed=0)
    assert coords.shape==(3,1)
    P=participation_coefficient(A,np.array([0,0,1]))
    assert P.shape==(3,)
    grid,counts,meta=gridify(np.array([[0,0],[1,1]],float),np.array([2.,3.]),grid_size=4)
    assert counts.sum()==2 and np.isclose(grid.sum(),5)


def test_gaussian_information_and_jsd():
    rng=np.random.default_rng(0); x=rng.normal(size=(1000,1)); y=x+.2*rng.normal(size=(1000,1))
    assert mi_gaussian_xy(x,y)>0
    X=np.c_[x,y,rng.normal(size=(1000,1))]
    tc,dtc,o=gaussian_tc_dtc_o(X)
    assert all(np.isfinite([tc,dtc,o]))
    assert jensen_shannon_divergence([1,0],[1,0])==0
