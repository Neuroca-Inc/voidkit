from fractions import Fraction
import cmath, math
N=12
chi=[0,1,0,0,0,-1,0,-1,0,0,0,1]
K=[[cmath.exp(-2j*math.pi*r*s/N)/math.sqrt(N) for r in range(N)] for s in range(N)]
F=[sum(K[s][r]*chi[r] for r in range(N)) for s in range(N)]
print('max Fourier residual', max(abs(F[s]-chi[s]) for s in range(N)))
print('unit q residues', {r: Fraction((r*r)%24,24) for r in [1,5,7,11]})
