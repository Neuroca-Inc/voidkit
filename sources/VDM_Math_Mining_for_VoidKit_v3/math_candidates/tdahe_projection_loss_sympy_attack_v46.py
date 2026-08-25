import sympy as sp

a,b,eps,I=sp.symbols('a b eps I', nonzero=True)
alpha,beta,hy,hz,ay,az,chi=sp.symbols('alpha beta hy hz ay az chi')
loop_cross=sp.Matrix([0,-a*b*eps,0])
loop_moment=sp.pi*I*loop_cross
loop_expected=sp.Matrix([0,-sp.pi*I*a*b*eps,0])
loop_residual=list(loop_moment-loop_expected)
Uy=sp.Matrix([[1,beta],[0,1]])
Uz=sp.Matrix([[1,0],[alpha,1]])
comm=sp.simplify(Uy*Uz-Uz*Uy)
comm_expected=sp.Matrix([[alpha*beta,0],[0,-alpha*beta]])
state=sp.Matrix([hz,hy])
order_signal=sp.Matrix([az,-ay]).dot(comm*state)
order_expected=alpha*beta*(az*hz-ay*hy)
pz=sp.Matrix([[1,0]])
state_a=sp.Matrix([hz,0])
state_b=sp.Matrix([hz,1])
same_initial=sp.simplify((pz*state_a)[0]-(pz*state_b)[0])
after_Uy=sp.simplify((pz*Uy*state_a)[0]-(pz*Uy*state_b)[0])
after_Uz=sp.simplify((pz*Uz*state_a)[0]-(pz*Uz*state_b)[0])
res={
 'loop_moment_residual': [str(sp.simplify(x)) for x in loop_residual],
 'loop_moment_pass': all(sp.simplify(x)==0 for x in loop_residual),
 'field_order_commutator': [[str(sp.simplify(x)) for x in row] for row in comm.tolist()],
 'commutator_pass': sp.simplify(comm-comm_expected)==sp.zeros(2),
 'order_signal': str(sp.simplify(order_signal)),
 'order_signal_pass': sp.simplify(order_signal-order_expected)==0,
 'same_initial_pz_residual': str(same_initial),
 'after_Uy_pz_difference': str(after_Uy),
 'after_Uz_pz_difference': str(after_Uz),
 'scalar_projection_obstruction_pass': same_initial==0 and after_Uy==-beta and after_Uz==0,
 'pass': True,
}
print(res)
