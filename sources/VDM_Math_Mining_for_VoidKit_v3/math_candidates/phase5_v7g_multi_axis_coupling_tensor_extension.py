#!/usr/bin/env python3
# Re-run by executing the package builder logic from the archived artifact generation context.
# Core equations:
# c_ij=sum sign*(q_i+1)*(q_j+1) mod lcm(D_i,D_j)
# B_C=sum x_i y_i/D_i + sum c_ij(x_i y_j+x_j y_i)/lcm(D_i,D_j)
# K=|A|^-1/2 exp(-2 pi i B_C), G=exp(pi i B_C(x,x))
print('Phase 5 v7g script surface: see outputs/*.csv and notebook for executable checks.')
