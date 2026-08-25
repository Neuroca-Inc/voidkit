# Carry the state. L never resets. End of one domain = start of the next, into the
# unlocked DOF. One continuous phase walk. Read the seats as the walk goes.
import math

def med(u,v):
    a,b = sorted((v,u+v)); return a,b

# state carried the whole way
u,v = 1,1
theta = 0.0          # accruing phase, NEVER reset
depth = 0            # axes unlocked = number of L's
word = []
seats = []           # (global_step, depth, theta_quarters, sign) at each Q

def phase_sign(theta):
    re = round(math.cos(theta)); im = round(math.sin(theta))
    return re if re != 0 else im

step = 0
positions_in_domain = 0
saturate_at = 6      # phase positions available before an L is forced
while depth < 2:
    # B: advance the corridor, carried
    u,v = med(u,v); word.append('B')
    # Q: quarter turn, phase accrues and is NEVER reset
    theta += math.pi/2; word.append('Q')
    step += 1; positions_in_domain += 1
    seats.append((step, depth, round(theta/(math.pi/2)), phase_sign(theta)))
    # saturation -> L : carry theta forward, unlock one axis, keep walking
    if positions_in_domain == saturate_at:
        word.append('L'); depth += 1; positions_in_domain = 0
        # NO reset. theta continues. (u,v) continues. only DOF (depth) grows.

print("word:", ''.join(word))
print()
print(" step depth  quarter-turns(theta/(pi/2))  phase-sign")
for s in seats:
    print(f"  {s[0]:>3}    {s[1]}            {s[2]:>3}                     {s[3]:+d}")
print()
sgn = [s[3] for s in seats]
print("all 12 phase signs in walk order:", sgn)
print("first 6 (domain 1):", sgn[:6])
print("next 6  (domain 2):", sgn[6:])
print()
print("position 1 vs position 7 (the n=7 question):")
print(f"  step 1 sign = {sgn[0]:+d}   step 7 sign = {sgn[6]:+d}   folded onto each other? {sgn[0]==sgn[6] and seats[0][2]%6==seats[6][2]%6}")
print(f"  step 7 quarter-turn index = {seats[6][2]}  (continues past 6, lands in unlocked axis; not folded onto step 1)")
