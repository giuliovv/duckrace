opti = Opti()
######################## VARIABLE/PARAMETERS ############################################################
x = opti.variable(5,N+1)            # States with dimension (n_x, prediction_horizon+1) (x(0),...,x(N+1))
u = opti.variable(2,N)#-delay)      # Inputs with dimension (n_u, prediction_horizon) (u(0),...,u(N))
slack1 = opti.variable(1,N)         # slack variable 1
slack2 = opti.variable(1,N)         # slack variable 2
p = opti.parameter(5,1)             # Parameter to set the initial state conditions (x(0))
r = opti.parameter(2,N+1)           # Parameter to set the reference position (r(0),...,r(N+1))
tr = opti.parameter(1,N+1)          # Parameter to set the reference angles
ins = opti.parameter(2,N+1)         # Parameter to set the inside boundary
out = opti.parameter(2,N+1)         # Parameter to set the outside boundary
u_delay = opti.parameter(2,delay)   # Parameter to set the delay (NOT USED)
u_prev = opti.parameter(2,1)        # Parameter to set previus input value (u(-1))
weights = opti.parameter(6)         # Parameter for the cost function weights  
theta = opti.parameter(4)           # Additional learned parameters
######################## COST FUNCTION WEIGHTS ############################################################
Q1 = weights[0]
Q2 = weights[1]
Q3 = weights[2]
Q4 = weights[3]
Q5 = weights[4]
R = weights[5]
theta_lambda = theta[0]
theta_f_1 = theta[1]
theta_f_2 = theta[2]
theta_f_3 = theta[3]
######################## INITIAL VALUES ############################################################
cost = theta_lambda +\
       Q1*sumsqr(x[:2, 0] - r[:,0]) +\
       Q2*heading_cost(x[2, 0],tr[:,0]) +\
       Q3*sumsqr(x[3, 0]-0.6) +\
       R*sumsqr(u[:,0]-u_prev)# initial cost

opti.subject_to(x[:,1]==F(x[:,0],u[:,0]))         # system dynamics (k=0)
opti.subject_to(x[:,0] == p)                      # set initial x_0
######################## STAGE COST/RECURSIVE CONTRAINTS ############################################################
for k in range(1,N):
    cost += gamma**k * (Q1*sumsqr(x[:2, k] - r[:,k]) +\
                        #Q2*sumsqr(fmin(2*np.pi - fabs(mod(x[2, k],2*pi)-tr[:,k]), fabs(mod(x[2, k],2*pi)-tr[:,k]))) +\
                        Q2*heading_cost(x[2, k],tr[:,k]) +\
                        Q3*sumsqr(x[3, k]-0.6) +\
                        #Q4*sumsqr(x[4, k]) +\
                        R*sumsqr(u[:,k]-u[:,k-1]) +\
                        1e1*slack1[:,k] + 1e1*slack2[:,k]
                        )
    opti.subject_to(x[:,k+1]==F(x[:,k],u[:,k]))
    opti.subject_to(sumsqr(x[:2,k]) + slack1[:,k] >= sumsqr(ins[:,k]))
    opti.subject_to(sumsqr(x[:2,k]) - slack2[:,k] <= sumsqr(out[:,k]))
######################## TERMINAL COST/CONSTRAINTS ############################################################
cost += gamma**N * (theta_f_1*sumsqr(x[:2, N]-r[:,N])+ theta_f_2*heading_cost(x[2, k],tr[:,k]) + theta_f_3*sumsqr(x[3, k]-0.6))
######################## ADDITIONAL PARAMETERS ############################################################
opti.subject_to(vec(u) <= 1)               # actuator LB
opti.subject_to(vec(u) >= -1)              # actuator LB
opti.subject_to(x[3,:] >= 0)               # Enforce forward movement
opti.subject_to(vec(slack1) >= 0)          # slack 1 condition
opti.subject_to(vec(slack2) >= 0)          # slack 2 condition
######################## INITIALIZE PARAMETERS ############################################################
opti.set_value(p,[pose.x, pose.y, pose.theta, 0, 0]) # Set the initial x0 value
opti.set_value(r,r0) # Set the initial reference points
opti.set_value(tr,t0) # Set the initial reference angles
opti.set_value(ins,ins0) # Set the initial reference angles
opti.set_value(out,out0) # Set the initial reference angles
opti.set_value(u_delay, u_delay0) # Initial delay
opti.set_value(u_prev, u_prev_0) # Initial delay
opti.set_value(weights, weights_0)
opti.set_value(theta, theta_0) 
######################## SOLVER SPECIFICATIONS ############################################################
# Ipopt
opts = dict()
opts["ipopt.print_level"] = 0
opts["print_time"] = False
opti.solver('ipopt',opts)
opti.minimize(cost)
######################## CREATE VALUE FUNCTION V(s) ############################################################
V = opti.to_function('V',[p, r, tr, ins, out, u_delay, u_prev, weights, theta], # inputs
                         [cost, u[:,0], gradient(cost + opti.lam_g.T @ opti.g, weights), gradient(cost + opti.lam_g.T @ opti.g, theta)], # outputs
                         ['p', 'r', 'tr', 'ins', 'out','u_delay', 'u_prev', 'weights', 'theta'],
                         ['cost', 'u_opt', 'gradient_weights', 'gradient_theta'])
######################## TESTING ############################################################
# Test over 1 time horizon
if test_results:
    sol = opti.solve()