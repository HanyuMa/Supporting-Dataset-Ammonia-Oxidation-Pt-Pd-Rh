
# coding: utf-8

# In[ ]:


"""
Microkinetic model for ammonia oxidation
Inspired by Grabow, Lars C. 
"Computational catalyst screening."
Computational Catalysis. RSC Publishing, 2013. 1-58.
"""


# In[ ]:


import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from copy import deepcopy


# In[ ]:


# Ammonia oxidation on Pt(211) is used as an example here
# Reaction conditions
P0  = 101325    # Unit: Pa
P = P0
PNH3 = 1000 * 10 **(-6) * P
PO2 = 0.02 * P
PH2O = 0.05 * P
PNO = 0 * P
PN2 = 0 * P
PN2O = 0 * P

# Physical constants
kb = 8.617*10**(-5)        # eV K-1
kbj = 1.38064852*10**(-23) # J K-1
R = 8.314                  # J mol-1 K-1
h = 4.135667*10**(-15)     # eV*s
CT = 1.467 * 10**19        # m-2 sites per unit area on the Pt 211/111 surface
NA = 6.02214086*10**23     # mol-1
evtoj= 96.485 * 1000       # J/eV

# Entropy of gases are from NIST-JANAF. Adsorbate entropies are estimated with harmonic oscillator approximation. 
gas_entropy = pd.read_csv("gas_entropy.csv")
adsorbate_entropy = pd.read_csv("adsorbate_entropy_Pt.csv")


# In[ ]:


# Calculate the entropy for gas adsorption
deltaS_O2 = adsorbate_entropy['PtO']*2 - gas_entropy['S_O2']/evtoj
deltaS_NH3 = adsorbate_entropy['PtNH3'] - gas_entropy['S_NH3']/evtoj
deltaS_NO = adsorbate_entropy['PtNO'] - gas_entropy['S_NO']/evtoj
deltaS_N2 = adsorbate_entropy['PtN']*2 - gas_entropy['S_N2']/evtoj
deltaS_H2O = adsorbate_entropy['PtH2O'] - gas_entropy['S_H2O']/evtoj
deltaS_N2O = adsorbate_entropy['PtN2O'] - gas_entropy['S_N2O']/evtoj


# In[ ]:


def get_rate_constants(T):
# Activation energy and prefactors for 40 reactions (20 forward and 20 backward)
    # i is used as an index to get gas entropies
    i = int((T-300)/100)
    
    # DFT computed activation energy for the reactions
    Ea_eV = np.array([0.0,   #0 O2 + 2* = 2O*
                      2.993,
                      0,     #1 NH3 + * = NH3*
                      0.773,
                      0.580, #2 NH3* + O* = NH2* + OH*
                      1.276, 
                      1.449, #3 NH2* + O* = NH* + OH*
                      1.203,
                      0.470, #4 NH* + O* = N* + OH*
                      0.692,
                      0.833, #5 NH3* + OH* = NH2* + H2O*
                      0.995,
                      0.793, #6 NH2* + OH* = NH* + H2O*
                      0.013,
                      0.838, #7 NH* + OH* = N* + H2O*
                      0.525, 
                      0.842, #8 OH* + OH* = O* + H2O*
                      0.308,
                      0,     #9 H2O + * = H2O*
                      0.252,
                      1.182, #10 N* + N* = N2 + *
                      1.813,
                      1.458, #11 N* + O* = NO* + *
                      1.657, 
                      2.329, #12 NO* = NO + *
                      0, 
                      1.625, #13 N* + NO* =N2O*
                      0.444,
                      0.000, #14 N2O* = N2O + *
                      0.095,
                      1.15, #15 NH3* + * = NH2* + H*
                      1.37,
                      1.61, #16 NH2* + * = NH* + H*
                      0.88,
                      1.30, #17 NH* + * = N* + H*
                      0.66,
                      0.50, #18 O* + H* = OH*
                      1.03, 
                      0.96, #19 OH* + H* = H2O*
                      0.64])

    # Gibbs free energy for O2 adsorption
    deltaG_O2 = Ea_eV[0] - Ea_eV[1] - T*deltaS_O2[i]
    # Equilibrium constant for O2 adsorption
    K_O2 = np.exp(-deltaG_O2/kb/T)
    # Forward reaction prefactor estimated with Hertz-Knudsen equation 
    A_O2_f = 1/CT/(2*3.1415*32/NA/1000*kbj*T)**0.5
    
    # Gibbs free energy, equilibrium constant and forward reaction prefactor for NH3 adsorption
    deltaG_NH3 = Ea_eV[2] - Ea_eV[3] - T*deltaS_NH3[i]
    K_NH3 = np.exp(-deltaG_NH3/kb/T)
    A_NH3_f = 1/CT/(2*3.1415*17/NA/1000*kbj*T)**0.5
    
    # Gibbs free energy and equilibrium constant for N* combination
    deltaG_NN = Ea_eV[20] - Ea_eV[21] - T*(-deltaS_N2[i])
    K_NN = np.exp(-deltaG_NN/kb/T)
    
    # Gibbs free energy, equilibrium constant and backward reaction prefactor for NO* desorption
    deltaG_NO = Ea_eV[24] - Ea_eV[25] - T*(-deltaS_NO[i])
    K_NO = np.exp(-deltaG_NO/kb/T)
    A_NO_b = 1/CT/(2*3.1415*30/NA/1000*kbj*T)**0.5

    # Gibbs free energy, equilibrium constant and forward reaction prefactor for H2O adsorption
    deltaG_H2O = Ea_eV[18] - Ea_eV[19] - T*deltaS_H2O[i]
    K_H2O = np.exp(-deltaG_H2O/kb/T)
    A_H2O_f = 1/CT/(2*3.1415*18/NA/1000*kbj*T)**0.5

    # Gibbs free energy, equilibrium constant and forward reaction prefactor for N2O* desorption
    deltaG_N2O = Ea_eV[28] - Ea_eV[29] - T*(-deltaS_N2O[i])
    K_N2O = np.exp(-deltaG_N2O/kb/T)
    A_N2O_b = 1/CT/(2*3.1415*44/NA/1000*kbj*T)**0.5

    # Prefactors of the reactions
    A = np.array([A_O2_f, 
                  A_O2_f/K_O2*np.exp(Ea_eV[1]/kb/T)*P0, 
                  A_NH3_f, 
                  A_NH3_f/K_NH3*np.exp(Ea_eV[3]/kb/T)*P0, 
                  kb*T/h, 
                  kb*T/h, 
                  kb*T/h, 
                  kb*T/h, 
                  kb*T/h, 
                  kb*T/h, 
                  kb*T/h, 
                  kb*T/h, 
                  kb*T/h, 
                  kb*T/h, 
                  kb*T/h, 
                  kb*T/h, 
                  kb*T/h, 
                  kb*T/h,
                  A_H2O_f, 
                  A_H2O_f/K_H2O*np.exp(Ea_eV[19]/kb/T)*P0,
                  kb*T/h, 
                  kb*T/h/K_NN*np.exp((Ea_eV[21]-Ea_eV[20])/kb/T)/P0,                                                 
                  kb*T/h, 
                  kb*T/h, 
                  K_NO*A_NO_b*np.exp((Ea_eV[24]-Ea_eV[25])/kb/T)*P0, 
                  A_NO_b, 
                  kb*T/h,
                  kb*T/h,
                  K_N2O*A_N2O_b*np.exp((Ea_eV[28]-Ea_eV[29])/kb/T)*P0, 
                  A_N2O_b,
                  kb*T/h,
                  kb*T/h,
                  kb*T/h,
                  kb*T/h,
                  kb*T/h,
                  kb*T/h,
                  kb*T/h,
                  kb*T/h,
                  kb*T/h,
                  kb*T/h])

    # Calculate rate constants with Eyring Equation
    k = np.zeros(40)
    for i in range(0, 40):
        k[i] = A[i]*np.exp(-Ea_eV[i]/kb/T) 
    return (k)


# In[ ]:


def get_rates(theta, k):
# returns the rates depending on the current coverages theta
    global PO2, PNH3, PH2O, PNO, PN2O, PN2
    # theta for O*, NH3*, NH2*, OH*, NH*, N*, NO*, H2O*, N2O*, H* and *
    tO    = theta[0]             
    tNH3  = theta[1]             
    tNH2  = theta[2]             
    tOH   = theta[3]             
    tNH   = theta[4]             
    tN    = theta[5]
    tNO   = theta[6]
    tH2O  = theta[7]
    tN2O  = theta[8]
    tH    = theta[9]
    tstar = 1.0 - tO - tNH3 - tNH2 - tOH - tNH - tN - tNO - tH2O - tN2O - tH

    # Caluclate the rates
    rate = np.zeros(40)
    rate[0] = k[0] * PO2  * tstar**2
    rate[1] = k[1] * tO**2
    rate[2] = k[2] * PNH3 * tstar
    rate[3] = k[3] * tNH3
    rate[4] = k[4] * tNH3 * tO
    rate[5] = k[5] * tNH2 * tOH
    rate[6] = k[6] * tNH2 * tO
    rate[7] = k[7] * tNH  * tOH
    rate[8] = k[8] * tNH  * tO
    rate[9] = k[9] * tN   * tOH
    rate[10] = k[10] * tNH3 * tOH
    rate[11] = k[11] * tNH2 * tH2O
    rate[12] = k[12] * tNH2 * tOH
    rate[13] = k[13] * tNH  * tH2O
    rate[14] = k[14] * tNH  * tOH
    rate[15] = k[15] * tN   * tH2O    
    rate[16] = k[16] * tOH**2
    rate[17] = k[17] * tH2O * tO    
    rate[18] = k[18] * PH2O * tstar
    rate[19] = k[19] * tH2O
    rate[20] = k[20] * tN**2
    rate[21] = k[21] * PN2 * tstar**2    
    rate[22] = k[22] * tN * tO
    rate[23] = k[23] * tNO * tstar
    rate[24] = k[24] * tNO
    rate[25] = k[25] * PNO * tstar
    rate[26] = k[26] * tN * tNO
    rate[27] = k[27] * tN2O * tstar
    rate[28] = k[28] * tN2O
    rate[29] = k[29] * PN2O * tstar
    rate[30] = k[30] * tNH3 * tstar
    rate[31] = k[31] * tNH2 * tH
    rate[32] = k[32] * tNH2 * tstar
    rate[33] = k[33] * tNH * tH
    rate[34] = k[34] * tNH * tstar
    rate[35] = k[35] * tN * tH
    rate[36] = k[36] * tO * tH
    rate[37] = k[37] * tOH * tstar
    rate[38] = k[38] * tOH * tH
    rate[39] = k[39] * tH2O
    return rate


# In[ ]:


def get_odes(theta, time, k):
    # returns the system of ODEs d(theta)/dt, calculated at the current value of theta.
    rate = get_rates(theta,k)     # calculate the current rates

    # Time derivatives of theta for O*, NH3*, NH2*, OH*, NH*, N*, NO*, H2O*, N2O* and H*
    dt = np.zeros(10)
    dt[0] = 2*rate[0] - 2*rate[1] - rate[4] + rate[5] - rate[6] + rate[7] - rate[8] + rate[9] + rate[16] - rate[17] - rate[22] + rate[23] - rate[36] + rate[37]    
    dt[1] = rate[2] - rate[3] - rate[4] + rate[5] - rate[10] + rate[11] - rate[30] + rate[31]
    dt[2] = rate[4] - rate[5] - rate[6] + rate[7] + rate[10] - rate[11] - rate[12] + rate[13] + rate[30] - rate[31] - rate[32] + rate[33]
    dt[3] = rate[4] - rate[5] + rate[6] - rate[7] + rate[8] - rate[9] - rate[10] + rate[11] - rate[12] + rate[13] - rate[14] + rate[15] -2*rate[16] + 2*rate[17] + rate[36] - rate[37] - rate[38] + rate[39]  
    dt[4] = rate[6] - rate[7] - rate[8] + rate[9] + rate[12] - rate[13] - rate[14] + rate[15] + rate[32] -rate[33] - rate[34] + rate[35]       
    dt[5] = rate[8] - rate[9] + rate[14] - rate[15] - 2*rate[20] + 2*rate[21] - rate[22] + rate[23] - rate[26] + rate[27] + rate[34] - rate[35]
    dt[6] = rate[22] - rate[23] - rate[24] + rate[25] - rate[26] + rate[27]     
    dt[7] = rate[10] - rate[11] + rate[12] - rate[13] + rate[14] - rate[15] + rate[16] - rate[17] + rate[18] - rate[19] + rate[38] - rate[39]
    dt[8] = rate[26] - rate[27] - rate[28] + rate[29]    
    dt[9] = rate[30] - rate[31] + rate[32] - rate[33] + rate[34] - rate[35] - rate[36] + rate[37] - rate[38] + rate[39]

    return dt


# In[ ]:


mxstep =100000
def solve_ode(k):
# Solve the system of ODEs using scipy.integrate.odeint
# Integrate the ODEs for 1E10 sec (enough to reach steady-state)
    global thetaguess, mxstep
    theta = odeint(get_odes,         # system of ODEs
                   thetaguess,       # initial guess
                   [0,1E10],         # time span
                   args = (k,),      # arguments to get_odes()
                   h0 = 1E-36,       # initial time step
                   mxstep = mxstep,  # maximum number of steps
                   rtol = 1E-12,     # relative tolerance
                   atol = 1E-15      # absolute tolerance
                   )
                             
    return theta [-1,:]


# In[ ]:


tol = 1.0e-20
def solve_findroot(k,theta0):
# Use mpmath’s findroot to solve the model
    global tol
    from mpmath import mp, findroot
    mp.dps = 25
    mp.pretty = True
    def get_findroot_eqns(*args):
        return get_odes(args,0,k)
    theta = findroot(get_findroot_eqns,
                     tuple(theta0),
                     solver='mdnewton',
                     tol=tol,
                     multidimensional=True)
    return np.array(theta)


# In[ ]:


# Initial guess of thetas
thetaguess = np.zeros(10)
# Create a dictionary to store coverages solved with ODE, which will serve as initial guesses for root finder.
thetaode={}

# Solve ODE to get surface coverages from 1200 to 400K 
for T in np.arange(1200, 398, -2):
    # Monitor the process
    if T%100 == 0:
        print (T)
    k = get_rate_constants(T)
    thetaode[T] = solve_ode(k)
    thetaguess = thetaode[T]


# In[ ]:


# Solve algebraic equations for surface coverages and rates at different temperatures
def get_theta(thetaode):
    global Ts, tol
    for T in Ts:
        tol = 1.0e-19
        if T == 400:
            tol = 1.0e-17
        k = get_rate_constants(T)
        # Use mpmath’s findroot to solve the model
        thetas[T] = solve_findroot(k,thetaode[T])
        # Surface coverage of each adsorbate vs T
        cov_O.append(thetas[T][0])
        cov_NH3.append(thetas[T][1])
        cov_NH2.append(thetas[T][2])
        cov_OH.append(thetas[T][3])
        cov_NH.append(thetas[T][4])
        cov_N.append(thetas[T][5])
        cov_NO.append(thetas[T][6])
        cov_H2O.append(thetas[T][7])
        cov_N2O.append(thetas[T][8])
        cov_vac.append(1-np.sum(thetas[T]))
        # Reaction rate of each species vs T
        r = get_rates(thetas[T], k)
        rN2.append((r[20]-r[21]))
        rNO.append(r[24]-r[25])
        rN2O.append((r[28]-r[29]))
        rNH3.append(r[2]-r[3])
    return thetas


# In[ ]:


# Solve algebraic equations from 1200 to 400K 
Ts = np.arange(400, 1250, 50)
cov_O=[]
cov_NH3=[]
cov_NH2=[]
cov_OH=[]
cov_NH=[]
cov_N=[]
cov_NO=[]
cov_H2O=[]
cov_N2O=[]
cov_vac=[]
rN2=[]
rNO=[]
rN2O=[]
rNH3 = []
thetas = {}
thetas = get_theta(thetaode)


# In[ ]:


# Parameters for plotting
from pylab import *
import pylab
from matplotlib.ticker import FormatStrFormatter
font = {'family' : 'Arial',
        'weight' : 'normal',
        'size'   : 15}
rcParams['axes.linewidth'] = 2
rcParams['lines.linewidth'] = 1.5
rcParams['lines.markersize'] = 8
rcParams['xtick.major.width'] = 2
rcParams['ytick.major.width'] = 2
matplotlib.rc('font', **font)
rcParams['figure.figsize'] = (4.2, 4.2)
lw=3
fontsize=15
fslegend=13


# In[ ]:


# Plot surface coverages
plt.plot(Ts, cov_O, 'o-',label='O',c='C3')
plt.plot(Ts, cov_N, 'o-',label='N',c='C0')
plt.plot(Ts, cov_NO, 'o-',label='NO',c='C1')
plt.plot(Ts, cov_vac, 'o-',label='*',c='C7')
plt.legend()
plt.xlabel('Temperature (K)')
plt.ylabel(r'Coverage $\theta$')
# plt.savefig("Pt_211_coverages.pdf",bbox_inches='tight')
plt.show()


# In[ ]:


# Plot selectivities
ratioN2 = [2*x/(2*x+y+2*z) for x,y,z in zip(rN2, rNO, rN2O)]
ratioNO = [y/(2*x+y+2*z) for x,y,z in zip(rN2, rNO, rN2O)]
ratioN2O = [2*z/(2*x+y+2*z) for x,y,z in zip(rN2, rNO, rN2O)]
plt.plot(Ts, ratioN2, 'o-',label=r'N$_{2}$')
plt.plot(Ts, ratioNO, 'o-',label=r'NO')
plt.plot(Ts, ratioN2O, 'o-',label=r'N$_{2}$O',c='violet')
plt.legend(loc='center right')
plt.xlabel('Temperature (K)')
plt.ylabel(r'Selectivity (%)')
pylab.ylim([-0.05,1.05])
plt.show()
# ratioN2


# In[ ]:


# Plot selectivies for all 211 surfaces
ratioN2_Pt211 = [0.9999933628152619, 0.9998525521373576, 0.9979985960477918, 0.9845277845439252, 0.9307561749445767, 0.7722979245361723, 0.4854640966756623, 0.1750920563508653, 0.04285820362743809, 0.008452069884490035, 0.0022916140770488824, 0.0007817237545473926, 0.0004299361504449355, 0.0002890644982857831, 0.00022146236855441633, 0.00017969010557194757, 0.00015275850128119355]
ratioN2_Pd211 = [0.9999999750071321, 0.9999991176397156, 0.9999803270795841, 0.9996286495679582, 0.9971812016996997, 0.9787926868620764, 0.9271611513262537, 0.7715612003961343, 0.6061678339033149, 0.36224651380625494, 0.21368438154866184, 0.07413353518079177, 0.030764133488777146, 0.012165051799805267, 0.009540186434243862, 0.00707513921616397, 0.006682448938547052]
ratioN2_Rh211 = [0.9999999999999952, 0.9999999999999718, 0.9999999999996609, 0.9999999999915251, 0.999999999868336, 0.9999999985413093, 0.9999999896390123, 0.999999932795935, 0.9999996502502384, 0.9999977148291653, 0.9999877581870187, 0.9999188512189722, 0.9996398863323889, 0.9981475608799903, 0.9937477373363371, 0.9759781913316655, 0.9385449357817264]
ratioNO_Pt211 = [1.8062011260768015e-10, 1.2515691955659034e-07, 2.823200003950442e-05, 0.002013133112233182, 0.02572598343596224, 0.14964619082468725, 0.4249903584328059, 0.7808439425105766, 0.9448683770130145, 0.9895720829149695, 0.9972039731902399, 0.9990783521424662, 0.9994957601897861, 0.9996688097790754, 0.999746897808938, 0.9997982438501616, 0.9998288993735306]
ratioNO_Pd211 = [1.4645804939495482e-08, 7.335612793005221e-07, 1.817463583737807e-05, 0.00036061318870366063, 0.0027688835358813473, 0.021015506437940146, 0.07232215807404657, 0.22736426041469843, 0.39213913622127555, 0.6358214505343198, 0.7843865280197255, 0.9247411055593618, 0.9685859893239978, 0.9875498376883693, 0.9902563876279811, 0.992804960230582, 0.9932253801157075]
ratioNO_Rh211 = [3.321544297268392e-19, 1.193056037304727e-17, 4.531997333768722e-16, 3.5672625014833755e-14, 1.1700133761205384e-12, 3.2990032324105526e-11, 5.241076632636767e-10, 1.0104445762418896e-08, 1.1340757505508875e-07, 1.3860689346449208e-06, 9.572409150571687e-06, 7.344628299566072e-05, 0.00034180221857734556, 0.0018091828913342875, 0.006165674097273553, 0.023850643492324165, 0.06116970211662987]
ratioN2O_Pt211 = [6.637004118090205e-06, 0.00014732270572288844, 0.0019731719521686023, 0.013459082343841754, 0.043517841619461116, 0.07805588463914036, 0.08954554489153185, 0.04406400113855807, 0.012273419359547426, 0.0019758472005404737, 0.0005044127327113309, 0.00013992410298627338, 7.430365976887386e-05, 4.212572263883023e-05, 3.163982250754064e-05, 2.2066044266486274e-05, 1.8342125188173448e-05]
ratioN2O_Pd211 = [1.034706293595437e-08, 1.4879900523632844e-07, 1.4982845785538815e-06, 1.0737243338084928e-05, 4.9914764418986914e-05, 0.00019180669998356708, 0.0005166905996996825, 0.001074539189167238, 0.0016930298754096384, 0.0019320356594253142, 0.001929090431612524, 0.0011253592598463676, 0.0006498771872250795, 0.00028511051182535243, 0.0002034259377750304, 0.00011990055325409633, 9.217094574545478e-05]
ratioN2O_Rh211 = [4.704711827382017e-15, 2.8193522984411122e-14, 3.387817200961216e-13, 8.439149034897317e-12, 1.3049400010460677e-10, 1.4257006658414876e-09, 9.836880204847514e-09, 5.7099619197203216e-08, 2.363421864977329e-07, 8.991019000214671e-07, 2.6694038306267306e-06, 7.702498032041363e-06, 1.831144903371132e-05, 4.325622867534055e-05, 8.658856638931885e-05, 0.00017116517601020717, 0.0002853621016438962]

plt.plot(Ts, ratioN2O_Pt211, ':',linewidth = 3, c='C0')
plt.plot(Ts, ratioN2O_Pd211, ':',linewidth = 3, c='C1')
plt.plot(Ts, ratioN2O_Rh211, ':', c = 'violet',linewidth = 3)

plt.plot(Ts, ratioNO_Pt211, '--',linewidth = 3, c='C0')
plt.plot(Ts, ratioNO_Pd211, '--',linewidth = 3, c='C1')
plt.plot(Ts, ratioNO_Rh211, '--', c = 'violet',linewidth = 3)

plt.plot(Ts, ratioN2_Pt211, '-',label=r'Pt(211)',linewidth = 3)
plt.plot(Ts, ratioN2_Pd211, '-',label=r'Pd(211)',linewidth = 3)
plt.plot(Ts, ratioN2_Rh211, '-',label=r'Rh(211)', c = 'violet',linewidth = 3)

plt.legend(loc='center right',fontsize=12)
plt.xlabel('Temperature (K)')
plt.ylabel(r'N$_2$ Selectivity')
pylab.ylim([-0.05,1.05])
# plt.savefig("selectivity_211_comparison.pdf",bbox_inches='tight')
plt.show()


# In[ ]:


# Plot the rates for all 211 surfaces
rNH3Pt211 = np.array([1.46172145e-03, 5.00978473e-03, 7.05701309e-03, 1.27804456e-02,
                       4.38727626e-02, 2.01994992e-01, 8.12499652e-01, 2.46428158e+00,
                       3.94825804e+00, 3.81997276e+00, 4.10332657e+00, 4.79963591e+00,
                       7.94760764e+00, 1.44141045e+01, 2.68770305e+01, 4.83540553e+01,
                       8.30464220e+01])

rNH3Pd211 = np.array([7.50904020e-06, 2.71259326e-04, 3.38060260e-03, 2.27965879e-02,
                       1.37327348e-01, 5.64378055e-01, 2.50292506e+00, 1.00042147e+01,
                       4.24609375e+01, 1.66027842e+02, 5.16434804e+02, 1.23706635e+03,
                       1.98726248e+03, 2.49310245e+03, 2.96440093e+03, 3.10633897e+03,
                       3.14457279e+03])

rNH3Rh211 = np.array([1.07826722e-09, 3.52530594e-07, 3.71915115e-05, 1.67020335e-03,
                       3.76799578e-02, 4.23130720e-01, 2.15258635e+00, 4.57985049e+00,
                       6.75479360e+00, 7.10641218e+00, 7.96413734e+00, 7.42558492e+00,
                       7.92397103e+00, 7.29908918e+00, 7.79119379e+00, 7.34274750e+00,
                       8.06216539e+00])

plt.semilogy(1/Ts, rNH3Pt211, 'o',label=r'Pt(211)', c='C0', markerfacecolor='None',markersize=12,markeredgewidth='3')
plt.semilogy(1/Ts, rNH3Pd211, 'o',label=r'Pd(211)', c='C1', markerfacecolor='None',markersize=12,markeredgewidth='3')
plt.semilogy(1/Ts, rNH3Rh211, 'o',label=r'Rh(211)', c='violet', markerfacecolor='None',markersize=12,markeredgewidth='3')

plt.legend(loc=(0.02,0.15))
plt.xlabel('T$^{-1}$ (K$^{-1}$)')
plt.ylabel(r'TOF (s$^{-1}$)')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
axes = plt.gca()
axes.set_xticklabels(['',r'$\frac{1}{1000}$',r'$\frac{1}{666}$',r'$\frac{1}{500}$',r'$\frac{1}{400}$'])
pylab.ylim([10**(-20),10**(5)])
# plt.savefig("rate_211_comparison.pdf",bbox_inches='tight')
plt.show()

