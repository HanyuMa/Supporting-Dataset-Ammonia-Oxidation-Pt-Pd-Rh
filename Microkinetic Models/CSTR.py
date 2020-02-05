
# coding: utf-8

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
PNH3_0 = 1000 * 10 **(-6) * P
PO2_0 = 0.02 * P
PH2O_0 = 0.05 * P

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
adsorbate_entropy = pd.read_csv("adsorbate_entropy_Pt211.csv")


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
                      0.0,   #1 NH3 + * = NH3*
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
                      0.0,   #9 H2O + * = H2O*
                      0.252,
                      1.182, #10 N* + N* = N2 + *
                      1.813,
                      1.458, #11 N* + O* = NO* + *
                      1.657, 
                      2.329, #12 NO* = NO + *
                      0.0, 
                      1.625, #13 N* + NO* =N2O*
                      0.444,
                      0.000, #14 N2O* = N2O + *
                      0.095,
                      1.15,  #15 NH3* + * = NH2* + H*
                      1.37,
                      1.61,  #16 NH2* + * = NH* + H*
                      0.88,
                      1.30,  #17 NH* + * = N* + H*
                      0.66,
                      0.50,  #18 O* + H* = OH*
                      1.03, 
                      0.96,  #19 OH* + H* = H2O*
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
    # theta for O*, NH3*, NH2*, OH*, NH*, N*, NO*, H2O*, N2O*, H* and *
    # Pressure for O2, NH3, NO, N2O, H2O, and N2
    global PO2, PNH3, PH2O, PNO, PN2O, PN2
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
    PO2  = theta[10]
    PNH3 = theta[11]
    PNO = theta[12]
    PN2O = theta[13]
    PH2O = theta[14]
    PN2 = theta[15]
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
    global sites, PO2, PNH3, PH2O, PNO, PN2O, PN2
    # returns the system of ODEs d(theta)/dt, calculated at the current value of theta.
    rate = get_rates(theta,k)     # calculate the current rates

    # Time derivatives of theta for O*, NH3*, NH2*, OH*, NH*, N*, NO*, H2O* and N2O*
    dt = np.zeros(16)
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
    
    # Time derivatives of P for O2, NH3, NO, N2O, H2O and N2
    dt[10] = 1/tau * (PO2_0 - PO2) + R * T * sites * (rate[1] - rate[0])
    dt[11] = 1/tau * (PNH3_0 - PNH3) + R * T * sites * (rate[3] - rate[2])
    dt[12] = 1/tau * (0 - PNO) + R * T * sites * (rate[24] - rate[25])
    dt[13] = 1/tau * (0 - PN2O) + R * T * sites * (rate[28] - rate[29])
    dt[14] = 1/tau * (PH2O_0 - PH2O) + R * T * sites * (rate[19] - rate[18])
    dt[15] = 1/tau * (0 - PN2)+ R * T * sites * (rate[20] - rate[21])
    return dt


# In[ ]:


mxstep = 50000000 # Large number of mxstep to approach steady-state
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


tol = 1.0e-17
def solve_findroot(k,theta0):
# Use mpmath’s findroot to solve the model
    global tol, sites
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


def get_theta(thetaode):
    global thetaguess
    k = get_rate_constants(T)
    try:
        thetas = solve_findroot(k,thetaode)
    except:
        thetas = thetaode
    cov_O.append(thetas[0])
    cov_NH3.append(thetas[1])
    cov_NH2.append(thetas[2])
    cov_OH.append(thetas[3])
    cov_NH.append(thetas[4])
    cov_N.append(thetas[5])
    cov_NO.append(thetas[6])
    cov_H2O.append(thetas[7])
    cov_N2O.append(thetas[8])
    cov_vac.append(1-np.sum(thetas[:9]))
    P_NH3.append(thetas[11])
    P_N2.append(thetas[15])
    P_NO.append(thetas[12])
    P_N2O.append(thetas[13])
    r = get_rates(thetas, k)
    r_N2.append (r[20]-r[21])
    r_NO.append(r[24]-r[25])
    r_N2O.append(r[28]-r[29])
    return thetas


# In[ ]:


# Solve the pressures of CSTR reactor at 500 K
T = 500
thetaguess = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, PO2_0, PNH3_0, 0, 0, PH2O_0, 0]
# Range of residence time
tau_range = np.arange(-4,7,0.5)
# Site density, unit: mol / m3
sites = 1 
thetaode={} # Create a dictionary to store coverages and pressures solved with ODE.
thetas = {} # Dictionary to store coverages and pressures solved with algebraic equations.

# Create lists to store coverages and pressures at different residence time.
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
r_NH3 = []
r_N2 = []
r_NO = []
r_N2O = []
P_NH3 = []
P_N2 = []
P_NO = []
P_N2O = []

# Solve to get coverages and pressures
for tau in tau_range:
    print (tau)
    tau = 10**tau
    k = get_rate_constants(T)
    thetaode[tau] = solve_ode(k)
    thetaguess = deepcopy(thetaode[tau])
    thetas[tau] = get_theta(thetaode[tau])


# In[ ]:


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
from matplotlib.ticker import FormatStrFormatter
plt.plot(tau_range, cov_O, 'o-',label='O',c='C3')
plt.plot(tau_range, cov_N, 'o-',label='N',c='C0')
plt.plot(tau_range, cov_NO, 'o-',label='NO',c='C1')
plt.plot(tau_range, cov_vac, 'o-',label='*',c='C7')
plt.legend()
plt.xlabel('Temperature (K)')
plt.ylabel(r'Coverage $\theta$')
plt.show()


# In[ ]:


# Calculate the conversion of NH3
Ptconversions = 1 - np.asarray(P_NH3)/PNH3_0
Ptconversions


# In[ ]:


# Plot the conversions at different residence time
Rhconversions = array([2.5814390625100714e-09, 0.0000004825110262243590101816182,
       0.00000152583385145659606926158, 0.000004825110426523546486267321,
       0.00001525834015778026743153133, 0.00004825112069109338725448313,
       0.0001525835658469216142801282, 0.0004825128496972387512456847,
       0.001525852087153176743737011, 0.004825292812524307,
       0.01526016491629285470653323, 0.04826939386959073,
       0.15276675039653353, 0.484276053459443, 0.9831014799458814,
       0.9975881771386867, 0.9993683505921652, 0.9997810269737402,
       0.9998992846447942, 0.9999394580628542, 0.999956086602898,
       0.9999653312771535])
Pdconversions = array([0.00001386877197263051044176255, 0.00004385380946724139128723086,
       0.0001386517903026879214732913, 0.0004382068232321517776801665,
       0.001383271171402948672346054, 0.004349923914436538408926,
       0.01351875859911123889411757, 0.04057272851538973,
       0.1110998834598295791689881, 0.25273024793082455,
       0.4476284680526954, 0.6346063858885884, 0.775030381923709,
       0.8670265226232001, 0.9231437028955892, 0.9561203089693989,
       0.9751167221533423, 0.9859421840454934, 0.9920747810830017,
       0.9955374943978135, 0.9974891295144912, 0.9985823322515655])


plt.plot(tau_range[:], Ptconversions[:], '-',label='Pt(211)', c='C0', linewidth = 3)
plt.plot(tau_range[:], Pdconversions[:], '-',label='Pd(211)', c='C1', linewidth = 3)
plt.plot(tau_range[:], Rhconversions, '-',label='Rh(211)', c='C6', linewidth = 3)
plt.legend(fontsize=14)
pylab.xlim([-4,6.5])

axes = plt.gca()
axes.set_xticklabels(['',r'10$^{-2.5}$',r'10$^{0}$',r'10$^{2.5}$',r'10$^{5.0}$'])

plt.xlabel('Residence Time (s)')
plt.ylabel(r'NH$_3$ Conversion')
# plt.savefig("CSTR_convesion_500K.pdf",bbox_inches='tight')

plt.show()


# In[ ]:


# Calculate the ratio(or selectivity) of the reaction products
ratioN2_Pt211 = [2*x/(2*x+y+2*z) for x,y,z in zip(P_N2, P_NO, P_N2O)]
ratioNO_Pt211 = [y/(2*x+y+2*z) for x,y,z in zip(P_N2, P_NO, P_N2O)]
ratioN2O_Pt211 = [2*z/(2*x+y+2*z) for x,y,z in zip(P_N2, P_NO, P_N2O)]


# In[ ]:


# Plot the selectivity
plt.plot(tau_range, ratioN2_Pt211, '-',label=r'N$_2$',linewidth = 3, c='C0')
plt.plot(tau_range, ratioNO_Pt211, '-',label=r'NO',linewidth = 3, c='C1')
plt.plot(tau_range, ratioN2O_Pt211, '-',label=r'N$_2$O',linewidth = 3, c='C6')
pylab.xlim([-4,6.5])
axes = plt.gca()
axes.set_xticklabels(['',r'10$^{-2.5}$',r'10$^{0}$',r'10$^{2.5}$',r'10$^{5.0}$'])
plt.xlabel('Residence Time (s)')
plt.legend()
plt.ylabel(r'Selectivity')
pylab.ylim([-0.05,1.05])
# plt.savefig("Pt_CSTR_Selectivity_500K.pdf",bbox_inches='tight')
plt.show()


# In[ ]:


ratioN2O_Pd211 = [1.4983239961111777e-06, 1.4983819497041997e-06, 1.498529334703836e-06, 1.4989660886033121e-06, 1.5003317590417966e-06, 1.5046304415296583e-06, 1.5180777375481613e-06, 1.5592525231179122e-06, 1.6783673250365006e-06, 1.985439682264929e-06, 2.6652895825222795e-06, 3.998770246949481e-06, 6.456996431750305e-06, 1.0881239121641258e-05, 1.8779291908126188e-05, 3.2841524675586194e-05, 5.785765149931561e-05, 0.00010234769166587434, 0.00018146680302792074, 0.00032212887021114146, 0.0005724272259541348, 0.0010160451719505338]
ratioN2_Pd211 = [0.9999870307341874, 0.9999921198129845, 0.9999958451058829, 0.9999975669951549, 0.9999981927376528, 0.9999983963528741, 0.999998449663795, 0.9999984297001812, 0.9999983172942467, 0.9999980123099714, 0.9999973330110538, 0.9999959994377672, 0.9999935406406253, 0.9999891152077037, 0.9999812149548917, 0.9999671487672224, 0.9999421255847578, 0.9998976229853392, 0.9998184815319875, 0.9996777797338549, 0.9994274106707923, 0.9989836674473163]
ratioNO_Pd211 = [1.1470941816568577e-05, 6.38180506567788e-06, 2.6563647824656513e-06, 9.340387566245446e-07, 3.069305881742533e-07, 9.90166844374555e-08, 3.2258467462191366e-08, 1.104729561524175e-08, 4.338428348864621e-09, 2.2503462699524164e-09, 1.6993636882235822e-09, 1.7919859500312425e-09, 2.362942962741469e-09, 3.5531745548436866e-09, 5.753200228325441e-09, 9.708102077592724e-09, 1.6763742797422377e-08, 2.932299493939156e-08, 5.1664984617757806e-08, 9.13959340219139e-08, 1.6210325354447786e-07, 2.873807332118674e-07]


# In[ ]:


plt.plot(tau_range, ratioN2_Pd211, '-',label=r'N$_2$',linewidth = 3, c='C0')
plt.plot(tau_range, ratioNO_Pd211, '-',label=r'NO',linewidth = 3, c='C1')
plt.plot(tau_range, ratioN2O_Pd211, '-',label=r'N$_2$O',linewidth = 3, c='C6')

pylab.xlim([-4,6.5])

axes = plt.gca()
axes.set_xticklabels(['',r'10$^{-2.5}$',r'10$^{0}$',r'10$^{2.5}$',r'10$^{5.0}$'])

plt.xlabel('Residence Time (s)')
plt.legend()
plt.ylabel(r'Selectivity')
pylab.ylim([-0.05,1.05])
# plt.savefig("Pd_CSTR_Selectivity_500K.pdf",bbox_inches='tight')
plt.show()


# In[ ]:


ratioN2O_Rh211 = [1.2412000081833892e-05, 3.3877851934719386e-13, 3.3877159868315204e-13, 3.387497154800791e-13, 3.3868053422570135e-13, 3.384619580245568e-13, 3.377726951132382e-13, 3.356122287641974e-13, 3.2896636917251363e-13, 3.0965004782241066e-13, 2.61654775554797e-13, 1.778782351407184e-13, 9.46514788958096e-14, 5.5179426598041385e-14, 5.682021124064217e-13, 9.987475674622962e-13, 9.224184691641474e-13, 7.970577578204315e-13, 7.148751873210605e-13, 6.633051367365525e-13, 6.278496848857708e-13, 5.98081302843984e-13]
ratioN2_Rh211 = [0.9999875502893368, 0.9999999999996609, 0.9999999999996609, 0.9999999999996609, 0.9999999999996609, 0.9999999999996612, 0.999999999999662, 0.9999999999996644, 0.9999999999996709, 0.9999999999996902, 0.9999999999997384, 0.999999999999822, 0.9999999999999054, 0.9999999999999448, 0.9999999999994318, 0.9999999999990011, 0.9999999999990776, 0.999999999999203, 0.9999999999992851, 0.9999999999993366, 0.9999999999993722, 0.9999999999994019]
ratioNO_Rh211 = [3.771058141316936e-08, 4.505922763586635e-16, 4.450555554555912e-16, 4.284089451625685e-16, 3.830967629511952e-16, 2.870806245565139e-16, 1.6015890574816527e-16, 6.680073877943326e-17, 2.3511781968181685e-17, 7.729966571143955e-18, 2.4955176561083912e-18, 8.155388595898813e-19, 2.8537447839910453e-19, 1.396514612387953e-19, 1.3169329293626499e-18, 2.1310321550827572e-18, 1.949526553631993e-18, 1.7280395216586231e-18, 1.6091761155590363e-18, 1.5547272415582599e-18, 1.5306880307752336e-18, 1.5199553668165203e-18]


# In[ ]:


plt.plot(tau_range, ratioN2_Rh211, '-',label=r'N$_2$',linewidth = 3, c='C0')
plt.plot(tau_range, ratioNO_Rh211, '-',label=r'NO',linewidth = 3, c='C1')
plt.plot(tau_range, ratioN2O_Rh211, '-',label=r'N$_2$O',linewidth = 3, c='C6')

pylab.xlim([-4,6.5])

axes = plt.gca()
axes.set_xticklabels(['',r'10$^{-2.5}$',r'10$^{0}$',r'10$^{2.5}$',r'10$^{5.0}$'])

plt.xlabel('Residence Time (s)')
plt.legend()
plt.ylabel(r'Selectivity')
pylab.ylim([-0.05,1.05])
# plt.savefig("Rh_CSTR_Selectivity_500K.pdf",bbox_inches='tight')
plt.show()

