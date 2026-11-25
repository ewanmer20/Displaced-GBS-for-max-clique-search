 
import strawberryfields as sf
from strawberryfields.ops import *
import numpy as np



qkd = sf.Program(2)
def qkd_wfhd(alpha_amp, alpha_phase, reflectivity, cutoff=2):
    with qkd.context as q:
    # prepare the input single photon and coherent state
        Fock(1) | q[0]
        Dgate(r=alpha_amp, phi=alpha_phase) | q[1]

    # linear interferometer
        BSgate(theta=np.arcsin(reflectivity),phi=np.pi/2) | (q[0], q[1])    


    eng = sf.Engine(backend="fock", backend_options={"cutoff_dim":cutoff})
    results = eng.run(qkd)
    state = results.state
    return state.all_fock_probs()

if __name__ == "__main__":
    alpha_amp = 0.5
    alpha_phase = 0.5
    reflectivity = 0.5
    print(qkd_wfhd(alpha_amp, alpha_phase, reflectivity))