import numpy as np
import matplotlib.pyplot as plt

Nlist_same = [64,50,40,32,20,16,10,8,7,6,5,4,3]
fidlist_same = [.448,.461,.478,.529,.546,.526,.596,.525,.671,.583,.786,.822,.520]
times_same = [31,23,18,15,9,7.1,3.6,2.9,2.8,2.5,1.7,1.4,2]

Nlist_shift = [64,50,40,32,20,16,10,8,7,6,5,4,3]
fidlist_shift = [.482,.493,.506,.52,.560,.584,.630,.640,.527,.641,.525,.613,.615]
times_shift =[14.6,11.2,9,7,4,3.1,1.7,1.2,.7,.4,.3,.2,.2]

plt.plot(Nlist_same,fidlist_same,'.',label='full')
plt.plot(Nlist_same,fidlist_shift,'.',label='half')
plt.xlabel("size")
plt.ylabel('fidelity')
plt.legend()
plt.show()


plt.plot(times_same,N_list_same)
plt.plot(times_shift,N_list_same)
plt.show()



