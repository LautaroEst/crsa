


import numpy as np
from itertools import product


p = 3

meanings_A = ["".join(l) for l in product("AB", repeat=p)]
meanings_B = ["".join(n) for n in product("12", repeat=p)]
categories = ["No (A,1) pair"] + [str(i+1) for i in range(p)]
utterances_A = [str(i+1) for i in range(p)]
utterances_B = [str(i+1) for i in range(p)]

prior = np.zeros((2**p,2**p,p+1))
for i in range(2**p):
    for j in range(2**p):
        # check if (meaninings_A[i][k] == "A" and meanings_B[j][k] == "1") happens only once for a fixed i,j for k in range(p)
        count = 0
        A1_idx = None
        for k in range(p):
            if meanings_A[i][k] == "A" and meanings_B[j][k] == "1":
                count += 1
                A1_idx = k
        if count == 1:
            prior[i,j,A1_idx+1] = 1
        elif count == 0:
            prior[i,j,0] = 1

lexicon_A = np.zeros((p,len(meanings_A)))
for u, utt in enumerate(utterances_A):
    for i, meaning in enumerate(meanings_A):
        if meaning[int(u)] == "A":
            lexicon_A[u,i] = 1
lexicon_A[:,-1] = 1

lexicon_B = np.zeros((p,len(meanings_B)))
for u, utt in enumerate(utterances_B):
    for i, meaning in enumerate(meanings_B):
        if meaning[int(u)] == "1":
            lexicon_B[u,i] = 1
lexicon_B[:,-1] = 1


print(meanings_A)
print(meanings_B)
print(categories)
print(utterances_A)
print(utterances_B)
print(prior)
print(lexicon_A)
print(lexicon_B)
        

                
