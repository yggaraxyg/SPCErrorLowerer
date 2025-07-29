import sys
import numpy as np
import scipy
import statistics

if (len(sys.argv)!=6):
    print("paired (0,1), less or greater (0,1), alpha, and 2  Filenames please!")
    exit(-1)

np.set_printoptions(threshold=np.inf, precision = 10000, suppress=True)

Paired = (int(sys.argv[1])>0)
lessgreat = int(sys.argv[2])
alpha = float(sys.argv[3])
data1 = sys.argv[4]
data2 = sys.argv[5]

info1 = np.load(data1)
error1 = info1['error']
info2 = np.load(data2)
error2 = info2['error']

for i in range(len(error1)):
    error1[i] = abs(error1[i])
for i in range(len(error2)):
    error2[i] = abs(error2[i])

print(f"Null: Mu of {data1} == Mu of {data2}")

#print(error1)
#print(error2)
if(lessgreat>0):
    alternative = 'greater'
    print(f"Alternate: Mu of {data1} > Mu of {data2}")  
elif(lessgreat==0):
    alternative = 'two-sided'
    print(f"Alternate: Mu of {data1} != Mu of {data2}")  
else:
    alternative = 'less'
    print(f"Alternate: Mu of {data1} < Mu of {data2}")  
  
if(Paired):
    res = scipy.stats.ttest_rel(error1,error2, alternative = alternative)
else:
    v1 = statistics.variance(error1)
    v2 = statistics.variance(error2)
    F = v1/v2
    pvalf = 1-scipy.stats.f.cdf(F, len(error1)-1, len(error2)-1)
    #print(pvalf)
    areVariancesSame = (pvalf <= alpha)
    print("pvalf: "+str(pvalf)+", alpha: "+str(alpha)+" Same Variance? "+str(areVariancesSame)+" according to F-test.")
    res = scipy.stats.ttest_ind(error1,error2,equal_var=areVariancesSame, alternative = alternative)

print(res)
ples= res.pvalue<=alpha
print("pval: "+str(res.pvalue)+", alpha: "+str(alpha)+", Is P-value less than alpha: "+str(ples))
