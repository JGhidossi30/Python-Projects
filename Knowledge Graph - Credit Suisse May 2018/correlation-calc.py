import pickle
from functools import reduce

from scipy.stats.stats import pearsonr

inputs = pickle.load(open("input.pkl", "rb"))
revs = pickle.load(open("output.pkl", "rb"))
revs = reduce(lambda x,y: x+y,revs)

print(inputs)
print(inputs[0][10])

correlations = []

for row in range(11):
    r = []
    for col in range(len(inputs)):
        r.append(inputs[col][row])
    correlations.append(pearsonr(r, revs))
print(correlations)
corrs = []
for i in range(len(correlations)):
    corrs.append(correlations[i][0])
print (corrs)

winter = []
dom_subs = []
for col in range(len(inputs)):
    dom_subs.append(inputs[col][1])
    winter.append(inputs[col][9])
corr = pearsonr(winter, dom_subs)
print(corr)

