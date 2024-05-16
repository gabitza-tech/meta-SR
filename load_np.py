import numpy as np
import sys

file = sys.argv[1]

enroll_dict = np.load(file, allow_pickle=True)

feat = enroll_dict['concat_features']

labels = np.asarray(enroll_dict['concat_labels'])

slices = np.asarray(enroll_dict['concat_slices'])

patchs = np.asarray(enroll_dict['concat_patchs'])

print(feat.shape)
print(labels.shape)
print(slices.shape)
print(patchs.shape)
for i in range(200,208):
   print([feat[i],labels[i],slices[i],patchs[i]])

