import matplotlib.pyplot as plt
import numpy as np

precision = [0.7854226214781543, 0.789305912596401, 0.7870010235414534, 0.7900649283726683, 0.7887785399815705, 0.7860427097169715, 0.7908194701170672, 0.7881486030089039, 0.7915466886055121, 0.7873822204014748, 0.787229681254484, 0.7921971252566735, 0.7873894260440238, 0.7883864337101747, 0.7904898839478278, 0.7876352395672334, 0.7855530474040632, 0.7882933854541714, 0.7903426791277258, 0.7881426202321725, 0.792590279945884, 0.7898985717242807, 0.7928144967005342, 0.7971503957783641, 0.8013421389007244, 0.797327632282202, 0.806758530183727, 0.8168161434977579, 0.824756446991404, 0.8313831032061784, 0.8396745632926538, 0.8465660009741841, 0.8484960552268245, 0.8565692056719789, 0.86224555114582, 0.872669992123917, 0.8808982756316001, 0.8879616963064295, 0.8931671800616074, 0.9016793454858619, 0.9088096641131408, 0.9150090415913201, 0.9198318804483188, 0.9265509482481518, 0.9356881039653449, 0.9411764705882353, 0.9488029465930018, 0.9552950526524936, 0.9673865138827678, 0.9801792017377138, 1.0]
recall = [0.8548888888888889, 0.8528888888888889, 0.8543333333333333, 0.8517777777777777, 0.856, 0.8547777777777777, 0.8556666666666667, 0.8556666666666667, 0.8552222222222222, 0.8542222222222222, 0.8534444444444444, 0.8573333333333333, 0.8505555555555555, 0.8523333333333334, 0.8552222222222222, 0.8493333333333334, 0.8506666666666667, 0.8514444444444444, 0.8456666666666667, 0.8448888888888889, 0.8462222222222222, 0.848, 0.841, 0.8392222222222222, 0.8358888888888889, 0.8287777777777777, 0.8196666666666667, 0.8095555555555556, 0.7995555555555556, 0.7894444444444444, 0.7797777777777778, 0.7724444444444445, 0.7647777777777778, 0.7584444444444445, 0.7483333333333333, 0.7386666666666667, 0.7322222222222222, 0.7212222222222222, 0.7087777777777777, 0.698, 0.6854444444444444, 0.6746666666666666, 0.6565555555555556, 0.6405555555555555, 0.624, 0.5991111111111111, 0.5724444444444444, 0.5342222222222223, 0.48777777777777775, 0.4011111111111111, 0.0005555555555555556]
TP = [7347, 7337, 7328, 7321, 7350, 7337, 7332, 7351, 7320, 7334, 7345, 7354, 7308, 7325, 7354, 7304, 7307, 7333, 7288, 7280, 7292, 7328, 7263, 7261, 7264, 7199, 7135, 7079, 7018, 6950, 6873, 6819, 6781, 6720, 6644, 6574, 6525, 6425, 6311, 6232, 6122, 6032, 5879, 5742, 5597, 5375, 5140, 4795, 4383, 3604, 5]
FP =[2102, 2049, 2081, 2037, 2063, 2094, 2037, 2070, 2027, 2076, 2076, 2024, 2067, 2059, 2040, 2061, 2090, 2058, 2019, 2044, 1993, 2030, 1978, 1922, 1865, 1896, 1767, 1634, 1529, 1441, 1340, 1260, 1229, 1143, 1076, 970, 891, 819, 763, 685, 619, 564, 515, 457, 386, 337, 278, 225, 148, 73, 0]
TN = [92898, 92951, 92919, 92963, 92937, 92906, 92963, 92930, 92973, 92924, 92924, 92976, 92933, 92941, 92960, 92939, 92910, 92942, 92981, 92956, 93007, 92970, 93022, 93078, 93135, 93104, 93233, 93366, 93471, 93559, 93660, 93740, 93771, 93857, 93924, 94030, 94109, 94181, 94237, 94315, 94381, 94436, 94485, 94543, 94614, 94663, 94722, 94775, 94852, 94927, 95000]
FN = [1306, 1324, 1311, 1334, 1296, 1307, 1299, 1299, 1303, 1312, 1319, 1284, 1345, 1329, 1303, 1356, 1344, 1337, 1389, 1396, 1384, 1368, 1431, 1447, 1477, 1541, 1623, 1714, 1804, 1895, 1982, 2048, 2117, 2174, 2265, 2352, 2410, 2509, 2621, 2718, 2831, 2928, 3091, 3235, 3384, 3608, 3848, 4192, 4610, 5390, 8995]
WP = [347, 339, 361, 345, 354, 356, 369, 350, 377, 354, 336, 362, 347, 346, 343, 340, 349, 330, 323, 324, 324, 304, 306, 292, 259, 260, 242, 207, 178, 155, 145, 133, 102, 106, 91, 74, 65, 66, 68, 50, 47, 40, 30, 23, 19, 17, 12, 13, 7, 6, 0]

TP_arr = np.array(TP)
WP_arr = np.array(WP)
FN_arr = np.array(FN)
FP_arr = np.array(FP)
TN_arr = np.array(TN)

TPR_arr = TP_arr / (TP_arr + WP_arr + FN_arr)
WPR_arr = WP_arr / (TP_arr + WP_arr + FN_arr)
FPR_arr = FP_arr / (FP_arr + TN_arr)

r20Precision_arr = TPR_arr / (TPR_arr + WPR_arr + 20 * FPR_arr)
r20Precision = r20Precision_arr.tolist()
TPR = TPR_arr.tolist()
WPR = WPR_arr.tolist()
FPR = FPR_arr.tolist()

plt.figure(figsize=(8, 6))

# Plot Precision vs Recall
plt.scatter(recall, precision, label='Precision', color='blue')
plt.plot(recall, precision, linestyle='-', color='blue', alpha=0.5)

# Plot rPrecision vs Recall
plt.scatter(recall, r20Precision, label='r20Precision', color='red')
plt.plot(recall, r20Precision, linestyle='-', color='red', alpha=0.5)

plt.xlabel('Recall')
plt.ylabel('Precision / r20Precision')
plt.title('Precision and r20Precision vs Recall')
plt.grid(True)
plt.legend()

plt.show()

plt.figure(figsize=(8, 6))

# Plot TPR vs FPR
plt.scatter(FPR, TPR, color='blue')
plt.plot(FPR, TPR, linestyle='-', color='blue', alpha=0.5)

plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('TPR vs FPR')
plt.grid(True)

plt.show()

# Threshold range: 0.4 - 1.0

# Define the indices of the points to keep
indices_to_keep = []  # Indices of the points to keep

# Combine the specific points to keep with the rest of the points
indices_to_keep.extend(range(20, len(recall)))

# Select the subset of points based on the defined indices
undersampled_recall = [recall[i] for i in indices_to_keep]
undersampled_precision = [precision[i] for i in indices_to_keep]
undersampled_r20Precision = [r20Precision[i] for i in indices_to_keep]
undersampled_FPR = [FPR[i] for i in indices_to_keep]
undersampled_TPR = [TPR[i] for i in indices_to_keep]

# Plot Precision vs Recall
plt.figure(figsize=(8, 6))
plt.scatter(undersampled_recall, undersampled_precision, label='Precision', color='blue')
plt.plot(undersampled_recall, undersampled_precision, linestyle='-', color='blue', alpha=0.5)

# Plot rPrecision vs Recall
plt.scatter(undersampled_recall, undersampled_r20Precision, label='r20Precision', color='red')
plt.plot(undersampled_recall, undersampled_r20Precision, linestyle='-', color='red', alpha=0.5)

plt.xlabel('Recall')
plt.ylabel('Precision / r20Precision')
plt.title('Precision and r20Precision vs Recall')
plt.grid(True)
plt.legend()
plt.show()

# Plot TPR vs FPR
plt.figure(figsize=(8, 6))
plt.scatter(undersampled_FPR, undersampled_TPR, color='blue')
plt.plot(undersampled_FPR, undersampled_TPR, linestyle='-', color='blue', alpha=0.5)

plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('TPR vs FPR')
plt.grid(True)
plt.show()

# Cut off when not monotonically decreasing

# Reverse all lists
recall = recall[::-1]
precision = precision[::-1]
r20Precision = r20Precision[::-1]
FPR = FPR[::-1]
TPR = TPR[::-1]

# Find the index where precision is no longer monotonically decreasing
cutoff_index = None
for i in range(1, len(precision)):
    if precision[i] >= precision[i - 1]:
        cutoff_index = i - 1
        break

# If cutoff_index is not found, it means precision is monotonically decreasing throughout
if cutoff_index is None:
    cutoff_index = len(precision) - 1

# Cut off the lists at the identified index
recall = recall[:cutoff_index + 1]
precision = precision[:cutoff_index + 1]
r20Precision = r20Precision[:cutoff_index + 1]
FPR = FPR[:cutoff_index + 1]
TPR = TPR[:cutoff_index + 1]

# Plot Precision vs Recall
plt.figure(figsize=(8, 6))
plt.scatter(recall, precision, label='Precision', color='blue')
plt.plot(recall, precision, linestyle='-', color='blue', alpha=0.5)

# Plot rPrecision vs Recall
plt.scatter(recall, r20Precision, label='r20Precision', color='red')
plt.plot(recall, r20Precision, linestyle='-', color='red', alpha=0.5)

plt.xlabel('Recall')
plt.ylabel('Precision / r20Precision')
plt.title('Precision and r20Precision vs Recall')
plt.grid(True)
plt.legend()
plt.show()

# Plot TPR vs FPR
plt.figure(figsize=(8, 6))
plt.scatter(FPR, TPR, color='blue')
plt.plot(FPR, TPR, linestyle='-', color='blue', alpha=0.5)

plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('TPR vs FPR')
plt.grid(True)
plt.show()

print("Cut off index:", cutoff_index)
