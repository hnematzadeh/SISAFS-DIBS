import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score 
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

df = pd.read_excel('D:\\DataS Projects\\DataS\\dataset2\\Colon.xlsx', header=None)
le = LabelEncoder()
y_encoded = le.fit_transform(df.iloc[:, -1])  # returns numpy array of ints

# Assign as pandas Series with integer dtype
y = pd.Series(y_encoded, dtype=int)
# Separate features and labels
X = df.iloc[:, :-1]
# Create class-specific subsets
labels = np.unique(y)
num_features = X.shape[1]
##############################   
###################### CNS; 
df = pd.read_excel ('D:\DataS Projects\DataS\dataset2\CNS.xlsx', header=None)
le = LabelEncoder()
df.iloc[:, -1] = le.fit_transform(df.iloc[:, -1])
# Separate features and labels
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
# Create class-specific subsets
labels = np.unique(y)
num_features = X.shape[1]
###############################
###############################
########################## GLI
df = pd.read_csv ('D:\DataS Projects\DataS\dataset2\GLI.csv', header=None)
le = LabelEncoder()
df.iloc[:, -1] = le.fit_transform(df.iloc[:, -1])
# Separate features and labels
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
# Create class-specific subsets
labels = np.unique(y)
# X_subsets1 = {label: X[y == label].reset_index(drop=True) for label in labels}
# Initialize hc1 and hc2: each feature m maps to a list of values
num_features = X.shape[1]

###############  SMK
df = pd.read_csv ('D:\DataS Projects\DataS\dataset2\SMK.csv', header=None)
le = LabelEncoder()
df.iloc[:, -1] = le.fit_transform(df.iloc[:, -1])
# Separate features and labels
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
# Create class-specific subsets
labels = np.unique(y)
# X_subsets1 = {label: X[y == label].reset_index(drop=True) for label in labels}
# Initialize hc1 and hc2: each feature m maps to a list of values
num_features = X.shape[1]
###################################################
################# Covid-3c
df = pd.read_csv ('D:\DataS Projects\DataS\dataset2\Covid.csv', header=None)
le = LabelEncoder()
df.iloc[:, -1] = le.fit_transform(df.iloc[:, -1])
df.iloc[:, -1] = df.iloc[:, -1].astype(int)   # force dtype change# Separate features and labels
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
y = y.astype(int)
# Create class-specific subsets
labels = np.unique(y)
# X_subsets1 = {label: X[y == label].reset_index(drop=True) for label in labels}
# Initialize hc1 and hc2: each feature m maps to a list of values
num_features = X.shape[1]
# X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)


####################Leukemia-3c
df = pd.read_excel ('D:\DataS Projects\DataS\dataset2\Leukemia_3c.xlsx', header=None)
le = LabelEncoder()
df.iloc[:, -1] = le.fit_transform(df.iloc[:, -1])
# Separate features and labels
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
# Create class-specific subsets
labels = np.unique(y)
# X_subsets1 = {label: X[y == label].reset_index(drop=True) for label in labels}
# Initialize hc1 and hc2: each feature m maps to a list of values
num_features = X.shape[1]
#######################MLL-3c
df = pd.read_excel ('D:\DataS Projects\DataS\dataset2\MLL.xlsx', header=None)
le = LabelEncoder()
df.iloc[:, -1] = le.fit_transform(df.iloc[:, -1])
# Separate features and labels
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
# Create class-specific subsets
labels = np.unique(y)
# X_subsets1 = {label: X[y == label].reset_index(drop=True) for label in labels}
# Initialize hc1 and hc2: each feature m maps to a list of values
num_features = X.shape[1]

#######################SRBCT-4c
df = pd.read_excel ('D:\DataS Projects\DataS\dataset2\SRBCT.xlsx', header=None)
le = LabelEncoder()
df.iloc[:, -1] = le.fit_transform(df.iloc[:, -1])
# Separate features and labels
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
# Create class-specific subsets
labels = np.unique(y)
# X_subsets1 = {label: X[y == label].reset_index(drop=True) for label in labels}
# Initialize hc1 and hc2: each feature m maps to a list of values
num_features = X.shape[1]

  
###################  Original Metrics ##############################
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
clf = DecisionTreeClassifier(random_state=42)

cv_accuracies = []
cv_precisions = []
cv_recalls = []
cv_f1s = []

for train_idx, test_idx in kfold.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    
    cv_accuracies.append(acc)
    cv_precisions.append(prec)
    cv_recalls.append(rec)
    cv_f1s.append(f1)

# Print rounded mean scores
print("Mean Accuracy:", round(sum(cv_accuracies)/len(cv_accuracies), 2))
print("Mean Precision:", round(sum(cv_precisions)/len(cv_precisions), 2))
print("Mean Recall:", round(sum(cv_recalls)/len(cv_recalls), 2))
print("Mean F1 Score:", round(sum(cv_f1s)/len(cv_f1s), 2))


# ========= Helper function =========


def per_feature_scores(X_scaled, y, test_idx, epsilon=1e-8):
    """Compute per-feature S_f values for a single test sample (LOO)."""
    x = X_scaled.iloc[test_idx].values
    label = y.iloc[test_idx]
    
    # Exclude test sample
    X_train = X_scaled.drop(index=test_idx)
    y_train = y.drop(index=test_idx)
    
    # Split into true class and others
    X_true = X_train[y_train == label]
    X_others = X_train[y_train != label]
    
    # Per-feature intra-class and inter-class mean absolute differences
    I_f = np.mean(np.abs(X_true.values - x), axis=0)
    J_f = np.mean(np.abs(X_others.values - x), axis=0)
    
    S_f = J_f / (I_f + epsilon)
    return S_f



########### gamma tunning


# --- Make sure X_scaled and y are aligned ---
X_scaled = X.reset_index(drop=True)
y = y.reset_index(drop=True)

# --- Candidate gamma values ---
gamma_values = np.arange(0.1, 1.1, 0.1)
results = {}

for gamma in gamma_values:
    correct_preds = 0
    
    for i in range(len(X_scaled)):
        x = X_scaled.iloc[i].values
        label = y.iloc[i]

        X_train = X_scaled.drop(index=i).reset_index(drop=True)
        y_train = y.drop(index=i).reset_index(drop=True)
        classes = np.unique(y_train)

        aucs = []
        for c in classes:
            class_samples = X_train[y_train == c]
            n_c = len(class_samples)
            dists = np.linalg.norm(class_samples.values - x, axis=1)
            auc_c = np.trapz(dists, np.arange(1, len(dists)+1))

            mean_vec = class_samples.values.mean(axis=0)
            intra_dists = np.linalg.norm(class_samples.values - mean_vec, axis=1)
            mean_intra = np.mean(intra_dists)

            # Apply gamma
            auc_c_norm = (auc_c / (n_c + 1e-8)) / (mean_intra ** gamma + 1e-8)
            aucs.append(auc_c_norm)
        
        predicted_label = classes[np.argmin(aucs)]
        if predicted_label == label:
            correct_preds += 1

    # --- Compute overall accuracy ---
    overall_acc = round (correct_preds / len(X_scaled),2)
    results[gamma] = overall_acc

# --- Find best gamma ---
best_gamma = max(results, key=results.get)
print(f"\nBest gamma by overall accuracy: {best_gamma:.2f}")
print(f"Overall accuracy: {results[best_gamma]:.3f}")

# --- Show all results sorted by accuracy ---
df_results = pd.DataFrame([
    {"gamma": g, "overall_acc": results[g]} 
    for g in results
])
print("\nGamma search results (sorted):")
print(df_results.sort_values("overall_acc", ascending=False))



# ========= Main LOO loop =========
class_correct = {c: 0 for c in np.unique(y)}
class_total = {c: 0 for c in np.unique(y)}
n_samples, n_features = X.shape
top_fraction = 0.1  # top 10% of features per sample
epsilon = 1e-8
gamma = 0.8
correct_preds = 0

# Keep track of which sample belongs to which class
informative_features_all = []
informative_sample_classes = []

for i in range(n_samples):
    x = X.iloc[i].values
    label = y.iloc[i]
    class_total[label] += 1  # count this sample in its true class
    
    # Exclude current sample
    X_train = X.drop(index=i)
    y_train = y.drop(index=i)
    classes = np.unique(y_train)
    
    aucs = []
    for c in classes:
        class_samples = X_train[y_train == c]
        n_c = len(class_samples)
        dists = np.linalg.norm(class_samples.values - x, axis=1)
        auc_c = np.trapz(dists, np.arange(1, len(dists) + 1))
        
        # normalize by intra-class dispersion
        mean_vec = class_samples.values.mean(axis=0)
        intra_dists = np.linalg.norm(class_samples.values - mean_vec, axis=1)
        mean_intra = np.mean(intra_dists)

        # normalize by both class size and dispersion
        auc_c_norm = (auc_c / (n_c + 1e-8)) / (mean_intra ** gamma)
        aucs.append(auc_c_norm)
    
    predicted_label = classes[np.argmin(aucs)]
    
    # Track per-class correctness
    if predicted_label == label:
        correct_preds += 1
        class_correct[label] += 1
        
        # Compute feature importance for this correctly predicted sample
        S_f = per_feature_scores(X, y, test_idx=i, epsilon=epsilon)
        k = max(1, int(np.ceil(top_fraction * n_features)))
        top_features = np.argsort(S_f)[-k:]
        
        informative_features_all.append(set(top_features))
        informative_sample_classes.append(label)

# ========= Aggregate results =========
accuracy = round(correct_preds / n_samples, 2)
print(f"\nOverall LOO accuracy (AUC-based): {accuracy:.3f}")

# --- Per-class accuracy breakdown ---
class_acc = {}
print("\nPer-class correct prediction breakdown:")
for c in np.unique(y):
    total = class_total[c]
    correct = class_correct[c]
    acc = correct / total if total > 0 else 0
    class_acc[c] = acc
    print(f"  Class {c}: {correct}/{total} correct ({acc*100:.1f}%)")





# ----------------------------
# Prepare per-class accuracies and weights
# ----------------------------
acc_sum = sum(class_acc.values()) if sum(class_acc.values()) > 0 else 1.0
class_weights = {c: class_acc[c] / acc_sum for c in class_acc}

# ----------------------------
# Candidate total_limit values to test
# ----------------------------
limit_candidates = range(10, 100, 10)  # e.g. test 50, 75, 100, ..., 300
cv_results = []

# Store best performing combination
best_limit = None
best_acc = -np.inf

for total_limit in limit_candidates:
    # Adaptive per-class limits
    Limit_per_class = {c: max(1, round(total_limit * class_weights[c])) for c in class_acc}
    
    # --- Select top features per class ---
    informative_top_features_per_class = {}
    if informative_features_all:
        informative_features_per_class = {c: [] for c in np.unique(y)}
        for i, top_features in enumerate(informative_features_all):
            sample_label = informative_sample_classes[i]
            informative_features_per_class[sample_label].extend(top_features)
        
        for c, features_list in informative_features_per_class.items():
            if features_list:
                freq = pd.Series(features_list).value_counts().sort_values(ascending=False)
                Limit = Limit_per_class[c]
                top_features = freq.head(Limit).index.tolist()
                informative_top_features_per_class[c] = top_features
            else:
                informative_top_features_per_class[c] = []
    
    # Combine features across all classes
    combined_features = sorted(set([f for features in informative_top_features_per_class.values() for f in features]))
    X_selected = X.iloc[:, combined_features]
    
    # ----------------------------
    # Stratified K-Fold Cross-Validation
    # ----------------------------
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf = DecisionTreeClassifier(random_state=42)
    cv_accuracies = []

    for train_idx, test_idx in kfold.split(X_selected, y):
        X_train, X_test = X_selected.iloc[train_idx], X_selected.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        cv_accuracies.append(accuracy_score(y_test, y_pred))
    
    mean_acc = np.mean(cv_accuracies)
    cv_results.append((total_limit, mean_acc))
    
    print(f"Tested total_limit={total_limit} → mean CV acc={mean_acc:.3f}")
    
    # if mean_acc > best_acc:
    #     best_acc = mean_acc
    #     best_limit = total_limit
    if (mean_acc > best_acc) or (np.isclose(mean_acc, best_acc) and total_limit > best_limit):
    # if (mean_acc > best_acc):

        best_acc = mean_acc
        best_limit = total_limit

# ----------------------------
# Final Results
# ----------------------------
print("\n==== Cross-Validation Summary ====")
for total_limit, acc in cv_results:
    print(f"  total_limit={total_limit}: mean_acc={acc:.3f}")

print(f"\nOptimal total_limit: {best_limit} (mean CV acc={best_acc:.8f})")

# Recompute final feature subset using best total_limit
Limit_per_class = {c: max(1, round(best_limit * class_weights[c])) for c in class_acc}
# Adjust total to ensure it equals best_limit
diff = best_limit - sum(Limit_per_class.values())

if diff != 0:
    # Sort classes by fractional part of ideal (unrounded) allocation
    ideal_alloc = {c: best_limit * class_weights[c] for c in class_acc}
    frac_part = {c: ideal_alloc[c] - int(ideal_alloc[c]) for c in class_acc}
    
    # If we need to add 1s, add to classes with largest fractional parts
    # If we need to subtract, remove from smallest fractional parts
    sorted_classes = sorted(frac_part, key=frac_part.get, reverse=(diff > 0))
    
    for c in sorted_classes:
        if diff == 0:
            break
        Limit_per_class[c] += 1 if diff > 0 else -1
        diff += -1 if diff > 0 else 1


informative_top_features_per_class = {}

if informative_features_all:
    informative_features_per_class = {c: [] for c in np.unique(y)}
    for i, top_features in enumerate(informative_features_all):
        sample_label = informative_sample_classes[i]
        informative_features_per_class[sample_label].extend(top_features)
    
    for c, features_list in informative_features_per_class.items():
        if features_list:
            freq = pd.Series(features_list).value_counts().sort_values(ascending=False)
            Limit = Limit_per_class[c]
            top_features = freq.head(Limit).index.tolist()
            informative_top_features_per_class[c] = top_features
            print(f"\nTop {Limit} informative features for Class {c}: {top_features}")
        else:
            informative_top_features_per_class[c] = []

# Combine final selected features
combined_features = sorted(set([f for features in informative_top_features_per_class.values() for f in features]))
print(f"\n✅ Optimal combined feature subset size: {len(combined_features)}")







# Create new reduced dataset
X_selected = X.iloc[:, combined_features]
print(f"New dataset shape: {X_selected.shape}")

# ----------------------------
# Stratified K-Fold Cross-Validation
# ----------------------------
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
clf = DecisionTreeClassifier(random_state=42)

cv_accuracies = []

for train_idx, test_idx in kfold.split(X_selected, y):
    X_train, X_test = X_selected.iloc[train_idx], X_selected.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cv_accuracies.append(acc)

# ----------------------------
# Results
# ----------------------------
mean_acc = round (np.mean(cv_accuracies),2)
std_acc = np.std(cv_accuracies)

print(f"\nDecision Tree Stratified 5-Fold Accuracy: {mean_acc:.3f} ± {std_acc:.3f}")




###### accuracy of original dataset

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
clf = DecisionTreeClassifier(random_state=42)

cv_accuracies = []

for train_idx, test_idx in kfold.split(X_selected, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cv_accuracies.append(acc)

# ----------------------------
# Results
# ----------------------------
mean_acc = round(np.mean(cv_accuracies),2)
std_acc = np.std(cv_accuracies)

print(f"\nDecision Tree Stratified 5-Fold Accuracy: {mean_acc:.3f} ± {std_acc:.3f}")











X_reduced = X.iloc[:, combined_features].reset_index(drop=True)




MAX_SUBSET_SIZE = 10 

# # -------------------------------------------------------------------
# # --- Utility Functions ---
# # -------------------------------------------------------------------
import random
def generate_neighbors(solution, all_features, num_neighbors=3, min_size=1, max_size=10):
    """Generate neighbors by randomly adding/removing features."""
    neighbors = []
    for _ in range(num_neighbors):
        neighbor = solution.copy()
        available = [f for f in all_features if f not in neighbor]
        if random.random() < 0.5 and len(neighbor) > min_size:  
            num_remove = random.randint(1, min(len(neighbor) - min_size, 3))
            to_remove = random.sample(neighbor, num_remove)
            for f in to_remove:
                neighbor.remove(f)
        elif len(neighbor) < max_size:  
            if available:
                num_add = random.randint(1, min(len(available), max_size - len(neighbor), 3))
                to_add = random.sample(available, num_add)
                neighbor.extend(to_add)
        neighbors.append(sorted(neighbor))
    return neighbors
def evaluate_subset(X, y, subset):
    if not subset:
        return 0.0
    valid_subset = [f for f in subset if f in X.columns]
    if not valid_subset:
        return 0.0
    X_sub = X.loc[:, valid_subset]
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf = DecisionTreeClassifier(random_state=42)
    accs = []
    try:
        for train_idx, test_idx in kfold.split(X_sub, y):
            clf.fit(X_sub.iloc[train_idx], y.iloc[train_idx])
            y_pred = clf.predict(X_sub.iloc[test_idx])
            accs.append(accuracy_score(y.iloc[test_idx], y_pred))
    except ValueError:
        return 0.0 
    return np.mean(accs)

def jaccard_distance(a, b):
    a, b = set(a), set(b)
    if len(a | b) == 0:
        return 0
    return 1 - len(a & b) / len(a | b)

def jaccard_similarity(a, b):
    return 1 - jaccard_distance(a, b)

def average_jaccard_similarity(solutions):
    if len(solutions) <= 1:
        return 0.0
    total, count = 0.0, 0
    for i in range(len(solutions)):
        for j in range(i + 1, len(solutions)):
            total += jaccard_similarity(solutions[i], solutions[j])
            count += 1
    return total / count if count > 0 else 0.0

def feature_utility(beam_with_scores, all_features, best_solution, alpha=0.5):
    if not best_solution:
        return {f: 1.0 for f in all_features}
    feature_scores = {f: [] for f in all_features}
    for score, solution in beam_with_scores:
        div_score = jaccard_distance(solution, best_solution)
        sol_util = alpha * score + (1 - alpha) * div_score
        for f in solution:
            feature_scores[f].append(sol_util)
    avg_util = {f: np.mean(vals) if vals else 0.0 for f, vals in feature_scores.items()}
    max_util = max(avg_util.values()) if avg_util else 1.0
    if max_util > 0:
        avg_util = {k: v / max_util for k, v in avg_util.items()}
    return avg_util
# -------------------------------------------------------------------
# --- GLOBAL BEAM SEARCH (NO MUTATION) WITH BEAM-WIDE DIVERSITY ---
# -------------------------------------------------------------------
MAX_SUBSET_SIZE= 10
def global_beam_search_continuous_hgs(X, y, beam_width=10, max_iters=60, max_subset_size=MAX_SUBSET_SIZE):
    """
    Hybrid Guided Beam Search (HGS) for feature selection.
    Combines local neighborhood search + H-Jump exploration with adaptive stagnation handling.
    """
    all_features = list(X.columns)
    num_features = len(all_features)
    
    H_JUMP_SIZE = 30
    TOP_K_FEATURES = 10
    
    
    nfe_count = 0
    performance_history = []  # (iteration, best_score, avg_similarity)
    best_solution = None
    best_score = 0.0
    no_improve_count = 0
    similarity_stagnation_count = 0
    last_similarity = None

    # --- Initialization ---
    initial_candidates = [
        sorted(random.sample(all_features, random.randint(1, min(num_features, max_subset_size))))
        for _ in range(beam_width)
    ]

    scored_initial = []
    for subset in initial_candidates:
        score = evaluate_subset(X, y, subset)
        nfe_count += 1
        scored_initial.append((score, subset))
        if score > best_score:
            best_score = score
            best_solution = subset

    scored_initial.sort(key=lambda x: x[0], reverse=True)
    beam_with_scores = scored_initial[:beam_width]
    beam = [s for score, s in beam_with_scores]

    avg_sim = average_jaccard_similarity(beam)
    performance_history.append((0, best_score, avg_sim))
    print(f"Initialization (Iter 0) NFE: {nfe_count}, Best: {best_score:.4f}, AvgSim: {avg_sim:.3f}")

    # --- Main Iterations ---
    for it in range(1, max_iters + 1):
        # 1️⃣ Generate local neighbors
        new_candidates = []
        for solution in beam:
            new_candidates.extend(generate_neighbors(solution, all_features, num_neighbors=5, max_size=max_subset_size))

        # 2️⃣ H-Jump: mix high-utility + non-high-utility features
        h_jump_candidates = []
        utilities = feature_utility(beam_with_scores, all_features, best_solution, alpha=0.5)
        sorted_features = sorted(utilities.items(), key=lambda x: x[1], reverse=True)

        high_utility_features = [f for f, _ in sorted_features[:TOP_K_FEATURES]]
        non_high_utility_features = [f for f, _ in sorted_features[TOP_K_FEATURES:]]

        for _ in range(H_JUMP_SIZE):
            k = random.randint(1, max_subset_size)
            mix_ratio = 0.5
            num_high = max(1, int(k * mix_ratio))
            num_low = k - num_high

            high_part = random.sample(high_utility_features, min(num_high, len(high_utility_features))) if high_utility_features else []
            low_part = random.sample(non_high_utility_features, min(num_low, len(non_high_utility_features))) if non_high_utility_features else []

            subset = sorted(list(set(high_part + low_part)))
            if subset:
                h_jump_candidates.append(subset)

        # 3️⃣ Evaluate new candidates (not old beam)
        scored_candidates = []
        improvement = False

        for subset in (new_candidates + h_jump_candidates):
            score = evaluate_subset(X, y, subset)
            nfe_count += 1
            scored_candidates.append((score, subset))
            if score > best_score:
                best_score = score
                best_solution = subset
                improvement = True

        # 4️⃣ Combine old beam + new scored ones
        combined_candidates = beam_with_scores + scored_candidates

        # 5️⃣ Select top beam
        combined_candidates.sort(key=lambda x: x[0], reverse=True)
        beam_with_scores = combined_candidates[:beam_width]
        beam = [s for score, s in beam_with_scores]

        # 6️⃣ Diversity monitoring
        avg_sim = average_jaccard_similarity(beam)
        if last_similarity is not None and abs(avg_sim - last_similarity) < 1e-3:
            similarity_stagnation_count += 1
        else:
            similarity_stagnation_count = 0
        last_similarity = avg_sim

        # 7️⃣ No-improvement logic
        no_improve_count = 0 if improvement else no_improve_count + 1

        # --- Optional logging ---
        if it % 5 == 0:
            print(f"H-JUMP active at iteration {it} — exploring {H_JUMP_SIZE} new mixes.")
        print(f"Iter {it}/{max_iters}, NFE: {nfe_count}, Best: {best_score:.4f}, "
              f"NoImprove: {no_improve_count}, AvgSim: {avg_sim:.3f}")

        performance_history.append((it, best_score, avg_sim))

        # 8️⃣ Early stopping
        if no_improve_count >= 29 or abs(best_score) >= 0.999:
            print("⛔ Stopping early due to stagnation or near-perfect accuracy.")
            break

    return best_solution, best_score, nfe_count, performance_history, beam




print("Starting Global Beam Search with Continuous Heuristic Jumps + Diversity Control...")
best_features, best_acc, nfe_count, performance_history, pop = global_beam_search_continuous_hgs(
    X_reduced, y, beam_width=10, max_iters=100)

SSize=len(best_features)
print("\n--- Final Results (Continuous HGS) ---")
print(f"Best Accuracy Found:{best_acc:.2f}")
print(f"Best Feature Subset: {best_features}")
print(f"Best Feature Subset Size:{SSize} ")
print(f"Total NFE: {nfe_count}") 


###################   Metrics after feature selection ##############################
X1=X[best_features]
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
clf = DecisionTreeClassifier(random_state=42)

cv_accuracies = []
cv_precisions = []
cv_recalls = []
cv_f1s = []

for train_idx, test_idx in kfold.split(X1, y):
    X_train, X_test = X1.iloc[train_idx], X1.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    
    cv_accuracies.append(acc)
    cv_precisions.append(prec)
    cv_recalls.append(rec)
    cv_f1s.append(f1)

# Print rounded mean scores
print("Mean Accuracy:", round(sum(cv_accuracies)/len(cv_accuracies), 2))
print("Mean Precision:", round(sum(cv_precisions)/len(cv_precisions), 2))
print("Mean Recall:", round(sum(cv_recalls)/len(cv_recalls), 2))
print("Mean F1 Score:", round(sum(cv_f1s)/len(cv_f1s), 2))
#######################################################################

import pickle
import matplotlib.pyplot as plt
# Similarity_threshold=0.8
accuracy_without_fs = 0.6  # Baseline accuracy
accuracy_SIFS = 0.6    # Accuracy with ISCFS

if performance_history:
    # Extract iteration, best_score, and avg_similarity
    iteration_values = [h[0] for h in performance_history]
    best_scores = [h[1] for h in performance_history]
    avg_sims = [h[2] for h in performance_history]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # --- Plot Accuracy ---
    ax1.plot(iteration_values, best_scores, marker='o', linestyle='-', markersize=8, color='blue', label='Accuracy with SISAFS-DIBS')
    # ax1.axhline(y=accuracy_without_fs, color='orange', linestyle='--', linewidth=3, label=f'Accuracy without FS ({accuracy_without_fs})')
    # ax1.axhline(y=accuracy_SIFS, color='green', linestyle='--', linewidth=3, label=f'Accuracy with ISCFS ({accuracy_SIFS})')
    ax1.set_xlabel('Iteration', fontsize=30)
    ax1.set_ylabel('Accuracy', fontsize=30, color='blue')
    ax1.tick_params(axis='y', labelcolor='blue', labelsize=26)
    ax1.tick_params(axis='x', labelsize=26)
    ax1.set_ylim(0.5, 1.01)

    # --- Optional: Plot AvgSim on secondary y-axis ---
    ax2 = ax1.twinx()
    ax2.plot(iteration_values, avg_sims, marker='x', linestyle='--', color='red', label='Average feature similarity in the beam', alpha=0.7)
    ax2.set_ylabel('AvgSim', fontsize=30, color='red')
    ax2.tick_params(axis='y', labelcolor='red', labelsize=26)
    ax2.set_ylim(0, 1.0)
    # ax2.axhline(y=Similarity_threshold, color='orange', linestyle='-', linewidth=3, label=f'Similarity threshold ({Similarity_threshold})')

    # --- Combine legends ---
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, fontsize=18, loc='lower right')
    
    # plt.yticks(fontsize=26)
    # plt.xticks(fontsize=26)

    
    ax1.grid(True)
    plt.title('CNS', fontsize=36)
    plt.tight_layout()
    plt.show()









accuracy_without_fs = 0.6 # Define the constant for clarity
accuracy_SIFS = 0.6

if performance_history:
    # Extract the iteration number (x-axis) and best score (y-axis)
    iteration_values = [h[0] for h in performance_history]
    best_scores = [h[1] for h in performance_history]

    plt.figure(figsize=(10, 6))
    
    # Plot the performance history
    plt.plot(iteration_values, best_scores, marker='o', linestyle='-', markersize=14,  color='blue', label='Accuracy with SISAFS-DIBS')
    # plt.plot(iteration_values, best_scoresGBS, marker='o', linestyle='-', markersize=22,  color='blue', label='Accuracy with GBS')

    # ----------------------------------------------------------------------
    # Add the horizontal red line at 0.74
    plt.axhline(y=accuracy_without_fs, color='orange', linestyle='-', linewidth=12, label=f'Accuracy without FS ({accuracy_without_fs})')
    plt.axhline(y=accuracy_SIFS, color='green', linestyle='-', linewidth=4, label=f'Accuracy with SISAFS ({accuracy_SIFS})')

    # ----------------------------------------------------------------------
    
    plt.title('CNS', fontsize=36) # Use 'fontsize' instead of 'font_size' for matplotlib
    plt.xlabel('Iteration', fontsize=30)
    plt.ylabel('Accuracy', fontsize=30)
    plt.ylim(0.5, 1.01)
    
    # Use plt.gca().tick_params for tick styling, as font_size is deprecated in plt.xticks
    # plt.xticks(np.unique(iteration_values), labelsize=14) 
    # plt.xticks(ticks=np.unique(iteration_values), fontsize=14)
    plt.yticks(fontsize=22)
    plt.xticks(fontsize=22)
    # plt.gca().tick_params(axis='both', which='major', labelsize=14) # Adjust tick size
    
    plt.grid(True)
    plt.legend(fontsize=16) # Add a legend to show the label for the horizontal line
    plt.tight_layout() # Adjust plot to prevent labels from being cut off
    plt.show()

file_path = r'D:\Papers\SIFS\Plots\ISCFS-DIBS\CNS\CNS\performance_historyISCFSDIBS.pkl'
### For Saving
with open(file_path, 'wb') as f:
    pickle.dump(performance_history, f)   
### For Loading    
with open(file_path, 'rb') as f:
    performance_history = pickle.load(f)







######################################################
######################################################
##### MUTUAL INFORMATION##############
from sklearn.feature_selection import mutual_info_classif

def mutual_information_scores(X, y):
    """
    Calculate Mutual Information scores for each feature in X with respect to target y.
    
    Args:
        X (pd.DataFrame): The feature matrix.
        y (pd.Series or np.array): The target vector.
        
    Returns:
        np.array: An array of Mutual Information scores.
    """
    # mutual_info_classif computes the score for each feature
    scores = mutual_info_classif(X, y, random_state=42)
    return scores

# Assuming X is a pandas DataFrame and y is a Series or array
# X, y = ...  # Your data should be loaded here

# Calculate Mutual Information scores for each feature
scores = mutual_information_scores(X, y)

# Get the indices of the top 50 features based on their scores
# np.argsort returns the indices that would sort an array
MI_indices = np.argsort(scores)[-30:][::-1]

# Get the names of the top 50 features
ranked_features = X.columns[MI_indices]

print("Top 30 feature indices:", MI_indices)
print("Top 30 feature names:", ranked_features.tolist())




# ----------------------------
# Stratified K-Fold Cross-Validation
# ----------------------------
X_selected = X[ranked_features]
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
clf = DecisionTreeClassifier(random_state=42)

cv_accuracies = []

for train_idx, test_idx in kfold.split(X_selected, y):
    X_train, X_test = X_selected.iloc[train_idx], X_selected.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cv_accuracies.append(acc)

# ----------------------------
# Results
# ----------------------------
mean_acc = round(np.mean(cv_accuracies), 2)
std_acc = np.std(cv_accuracies)

print(f"\nDecision Tree Stratified 5-Fold Accuracy (Top 30 MI Features): {mean_acc:.3f} ± {std_acc:.3f}")

####################################################
def correlation_feature_importance(X, y):
    """
    Calculate feature importances using absolute Pearson correlation with the target.
    
    Args:
        X (pd.DataFrame): The feature matrix.
        y (pd.Series or np.array): The target vector.
        
    Returns:
        np.array: An array of feature importance scores (absolute correlations).
    """
    # Ensure y is a Pandas Series for compatibility
    y = pd.Series(y)
    
    # Compute correlation of each feature with the target
    correlations = X.apply(lambda col: col.corr(y))
    
    # Take absolute value so both positive and negative correlations are treated equally
    importance_scores = correlations.abs().values
    
    return importance_scores

# Assuming X is a pandas DataFrame and y is a Series or array
# X, y = ...  # Your data should be loaded here

# Calculate correlation-based feature importances
scores = correlation_feature_importance(X, y)

# Get the indices of the top 50 features based on correlation strength
corr_indices = np.argsort(scores)[-30:][::-1]

# Get the names of the top 50 features
ranked_features = X.columns[corr_indices]

print("Top 30 feature indices:", corr_indices)
print("Top 30 feature names:", ranked_features.tolist())


# ----------------------------
# Stratified K-Fold Cross-Validation
# ----------------------------
X_selected = X[ranked_features]
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
clf = DecisionTreeClassifier(random_state=42)

cv_accuracies = []

for train_idx, test_idx in kfold.split(X_selected, y):
    X_train, X_test = X_selected.iloc[train_idx], X_selected.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cv_accuracies.append(acc)

# ----------------------------
# Results
# ----------------------------
mean_acc = round(np.mean(cv_accuracies), 2)
std_acc = np.std(cv_accuracies)

print(f"\nDecision Tree Stratified 5-Fold Accuracy (Top 30 MI Features): {mean_acc:.3f} ± {std_acc:.3f}")

#######################################################
#######################################################


from sklearn.feature_selection import chi2
from sklearn.preprocessing import KBinsDiscretizer

def chi2_feature_importance(X, y, n_bins=10, strategy="quantile"):
    """
    Calculate feature importances using Chi-Square test with discretization.
    
    Args:
        X (pd.DataFrame): The feature matrix (numeric features).
        y (pd.Series or np.array): The categorical target vector.
        n_bins (int): Number of bins to discretize numeric features.
        strategy (str): Binning strategy - "uniform", "quantile", or "kmeans".
        
    Returns:
        np.array: An array of feature importance scores (Chi2 statistics).
    """
    # Discretize numerical features into bins
    kb = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy=strategy)
    X_binned = kb.fit_transform(X)
    
    # Perform chi-square test
    chi2_stats, p_values = chi2(X_binned, y)
    
    return chi2_stats

# Assuming X is a pandas DataFrame and y is a Series or array
# X, y = ...  # Your data should be loaded here

# Calculate Chi-Square feature importances
scores = chi2_feature_importance(X, y, n_bins=10, strategy="quantile")

# Get the indices of the top 50 features
chi2_indices = np.argsort(scores)[-30:][::-1]

# Get the names of the top 50 features
ranked_features = X.columns[chi2_indices]

print("Top 30 feature indices:", chi2_indices)
print("Top 30 feature names:", ranked_features.tolist())

# ----------------------------
# Stratified K-Fold Cross-Validation
# ----------------------------
X_selected = X[ranked_features]
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
clf = DecisionTreeClassifier(random_state=42)

cv_accuracies = []

for train_idx, test_idx in kfold.split(X_selected, y):
    X_train, X_test = X_selected.iloc[train_idx], X_selected.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cv_accuracies.append(acc)

# ----------------------------
# Results
# ----------------------------
mean_acc = round(np.mean(cv_accuracies), 2)
std_acc = np.std(cv_accuracies)

print(f"\nDecision Tree Stratified 5-Fold Accuracy (Top 30 MI Features): {mean_acc:.3f} ± {std_acc:.3f}")

##########################################################
##########################################################
from skrebate import ReliefF


def reliefF_scores(X, y):
    """
    Calculate ReliefF scores for feature selection (classification).
    
    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series or np.array): Target vector (categorical).
        n_neighbors (int): Number of neighbors to consider (default=10).
        
    Returns:
        np.array: ReliefF feature importance scores.
    """
    # Ensure features are float (ReliefF requirement)
    X_float = X.astype(float)

    relief = ReliefF(n_features_to_select=X.shape[1])
    relief.fit(X_float.values, y)
    return relief.feature_importances_

# Example usage
scores = reliefF_scores(X, y)
relief_indices = np.argsort(scores)[-30:][::-1]
ranked_features = X.columns[relief_indices]

print("Top 30 ReliefF feature indices:", relief_indices)
print("Top 30 ReliefF feature names:", ranked_features.tolist())


# ----------------------------
# Stratified K-Fold Cross-Validation
# ----------------------------
X_selected = X[ranked_features]
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
clf = DecisionTreeClassifier(random_state=42)

cv_accuracies = []

for train_idx, test_idx in kfold.split(X_selected, y):
    X_train, X_test = X_selected.iloc[train_idx], X_selected.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cv_accuracies.append(acc)

# ----------------------------
# Results
# ----------------------------
mean_acc = round(np.mean(cv_accuracies), 2)
std_acc = np.std(cv_accuracies)

print(f"\nDecision Tree Stratified 5-Fold Accuracy (Top 30 MI Features): {mean_acc:.3f} ± {std_acc:.3f}")




#########################################################
#########################################################
##########
def fisher_score(X, y):
    scores = []
    classes = np.unique(y)
    overall_mean = X.mean(axis=0)
    
    for col in X.columns:
        numerator = 0
        denominator = 0
        for c in classes:
            x_c = X[y == c][col]
            mean_c = x_c.mean()
            var_c = x_c.var()
            n_c = len(x_c)
            numerator += n_c * (mean_c - overall_mean[col]) ** 2
            denominator += n_c * var_c
        # To avoid divide-by-zero
        score = numerator / (denominator + 1e-6)
        scores.append(score)
    
    return np.array(scores)

# Calculate scores and get top 50 features
scores = fisher_score(X, y)
Fisher = np.argsort(scores)[-30:][::-1]
ranked_features = X.columns[Fisher]

print("Top 20 feature indices:", Fisher)
print("Top 20 feature names:", ranked_features.tolist())


# ----------------------------
# Stratified K-Fold Cross-Validation
# ----------------------------
X_selected = X[ranked_features]
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
clf = DecisionTreeClassifier(random_state=42)

cv_accuracies = []

for train_idx, test_idx in kfold.split(X_selected, y):
    X_train, X_test = X_selected.iloc[train_idx], X_selected.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cv_accuracies.append(acc)

# ----------------------------
# Results
# ----------------------------
mean_acc = round(np.mean(cv_accuracies), 2)
std_acc = np.std(cv_accuracies)

print(f"\nDecision Tree Stratified 5-Fold Accuracy (Top 20 MI Features): {mean_acc:.3f} ± {std_acc:.3f}")


      
#######################################################
from sklearn.ensemble import RandomForestClassifier
def random_forest_feature_importance(X, y):
    """
    Calculate feature importances using a Random Forest Classifier.
    
    Args:
        X (pd.DataFrame): The feature matrix.
        y (pd.Series or np.array): The target vector.
        
    Returns:
        np.array: An array of feature importance scores.
    """
    # Initialize Random Forest with a fixed random_state for reproducibility
    rf_model = RandomForestClassifier(random_state=42)
    
    # Fit the model to the data
    rf_model.fit(X, y)
    
    # Get feature importances
    return rf_model.feature_importances_

# Assuming X is a pandas DataFrame and y is a Series or array
# X, y = ...  # Your data should be loaded here

# Calculate Random Forest feature importances
scores = random_forest_feature_importance(X, y)

# Get the indices of the top 50 features based on their importance scores
RF_indices = np.argsort(scores)[-40:][::-1]

# Get the names of the top 50 features
ranked_features = X.columns[RF_indices]

print("Top 50 feature indices:", RF_indices)
print("Top 50 feature names:", ranked_features.tolist())


# ----------------------------
# Stratified K-Fold Cross-Validation
# ----------------------------
X_selected = X[ranked_features]
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
clf = DecisionTreeClassifier(random_state=42)

cv_accuracies = []

for train_idx, test_idx in kfold.split(X_selected, y):
    X_train, X_test = X_selected.iloc[train_idx], X_selected.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cv_accuracies.append(acc)

# ----------------------------
# Results
# ----------------------------
mean_acc = round(np.mean(cv_accuracies), 2)
std_acc = np.std(cv_accuracies)

print(f"\nDecision Tree Stratified 5-Fold Accuracy (Top 20 MI Features): {mean_acc:.3f} ± {std_acc:.3f}")


#######################################################
#######################################################

import xgboost as xgb

def xgboost_feature_importance(X, y):
    """
    Calculate feature importances using an XGBoost Classifier.
    
    Args:
        X (pd.DataFrame): The feature matrix.
        y (pd.Series or np.array): The target vector.
        
    Returns:
        np.array: An array of feature importance scores.
    """
    # Initialize XGBoost Classifier with a fixed random_state for reproducibility
    # eval_metric='logloss' is added to avoid a common warning in XGBoost
    xgb_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    
    # Fit the model to the data
    xgb_model.fit(X, y)
    
    # Get feature importances
    return xgb_model.feature_importances_

# Assuming X is a pandas DataFrame and y is a Series or array
# X, y = ...  # Your data should be loaded here

# Calculate XGBoost feature importances
scores = xgboost_feature_importance(X, y)

# Get the indices of the top 50 features based on their importance scores
XGB_indices = np.argsort(scores)[-40:][::-1]

# Get the names of the top 50 features
ranked_features = X.columns[XGB_indices]

print("Top 50 feature indices:", XGB_indices)
print("Top 50 feature names:", ranked_features.tolist())


# ----------------------------
# Stratified K-Fold Cross-Validation
# ----------------------------
X_selected = X[ranked_features]
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
clf = DecisionTreeClassifier(random_state=42)

cv_accuracies = []

for train_idx, test_idx in kfold.split(X_selected, y):
    X_train, X_test = X_selected.iloc[train_idx], X_selected.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cv_accuracies.append(acc)

# ----------------------------
# Results
# ----------------------------
mean_acc = round(np.mean(cv_accuracies), 2)
std_acc = np.std(cv_accuracies)

print(f"\nDecision Tree Stratified 5-Fold Accuracy (Top 20 MI Features): {mean_acc:.3f} ± {std_acc:.3f}")


########################################################
########### Comparison with Filters ####################
########################################################
import matplotlib.pyplot as plt
import numpy as np
##########   COLON  ########################

# Data
methods = ["PCC", "Chi\u00b2", "MI", "SISAFS", "RF", "FS"]
scores = [0.76, 0.74, 0.84, 0.82, 0.71, 0.77]


# Define colors — make SISCFS red, others blue
colors = ['deepskyblue' if method != 'SISAFS' else 'red' for method in methods]

# Bar width and spacing
bar_width = 0.3
margin = 0.4  # left and right margin

# Compute x positions so bars touch each other but leave margins
x = np.arange(margin, margin + len(methods) * bar_width, bar_width)

# Create bar chart — no gaps among bars
plt.bar(x, scores, width=bar_width, color=colors, edgecolor='black', align='edge')

# Labels and title
plt.xlabel("Methods", fontsize=24)
plt.ylabel("Accuracy", fontsize=24)
plt.title("Colon", fontsize=30)

# Center method labels under bars
plt.xticks(x + bar_width / 2, methods, fontsize=16, rotation=45)
plt.yticks(fontsize=18)

# Add horizontal reference lines
y_discfs = 0.84
y_baseline = 0.74
plt.axhline(y=y_discfs, color='orange', linestyle='--', linewidth=3)
plt.axhline(y=y_baseline, color='green', linestyle='--', linewidth=3)

# Get current x-axis limit
x_max = plt.gca().get_xlim()[1]

# Add text labels slightly above the lines
offset = 0.01  # small vertical offset above the line
plt.text(x_max, y_discfs + offset, 'DISAFS', color='orange', fontsize=16, ha='left', va='bottom')
plt.text(x_max, y_baseline + offset, 'Baseline', color='green', fontsize=16, ha='left', va='bottom')

# Adjust axis limits to include margins
plt.xlim(0, margin * 2 + len(methods) * bar_width)
plt.ylim(0.6, 1)

# Clean visual style
for spine in ["top", "right"]:
    plt.gca().spines[spine].set_visible(False)

plt.grid(False)
plt.tight_layout()
plt.show()



###########  CNS #####################################


import matplotlib.pyplot as plt
import numpy as np

# Data
methods = ["PCC", "Chi\u00b2", "MI", "SISAFS", "RF", "FS"]
scores = [0.73, 0.72, 0.82, 0.60, 0.53, 0.75]

# Define colors — make SISCFS red, others blue
colors = ['deepskyblue' if method != 'SISAFS' else 'red' for method in methods]

# Bar width and spacing
bar_width = 0.3
margin = 0.4  # left and right margin

# Compute x positions so bars touch each other but leave margins
x = np.arange(margin, margin + len(methods) * bar_width, bar_width)

# Create bar chart — no gaps among bars
plt.bar(x, scores, width=bar_width, color=colors, edgecolor='black', align='edge')

# Labels and title
plt.xlabel("Methods", fontsize=24)
plt.ylabel("Accuracy", fontsize=24)
plt.title("CNS", fontsize=30)

# Center method labels under bars
plt.xticks(x + bar_width / 2, methods, fontsize=16, rotation=45)
plt.yticks(fontsize=18)

# Add horizontal reference lines
y_discfs = 0.68
y_baseline = 0.60
plt.axhline(y=y_discfs, color='orange', linestyle='--', linewidth=3)
plt.axhline(y=y_baseline, color='green', linestyle='--', linewidth=3)

# Get current x-axis limit
x_max = plt.gca().get_xlim()[1]

# Add text labels slightly above the lines
offset = 0.01  # small vertical offset above the line
plt.text(x_max, y_discfs + offset, 'DISAFS', color='orange', fontsize=16, ha='left', va='bottom')
plt.text(x_max, y_baseline + offset, 'Baseline', color='green', fontsize=16, ha='left', va='bottom')

# Adjust axis limits to include margins
plt.xlim(0, margin * 2 + len(methods) * bar_width)
plt.ylim(0.4, 1)

# Clean visual style
for spine in ["top", "right"]:
    plt.gca().spines[spine].set_visible(False)

plt.grid(False)
plt.tight_layout()
plt.show()



###########  GLI #####################################


import matplotlib.pyplot as plt
import numpy as np

# Data
methods = ["PCC", "Chi\u00b2", "MI", "SISAFS", "RF", "FS"]
scores = [0.84, 0.82, 0.76, 0.91, 0.84, 0.82]

# Define colors — make SISCFS red, others blue
colors = ['deepskyblue' if method != 'SISAFS' else 'red' for method in methods]

# Bar width and spacing
bar_width = 0.3
margin = 0.4  # left and right margin

# Compute x positions so bars touch each other but leave margins
x = np.arange(margin, margin + len(methods) * bar_width, bar_width)

# Create bar chart — no gaps among bars
plt.bar(x, scores, width=bar_width, color=colors, edgecolor='black', align='edge')

# Labels and title
plt.xlabel("Methods", fontsize=24)
plt.ylabel("Accuracy", fontsize=24)
plt.title("GLI", fontsize=30)

# Center method labels under bars
plt.xticks(x + bar_width / 2, methods, fontsize=16, rotation=45)
plt.yticks(fontsize=18)

# Add horizontal reference lines
y_discfs = 0.86
y_baseline = 0.81
plt.axhline(y=y_discfs, color='orange', linestyle='--', linewidth=3)
plt.axhline(y=y_baseline, color='green', linestyle='--', linewidth=3)

# Get current x-axis limit
x_max = plt.gca().get_xlim()[1]

# Add text labels slightly above the lines
offset = 0.01  # small vertical offset above the line
plt.text(x_max, y_discfs + offset, 'DISAFS', color='orange', fontsize=16, ha='left', va='bottom')
plt.text(x_max, y_baseline + offset, 'Baseline', color='green', fontsize=16, ha='left', va='bottom')

# Adjust axis limits to include margins
plt.xlim(0, margin * 2 + len(methods) * bar_width)
plt.ylim(0.6, 1)

# Clean visual style
for spine in ["top", "right"]:
    plt.gca().spines[spine].set_visible(False)

plt.grid(False)
plt.tight_layout()
plt.show()


###########  SMK #####################################


import matplotlib.pyplot as plt
import numpy as np

# Data
methods = ["PCC", "Chi\u00b2", "MI", "SISAFS", "RF", "FS"]
scores = [0.68, 0.68, 0.74, 0.66, 0.62, 0.68]

# Define colors — make SISCFS red, others blue
colors = ['deepskyblue' if method != 'SISAFS' else 'red' for method in methods]

# Bar width and spacing
bar_width = 0.3
margin = 0.4  # left and right margin

# Compute x positions so bars touch each other but leave margins
x = np.arange(margin, margin + len(methods) * bar_width, bar_width)

# Create bar chart — no gaps among bars
plt.bar(x, scores, width=bar_width, color=colors, edgecolor='black', align='edge')

# Labels and title
plt.xlabel("Methods", fontsize=24)
plt.ylabel("Accuracy", fontsize=24)
plt.title("SMK", fontsize=30)

# Center method labels under bars
plt.xticks(x + bar_width / 2, methods, fontsize=16, rotation=45)
plt.yticks(fontsize=18)

# Add horizontal reference lines
y_discfs = 0.69
y_baseline = 0.59
plt.axhline(y=y_discfs, color='orange', linestyle='--', linewidth=3)
plt.axhline(y=y_baseline, color='green', linestyle='--', linewidth=3)

# Get current x-axis limit
x_max = plt.gca().get_xlim()[1]

# Add text labels slightly above the lines
offset = 0.01  # small vertical offset above the line
plt.text(x_max, y_discfs + offset, 'DISAFS', color='orange', fontsize=16, ha='left', va='bottom')
plt.text(x_max, y_baseline + offset, 'Baseline', color='green', fontsize=16, ha='left', va='bottom')

# Adjust axis limits to include margins
plt.xlim(0, margin * 2 + len(methods) * bar_width)
plt.ylim(0.5, 1)

# Clean visual style
for spine in ["top", "right"]:
    plt.gca().spines[spine].set_visible(False)

plt.grid(False)
plt.tight_layout()
plt.show()


###########  Covid-19 #####################################


import matplotlib.pyplot as plt
import numpy as np

# Data
methods = ["PCC", "Chi\u00b2", "MI", "SISAFS", "RF", "FS"]
scores = [0.54, 0.73, 0.75, 0.65, 0.65, 0.65]

# Define colors — make SISCFS red, others blue
colors = ['deepskyblue' if method != 'SISAFS' else 'red' for method in methods]

# Bar width and spacing
bar_width = 0.3
margin = 0.4  # left and right margin

# Compute x positions so bars touch each other but leave margins
x = np.arange(margin, margin + len(methods) * bar_width, bar_width)

# Create bar chart — no gaps among bars
plt.bar(x, scores, width=bar_width, color=colors, edgecolor='black', align='edge')

# Labels and title
plt.xlabel("Methods", fontsize=24)
plt.ylabel("Accuracy", fontsize=24)
plt.title("Covid-19", fontsize=30)

# Center method labels under bars
plt.xticks(x + bar_width / 2, methods, fontsize=16, rotation=45)
plt.yticks(fontsize=18)

# Add horizontal reference lines
y_discfs = 0.43
y_baseline = 0.68
plt.axhline(y=y_discfs, color='orange', linestyle='--', linewidth=3)
plt.axhline(y=y_baseline, color='green', linestyle='--', linewidth=3)

# Get current x-axis limit
x_max = plt.gca().get_xlim()[1]

# Add text labels slightly above the lines
offset = 0.01  # small vertical offset above the line
plt.text(x_max, y_discfs + offset, 'DISAFS', color='orange', fontsize=16, ha='left', va='bottom')
plt.text(x_max, y_baseline + offset, 'Baseline', color='green', fontsize=16, ha='left', va='bottom')

# Adjust axis limits to include margins
plt.xlim(0, margin * 2 + len(methods) * bar_width)
plt.ylim(0.3, 1)

# Clean visual style
for spine in ["top", "right"]:
    plt.gca().spines[spine].set_visible(False)

plt.grid(False)
plt.tight_layout()
plt.show()


###########  Leukemia #####################################


import matplotlib.pyplot as plt
import numpy as np

# Data
methods = ["PCC", "Chi\u00b2", "MI", "SISAFS", "RF", "FS"]
scores = [0.87, 0.80, 0.86, 0.89, 0.82, 0.92]

# Define colors — make SISCFS red, others blue
colors = ['deepskyblue' if method != 'SISAFS' else 'red' for method in methods]

# Bar width and spacing
bar_width = 0.3
margin = 0.4  # left and right margin

# Compute x positions so bars touch each other but leave margins
x = np.arange(margin, margin + len(methods) * bar_width, bar_width)

# Create bar chart — no gaps among bars
plt.bar(x, scores, width=bar_width, color=colors, edgecolor='black', align='edge')

# Labels and title
plt.xlabel("Methods", fontsize=24)
plt.ylabel("Accuracy", fontsize=24)
plt.title("Leukemia", fontsize=30)

# Center method labels under bars
plt.xticks(x + bar_width / 2, methods, fontsize=16, rotation=45)
plt.yticks(fontsize=18)

# Add horizontal reference lines
y_discfs = 0.89
y_baseline = 0.82
plt.axhline(y=y_discfs, color='orange', linestyle='--', linewidth=3)
plt.axhline(y=y_baseline, color='green', linestyle='--', linewidth=3)

# Get current x-axis limit
x_max = plt.gca().get_xlim()[1]

# Add text labels slightly above the lines
offset = 0.01  # small vertical offset above the line
plt.text(x_max, y_discfs + offset, 'DISAFS', color='orange', fontsize=16, ha='left', va='bottom')
plt.text(x_max, y_baseline + offset, 'Baseline', color='green', fontsize=16, ha='left', va='bottom')

# Adjust axis limits to include margins
plt.xlim(0, margin * 2 + len(methods) * bar_width)
plt.ylim(0.7, 1)

# Clean visual style
for spine in ["top", "right"]:
    plt.gca().spines[spine].set_visible(False)

plt.grid(False)
plt.tight_layout()
plt.show()



###########  MLL #####################################


import matplotlib.pyplot as plt
import numpy as np

# Data
methods = ["PCC", "Chi\u00b2", "MI", "SISAFS", "RF", "FS"]
scores = [0.82, 0.83, 0.89, 0.93, 0.90, 0.93]

# Define colors — make SISCFS red, others blue
colors = ['deepskyblue' if method != 'SISAFS' else 'red' for method in methods]

# Bar width and spacing
bar_width = 0.3
margin = 0.4  # left and right margin

# Compute x positions so bars touch each other but leave margins
x = np.arange(margin, margin + len(methods) * bar_width, bar_width)

# Create bar chart — no gaps among bars
plt.bar(x, scores, width=bar_width, color=colors, edgecolor='black', align='edge')

# Labels and title
plt.xlabel("Methods", fontsize=24)
plt.ylabel("Accuracy", fontsize=24)
plt.title("MLL", fontsize=30)

# Center method labels under bars
plt.xticks(x + bar_width / 2, methods, fontsize=16, rotation=45)
plt.yticks(fontsize=18)

# Add horizontal reference lines
y_discfs = 0.96
y_baseline = 0.85
plt.axhline(y=y_discfs, color='orange', linestyle='--', linewidth=3)
plt.axhline(y=y_baseline, color='green', linestyle='--', linewidth=3)

# Get current x-axis limit
x_max = plt.gca().get_xlim()[1]

# Add text labels slightly above the lines
offset = 0.01  # small vertical offset above the line
plt.text(x_max, y_discfs + offset, 'DISAFS', color='orange', fontsize=16, ha='left', va='bottom')
plt.text(x_max, y_baseline + offset, 'Baseline', color='green', fontsize=16, ha='left', va='bottom')

# Adjust axis limits to include margins
plt.xlim(0, margin * 2 + len(methods) * bar_width)
plt.ylim(0.7, 1)

# Clean visual style
for spine in ["top", "right"]:
    plt.gca().spines[spine].set_visible(False)

plt.grid(False)
plt.tight_layout()
plt.show()


###########  SRBCT #####################################


import matplotlib.pyplot as plt
import numpy as np

# Data
methods = ["PCC", "Chi\u00b2", "MI", "SISAFS", "RF", "FS"]
scores = [0.79, 0.79, 0.84, 0.87, 0.85, 0.84]

# Define colors — make SISCFS red, others blue
colors = ['deepskyblue' if method != 'SISAFS' else 'red' for method in methods]

# Bar width and spacing
bar_width = 0.3
margin = 0.4  # left and right margin

# Compute x positions so bars touch each other but leave margins
x = np.arange(margin, margin + len(methods) * bar_width, bar_width)

# Create bar chart — no gaps among bars
plt.bar(x, scores, width=bar_width, color=colors, edgecolor='black', align='edge')



# Labels and title
plt.xlabel("Methods", fontsize=24)
plt.ylabel("Accuracy", fontsize=24)
plt.title("SRBCT", fontsize=30)

# Center method labels under bars
plt.xticks(x + bar_width / 2, methods, fontsize=16, rotation=45)
plt.yticks(fontsize=18)

# Add horizontal reference lines
y_discfs = 0.90
y_baseline = 0.81
plt.axhline(y=y_discfs, color='orange', linestyle='--', linewidth=3)
plt.axhline(y=y_baseline, color='green', linestyle='--', linewidth=3)

# Get current x-axis limit
x_max = plt.gca().get_xlim()[1]

# Add text labels slightly above the lines
offset = 0.01  # small vertical offset above the line
plt.text(x_max, y_discfs + offset, 'DISAFS', color='orange', fontsize=16, ha='left', va='bottom')
plt.text(x_max, y_baseline + offset, 'Baseline', color='green', fontsize=16, ha='left', va='bottom')

# Adjust axis limits to include margins
plt.xlim(0, margin * 2 + len(methods) * bar_width)
plt.ylim(0.7, 1)

# Clean visual style
for spine in ["top", "right"]:
    plt.gca().spines[spine].set_visible(False)

plt.grid(False)
plt.tight_layout()
plt.show()


######################Comparison between SFFS and SISAFS  ################
######################  COLON ############################################
import matplotlib.pyplot as plt
import numpy as np

# Parameters
bar_width = 0.3
margin = 0.4  # left margin
gap_between_groups = 0.8  # gap between DISCFS/DFFS and SISCFS/SFFS group

# Define methods and accuracies
methods = ["DISAFS", "DFFS", "SISAFS", "SFFS-1", "SFFS-2"]
accuracies = [0.84, 0.82, 0.80, 0.78, 0.76]

# Define colors
colors = ["red", "deepskyblue", "red", "deepskyblue", "deepskyblue"]

# Compute x positions:
# DISCFS and DFFS placed next to each other (no gap)
x_discfs = margin
x_dffs = x_discfs + bar_width

# Add a space (gap_between_groups) after DFFS
x_siscfs = x_dffs + gap_between_groups
x_sffs1 = x_siscfs + bar_width
x_sffs2 = x_sffs1 + bar_width

x_positions = [x_discfs, x_dffs, x_siscfs, x_sffs1, x_sffs2]

# Create the bar chart
plt.bar(x_positions, accuracies, width=bar_width, color=colors, edgecolor='black', align='edge')

# Add baseline dashed line and dynamic label
base = 0.74
plt.axhline(y=base, color='green', linestyle='--', linewidth=3)

# Get current x-axis maximum dynamically
x_max = plt.gca().get_xlim()[1]

# Add dynamic text label slightly above the line
offset = 0.01  # small vertical offset above the line
plt.text(x_max, base + offset, 'Baseline', color='green', fontsize=16, ha='left', va='bottom')

# Labels and title
plt.xlabel("Methods", fontsize=24)
plt.ylabel("Accuracy", fontsize=24)
plt.title("Colon", fontsize=30)

# Center method labels under each bar
plt.xticks([x + bar_width / 2 for x in x_positions], methods, rotation=45, fontsize=16)
plt.yticks(fontsize=18)

# Set axis limits
plt.xlim(0, x_sffs2 + bar_width + margin)
plt.ylim(0.7, 1)

# Clean style
for spine in ["top", "right"]:
    plt.gca().spines[spine].set_visible(False)

plt.grid(False)
plt.tight_layout()
plt.show()

######################  CNS ############################################
import matplotlib.pyplot as plt
import numpy as np

# Parameters
bar_width = 0.3
margin = 0.4  # left margin
gap_between_groups = 0.8  # gap between DISCFS/DFFS and SISCFS/SFFS group

# Define methods and accuracies
methods = ["DISAFS", "DFFS", "SISAFS", "SFFS-1", "SFFS-2"]
accuracies = [0.68,	0.58,	0.60,	0.68,	0.48]

# Define colors
colors = ["red", "deepskyblue", "red", "deepskyblue", "deepskyblue"]

# Compute x positions:
# DISCFS and DFFS placed next to each other (no gap)
x_discfs = margin
x_dffs = x_discfs + bar_width

# Add a space (gap_between_groups) after DFFS
x_siscfs = x_dffs + gap_between_groups
x_sffs1 = x_siscfs + bar_width
x_sffs2 = x_sffs1 + bar_width

x_positions = [x_discfs, x_dffs, x_siscfs, x_sffs1, x_sffs2]

# Create the bar chart
plt.bar(x_positions, accuracies, width=bar_width, color=colors, edgecolor='black', align='edge')

# Add baseline dashed line and dynamic label
base = 0.60
plt.axhline(y=base, color='green', linestyle='--', linewidth=3)

# Get current x-axis maximum dynamically
x_max = plt.gca().get_xlim()[1]

# Add dynamic text label slightly above the line
offset = 0.01  # small vertical offset above the line
plt.text(x_max, base + offset, 'Baseline', color='green', fontsize=16, ha='left', va='bottom')

# Labels and title
plt.xlabel("Methods", fontsize=24)
plt.ylabel("Accuracy", fontsize=24)
plt.title("CNS", fontsize=30)

# Center method labels under each bar
plt.xticks([x + bar_width / 2 for x in x_positions], methods, rotation=45, fontsize=16)
plt.yticks(fontsize=18)

# Set axis limits
plt.xlim(0, x_sffs2 + bar_width + margin)
plt.ylim(0.4, 1)

# Clean style
for spine in ["top", "right"]:
    plt.gca().spines[spine].set_visible(False)

plt.grid(False)
plt.tight_layout()
plt.show()



######################  GLI ############################################
import matplotlib.pyplot as plt
import numpy as np

# Parameters
bar_width = 0.3
margin = 0.4  # left margin
gap_between_groups = 0.8  # gap between DISCFS/DFFS and SISCFS/SFFS group

# Define methods and accuracies
methods = ["DISAFS", "DFFS", "SISAFS", "SFFS-1", "SFFS-2"]
accuracies = [0.86,	0.91,	0.91,	0.85,	0.76]

# Define colors
colors = ["red", "deepskyblue", "red", "deepskyblue", "deepskyblue"]

# Compute x positions:
# DISCFS and DFFS placed next to each other (no gap)
x_discfs = margin
x_dffs = x_discfs + bar_width

# Add a space (gap_between_groups) after DFFS
x_siscfs = x_dffs + gap_between_groups
x_sffs1 = x_siscfs + bar_width
x_sffs2 = x_sffs1 + bar_width

x_positions = [x_discfs, x_dffs, x_siscfs, x_sffs1, x_sffs2]

# Create the bar chart
plt.bar(x_positions, accuracies, width=bar_width, color=colors, edgecolor='black', align='edge')

# Add baseline dashed line and dynamic label
base = 0.81
plt.axhline(y=base, color='green', linestyle='--', linewidth=3)

# Get current x-axis maximum dynamically
x_max = plt.gca().get_xlim()[1]

# Add dynamic text label slightly above the line
offset = 0.01  # small vertical offset above the line
plt.text(x_max, base + offset, 'Baseline', color='green', fontsize=16, ha='left', va='bottom')

# Labels and title
plt.xlabel("Methods", fontsize=24)
plt.ylabel("Accuracy", fontsize=24)
plt.title("GLI", fontsize=30)

# Center method labels under each bar
plt.xticks([x + bar_width / 2 for x in x_positions], methods, rotation=45, fontsize=16)
plt.yticks(fontsize=18)

# Set axis limits
plt.xlim(0, x_sffs2 + bar_width + margin)
plt.ylim(0.7, 1)

# Clean style
for spine in ["top", "right"]:
    plt.gca().spines[spine].set_visible(False)

plt.grid(False)
plt.tight_layout()
plt.show()


######################  SMK ############################################
import matplotlib.pyplot as plt
import numpy as np

# Parameters
bar_width = 0.3
margin = 0.4  # left margin
gap_between_groups = 0.8  # gap between DISCFS/DFFS and SISCFS/SFFS group

# Define methods and accuracies
methods = ["DISAFS", "DFFS", "SISAFS", "SFFS-1", "SFFS-2"]
accuracies = [0.69,	0.70,	0.66,	0.65,	0.58]

# Define colors
colors = ["red", "deepskyblue", "red", "deepskyblue", "deepskyblue"]

# Compute x positions:
# DISCFS and DFFS placed next to each other (no gap)
x_discfs = margin
x_dffs = x_discfs + bar_width

# Add a space (gap_between_groups) after DFFS
x_siscfs = x_dffs + gap_between_groups
x_sffs1 = x_siscfs + bar_width
x_sffs2 = x_sffs1 + bar_width

x_positions = [x_discfs, x_dffs, x_siscfs, x_sffs1, x_sffs2]

# Create the bar chart
plt.bar(x_positions, accuracies, width=bar_width, color=colors, edgecolor='black', align='edge')

# Add baseline dashed line and dynamic label
base = 0.59
plt.axhline(y=base, color='green', linestyle='--', linewidth=3)

# Get current x-axis maximum dynamically
x_max = plt.gca().get_xlim()[1]

# Add dynamic text label slightly above the line
offset = 0.01  # small vertical offset above the line
plt.text(x_max, base + offset, 'Baseline', color='green', fontsize=16, ha='left', va='bottom')

# Labels and title
plt.xlabel("Methods", fontsize=24)
plt.ylabel("Accuracy", fontsize=24)
plt.title("SMK", fontsize=30)

# Center method labels under each bar
plt.xticks([x + bar_width / 2 for x in x_positions], methods, rotation=45, fontsize=16)
plt.yticks(fontsize=18)

# Set axis limits
plt.xlim(0, x_sffs2 + bar_width + margin)
plt.ylim(0.4, 1)

# Clean style
for spine in ["top", "right"]:
    plt.gca().spines[spine].set_visible(False)

plt.grid(False)
plt.tight_layout()
plt.show()


######################  COVID ############################################
import matplotlib.pyplot as plt
import numpy as np

# Parameters
bar_width = 0.3
margin = 0.4  # left margin
gap_between_groups = 0.8  # gap between DISCFS/DFFS and SISCFS/SFFS group

# Define methods and accuracies
methods = ["DISAFS", "DFFS", "SISAFS", "SFFS-1", "SFFS-2"]
accuracies = [0.43,	0.72,	0.65,	0.71,	0.69]

# Define colors
colors = ["red", "deepskyblue", "red", "deepskyblue", "deepskyblue"]

# Compute x positions:
# DISCFS and DFFS placed next to each other (no gap)
x_discfs = margin
x_dffs = x_discfs + bar_width

# Add a space (gap_between_groups) after DFFS
x_siscfs = x_dffs + gap_between_groups
x_sffs1 = x_siscfs + bar_width
x_sffs2 = x_sffs1 + bar_width

x_positions = [x_discfs, x_dffs, x_siscfs, x_sffs1, x_sffs2]

# Create the bar chart
plt.bar(x_positions, accuracies, width=bar_width, color=colors, edgecolor='black', align='edge')

# Add baseline dashed line and dynamic label
base = 0.68
plt.axhline(y=base, color='green', linestyle='--', linewidth=3)

# Get current x-axis maximum dynamically
x_max = plt.gca().get_xlim()[1]

# Add dynamic text label slightly above the line
offset = 0.01  # small vertical offset above the line
plt.text(x_max, base + offset, 'Baseline', color='green', fontsize=16, ha='left', va='bottom')

# Labels and title
plt.xlabel("Methods", fontsize=24)
plt.ylabel("Accuracy", fontsize=24)
plt.title("Covid-19", fontsize=30)

# Center method labels under each bar
plt.xticks([x + bar_width / 2 for x in x_positions], methods, rotation=45, fontsize=16)
plt.yticks(fontsize=18)

# Set axis limits
plt.xlim(0, x_sffs2 + bar_width + margin)
plt.ylim(0.4, 1)

# Clean style
for spine in ["top", "right"]:
    plt.gca().spines[spine].set_visible(False)

plt.grid(False)
plt.tight_layout()
plt.show()



######################  Leukemia ############################################
import matplotlib.pyplot as plt
import numpy as np

# Parameters
bar_width = 0.3
margin = 0.4  # left margin
gap_between_groups = 0.8  # gap between DISCFS/DFFS and SISCFS/SFFS group

# Define methods and accuracies
methods = ["DISAFS", "DFFS", "SISAFS", "SFFS-1", "SFFS-2"]
accuracies = [0.89,	0.90,	0.89,	0.83,	0.86]

# Define colors
colors = ["red", "deepskyblue", "red", "deepskyblue", "deepskyblue"]

# Compute x positions:
# DISCFS and DFFS placed next to each other (no gap)
x_discfs = margin
x_dffs = x_discfs + bar_width

# Add a space (gap_between_groups) after DFFS
x_siscfs = x_dffs + gap_between_groups
x_sffs1 = x_siscfs + bar_width
x_sffs2 = x_sffs1 + bar_width

x_positions = [x_discfs, x_dffs, x_siscfs, x_sffs1, x_sffs2]

# Create the bar chart
plt.bar(x_positions, accuracies, width=bar_width, color=colors, edgecolor='black', align='edge')

# Add baseline dashed line and dynamic label
base = 0.82
plt.axhline(y=base, color='green', linestyle='--', linewidth=3)

# Get current x-axis maximum dynamically
x_max = plt.gca().get_xlim()[1]

# Add dynamic text label slightly above the line
offset = 0.01  # small vertical offset above the line
plt.text(x_max, base + offset, 'Baseline', color='green', fontsize=16, ha='left', va='bottom')

# Labels and title
plt.xlabel("Methods", fontsize=24)
plt.ylabel("Accuracy", fontsize=24)
plt.title("Leukemia", fontsize=30)

# Center method labels under each bar
plt.xticks([x + bar_width / 2 for x in x_positions], methods, rotation=45, fontsize=16)
plt.yticks(fontsize=18)

# Set axis limits
plt.xlim(0, x_sffs2 + bar_width + margin)
plt.ylim(0.7, 1)

# Clean style
for spine in ["top", "right"]:
    plt.gca().spines[spine].set_visible(False)

plt.grid(False)
plt.tight_layout()
plt.show()


######################  MLL ############################################
import matplotlib.pyplot as plt
import numpy as np

# Parameters
bar_width = 0.3
margin = 0.4  # left margin
gap_between_groups = 0.8  # gap between DISCFS/DFFS and SISCFS/SFFS group

# Define methods and accuracies
methods = ["DISAFS", "DFFS", "SISAFS", "SFFS-1", "SFFS-2"]
accuracies = [0.96,	0.96,	0.93,	0.82,	0.93]

# Define colors
colors = ["red", "deepskyblue", "red", "deepskyblue", "deepskyblue"]

# Compute x positions:
# DISCFS and DFFS placed next to each other (no gap)
x_discfs = margin
x_dffs = x_discfs + bar_width

# Add a space (gap_between_groups) after DFFS
x_siscfs = x_dffs + gap_between_groups
x_sffs1 = x_siscfs + bar_width
x_sffs2 = x_sffs1 + bar_width

x_positions = [x_discfs, x_dffs, x_siscfs, x_sffs1, x_sffs2]

# Create the bar chart
plt.bar(x_positions, accuracies, width=bar_width, color=colors, edgecolor='black', align='edge')

# Add baseline dashed line and dynamic label
base = 0.85
plt.axhline(y=base, color='green', linestyle='--', linewidth=3)

# Get current x-axis maximum dynamically
x_max = plt.gca().get_xlim()[1]

# Add dynamic text label slightly above the line
offset = 0.01  # small vertical offset above the line
plt.text(x_max, base + offset, 'Baseline', color='green', fontsize=16, ha='left', va='bottom')

# Labels and title
plt.xlabel("Methods", fontsize=24)
plt.ylabel("Accuracy", fontsize=24)
plt.title("MLL", fontsize=30)

# Center method labels under each bar
plt.xticks([x + bar_width / 2 for x in x_positions], methods, rotation=45, fontsize=16)
plt.yticks(fontsize=18)

# Set axis limits
plt.xlim(0, x_sffs2 + bar_width + margin)
plt.ylim(0.7, 1)

# Clean style
for spine in ["top", "right"]:
    plt.gca().spines[spine].set_visible(False)

plt.grid(False)
plt.tight_layout()
plt.show()


######################  SRBCT ############################################
import matplotlib.pyplot as plt
import numpy as np

# Parameters
bar_width = 0.3
margin = 0.4  # left margin
gap_between_groups = 0.8  # gap between DISCFS/DFFS and SISCFS/SFFS group

# Define methods and accuracies
methods = ["DISAFS", "DFFS", "SISAFS", "SFFS-1", "SFFS-2"]
accuracies = [0.90,	0.88,	0.87,	0.81,	0.84]

# Define colors
colors = ["red", "deepskyblue", "red", "deepskyblue", "deepskyblue"]

# Compute x positions:
# DISCFS and DFFS placed next to each other (no gap)
x_discfs = margin
x_dffs = x_discfs + bar_width

# Add a space (gap_between_groups) after DFFS
x_siscfs = x_dffs + gap_between_groups
x_sffs1 = x_siscfs + bar_width
x_sffs2 = x_sffs1 + bar_width

x_positions = [x_discfs, x_dffs, x_siscfs, x_sffs1, x_sffs2]

# Create the bar chart
plt.bar(x_positions, accuracies, width=bar_width, color=colors, edgecolor='black', align='edge')

# Add baseline dashed line and dynamic label
base = 0.81
plt.axhline(y=base, color='green', linestyle='--', linewidth=3)

# Get current x-axis maximum dynamically
x_max = plt.gca().get_xlim()[1]

# Add dynamic text label slightly above the line
offset = 0.01  # small vertical offset above the line
plt.text(x_max, base + offset, 'Baseline', color='green', fontsize=16, ha='left', va='bottom')

# Labels and title
plt.xlabel("Methods", fontsize=24)
plt.ylabel("Accuracy", fontsize=24)
plt.title("SRBCT", fontsize=30)

# Center method labels under each bar
plt.xticks([x + bar_width / 2 for x in x_positions], methods, rotation=45, fontsize=16)
plt.yticks(fontsize=18)

# Set axis limits
plt.xlim(0, x_sffs2 + bar_width + margin)
plt.ylim(0.7, 1)

# Clean style
for spine in ["top", "right"]:
    plt.gca().spines[spine].set_visible(False)

plt.grid(False)
plt.tight_layout()
plt.show()



##################  Radar plot ####################
##################  Colon  ########################

import matplotlib.pyplot as plt
import numpy as np

# === Data ===
metrics = ["Accuracy", "Precision", "Recall", "F-score"]
before = [0.74, 0.71, 0.71, 0.71]
after = [0.97, 0.98, 0.95, 0.96]

# === Prepare data for radar plot ===
N = len(metrics)
values_before = before + [before[0]]  # close the loop
values_after = after + [after[0]]
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

# === Create radar chart ===
plt.figure(figsize=(7, 7))
ax = plt.subplot(111, polar=True)

# Plot After first so it's behind the 'Before' layer
ax.plot(angles, values_after, color="deepskyblue", linewidth=2, linestyle='-', label="After SISAFS-DIBS", zorder=1)
ax.fill(angles, values_after, color="deepskyblue", alpha=0.4, zorder=1)

# Plot Before on top for clear visibility
ax.plot(angles, values_before, color="lightcoral", linewidth=2, linestyle='-', label="With all features", zorder=2)
ax.fill(angles, values_before, color="lightcoral", alpha=0.4, zorder=2)


# === Customize axes ===
ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics, fontsize=16)
ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
ax.set_yticklabels(["0.6", "0.7", "0.8", "0.9", "1.0"], fontsize=16)
ax.set_ylim(0.6, 1.0)

# === Style and legend ===
plt.title("Colon", size=36, y=1.08)
plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1), frameon=False, fontsize=14)

# Remove gridlines for cleaner look
ax.spines['polar'].set_visible(False)
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()


##################  CNS  ########################

import matplotlib.pyplot as plt
import numpy as np

# === Data ===
metrics = ["Accuracy", "Precision", "Recall", "F-score"]
before = [0.6, 0.57, 0.58, 0.57]
after = [0.85, 0.84, 0.82, 0.82]

# === Prepare data for radar plot ===
N = len(metrics)
values_before = before + [before[0]]  # close the loop
values_after = after + [after[0]]
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

# === Create radar chart ===
plt.figure(figsize=(7, 7))
ax = plt.subplot(111, polar=True)

# Plot After first so it's behind the 'Before' layer
ax.plot(angles, values_after, color="deepskyblue", linewidth=2, linestyle='-', label="After SISAFS-DIBS", zorder=1)
ax.fill(angles, values_after, color="deepskyblue", alpha=0.4, zorder=1)

# Plot Before on top for clear visibility
ax.plot(angles, values_before, color="lightcoral", linewidth=2, linestyle='-', label="With all features", zorder=2)
ax.fill(angles, values_before, color="lightcoral", alpha=0.4, zorder=2)

# === Customize axes ===
ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics, fontsize=16)
ax.set_yticks([0.5,0.6, 0.7, 0.8, 0.9, 1.0])
ax.set_yticklabels(["0.5","0.6", "0.7", "0.8", "0.9", "1.0"], fontsize=16)
ax.set_ylim(0.5, 1.0)  # slightly lower limit to ensure full shape is visible

# === Style and legend ===
plt.title("CNS", size=36, y=1.08)
plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1), frameon=False, fontsize=14)

# Remove gridlines for cleaner look
ax.spines['polar'].set_visible(False)
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()


##################  GLI  ########################

import matplotlib.pyplot as plt
import numpy as np

# === Data ===
metrics = ["Accuracy", "Precision", "Recall", "F-score"]
before = [0.81,	0.79,	0.80,	0.79]
after = [0.93,	0.92,	0.92,	0.92]

# === Prepare data for radar plot ===
N = len(metrics)
values_before = before + [before[0]]  # close the loop
values_after = after + [after[0]]
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

# === Create radar chart ===
plt.figure(figsize=(7, 7))
ax = plt.subplot(111, polar=True)

# Plot After first so it's behind the 'Before' layer
ax.plot(angles, values_after, color="deepskyblue", linewidth=2, linestyle='-', label="After SISAFS-DIBS", zorder=1)
ax.fill(angles, values_after, color="deepskyblue", alpha=0.4, zorder=1)

# Plot Before on top for clear visibility
ax.plot(angles, values_before, color="lightcoral", linewidth=2, linestyle='-', label="With all features", zorder=2)
ax.fill(angles, values_before, color="lightcoral", alpha=0.4, zorder=2)

# === Customize axes ===
ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics, fontsize=16)
ax.set_yticks([  0.7, 0.8, 0.9, 1.0])
ax.set_yticklabels([ "0.7", "0.8", "0.9", "1.0"], fontsize=16)
ax.set_ylim(0.7, 1.0)  # slightly lower limit to ensure full shape is visible

# === Style and legend ===
plt.title("GLI", size=36, y=1.08)
plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1), frameon=False, fontsize=14)

# Remove gridlines for cleaner look
ax.spines['polar'].set_visible(False)
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()



##################  SMK  ########################

import matplotlib.pyplot as plt
import numpy as np

# === Data ===
metrics = ["Accuracy", "Precision", "Recall", "F-score"]
before = [0.59,	0.59,	0.59,	0.59]
after = [0.78,	0.78,	0.78,	0.78]

# === Prepare data for radar plot ===
N = len(metrics)
values_before = before + [before[0]]  # close the loop
values_after = after + [after[0]]
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

# === Create radar chart ===
plt.figure(figsize=(7, 7))
ax = plt.subplot(111, polar=True)

# Plot After first so it's behind the 'Before' layer
ax.plot(angles, values_after, color="deepskyblue", linewidth=2, linestyle='-', label="After SISAFS-DIBS", zorder=1)
ax.fill(angles, values_after, color="deepskyblue", alpha=0.4, zorder=1)

# Plot Before on top for clear visibility
ax.plot(angles, values_before, color="lightcoral", linewidth=2, linestyle='-', label="With all features", zorder=2)
ax.fill(angles, values_before, color="lightcoral", alpha=0.4, zorder=2)

# === Customize axes ===
ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics, fontsize=16)
ax.set_yticks([ 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.set_yticklabels(["0.5","0.6", "0.7", "0.8", "0.9", "1.0"], fontsize=16)
ax.set_ylim(0.5, 1.0)  # slightly lower limit to ensure full shape is visible

# === Style and legend ===
plt.title("SMK", size=36, y=1.08)
plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1), frameon=False, fontsize=14)

# Remove gridlines for cleaner look
ax.spines['polar'].set_visible(False)
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()


##################  Covid  ########################

import matplotlib.pyplot as plt
import numpy as np

# === Data ===
metrics = ["Accuracy", "Precision", "Recall", "F-score"]
before = [0.68,	0.66,	0.62,	0.62]
after = [0.75,	0.72,	0.71,	0.71]

# === Prepare data for radar plot ===
N = len(metrics)
values_before = before + [before[0]]  # close the loop
values_after = after + [after[0]]
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

# === Create radar chart ===
plt.figure(figsize=(7, 7))
ax = plt.subplot(111, polar=True)

# Plot After first so it's behind the 'Before' layer
ax.plot(angles, values_after, color="deepskyblue", linewidth=2, linestyle='-', label="After SISAFS-DIBS", zorder=1)
ax.fill(angles, values_after, color="deepskyblue", alpha=0.4, zorder=1)

# Plot Before on top for clear visibility
ax.plot(angles, values_before, color="lightcoral", linewidth=2, linestyle='-', label="With all features", zorder=2)
ax.fill(angles, values_before, color="lightcoral", alpha=0.4, zorder=2)

# === Customize axes ===
ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics, fontsize=16)
ax.set_yticks([  0.6, 0.7, 0.8, 0.9, 1.0])
ax.set_yticklabels(["0.6", "0.7", "0.8", "0.9", "1.0"], fontsize=16)
ax.set_ylim(0.6, 1.0)  # slightly lower limit to ensure full shape is visible

# === Style and legend ===
plt.title("Covid-19", size=36, y=1.08)
plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1), frameon=False, fontsize=14)

# Remove gridlines for cleaner look
ax.spines['polar'].set_visible(False)
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()


##################  Leukemia  ########################

import matplotlib.pyplot as plt
import numpy as np

# === Data ===
metrics = ["Accuracy", "Precision", "Recall", "F-score"]
before = [0.82,	0.80,	0.83,	0.80]
after = [0.97,	0.99,	0.95,	0.96]

# === Prepare data for radar plot ===
N = len(metrics)
values_before = before + [before[0]]  # close the loop
values_after = after + [after[0]]
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

# === Create radar chart ===
plt.figure(figsize=(7, 7))
ax = plt.subplot(111, polar=True)

# Plot After first so it's behind the 'Before' layer
ax.plot(angles, values_after, color="deepskyblue", linewidth=2, linestyle='-', label="After SISAFS-DIBS", zorder=1)
ax.fill(angles, values_after, color="deepskyblue", alpha=0.4, zorder=1)

# Plot Before on top for clear visibility
ax.plot(angles, values_before, color="lightcoral", linewidth=2, linestyle='-', label="With all features", zorder=2)
ax.fill(angles, values_before, color="lightcoral", alpha=0.4, zorder=2)

# === Customize axes ===
ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics, fontsize=16)
ax.set_yticks([  0.7, 0.8, 0.9, 1.0])
ax.set_yticklabels([ "0.7", "0.8", "0.9", "1.0"], fontsize=16)
ax.set_ylim(0.7, 1.0)  # slightly lower limit to ensure full shape is visible

# === Style and legend ===
plt.title("Leukemia", size=36, y=1.08)
plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1), frameon=False, fontsize=14)

# Remove gridlines for cleaner look
ax.spines['polar'].set_visible(False)
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()



##################  MLL  ########################

import matplotlib.pyplot as plt
import numpy as np

# === Data ===
metrics = ["Accuracy", "Precision", "Recall", "F-score"]
before = [0.85,	0.87,	0.86,	0.84]
after = [0.96,	0.97,	0.96,	0.96]

# === Prepare data for radar plot ===
N = len(metrics)
values_before = before + [before[0]]  # close the loop
values_after = after + [after[0]]
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

# === Create radar chart ===
plt.figure(figsize=(7, 7))
ax = plt.subplot(111, polar=True)

# Plot After first so it's behind the 'Before' layer
ax.plot(angles, values_after, color="deepskyblue", linewidth=2, linestyle='-', label="After SISAFS-DIBS", zorder=1)
ax.fill(angles, values_after, color="deepskyblue", alpha=0.4, zorder=1)

# Plot Before on top for clear visibility
ax.plot(angles, values_before, color="lightcoral", linewidth=2, linestyle='-', label="With all features", zorder=2)
ax.fill(angles, values_before, color="lightcoral", alpha=0.4, zorder=2)

# === Customize axes ===
ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics, fontsize=16)
ax.set_yticks([   0.8, 0.9, 1.0])
ax.set_yticklabels([  "0.8", "0.9", "1.0"], fontsize=16)
ax.set_ylim(0.8, 1.0)  # slightly lower limit to ensure full shape is visible

# === Style and legend ===
plt.title("MLL", size=36, y=1.08)
plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1), frameon=False, fontsize=14)

# Remove gridlines for cleaner look
ax.spines['polar'].set_visible(False)
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()


##################  SRBCT  ########################

import matplotlib.pyplot as plt
import numpy as np

# === Data ===
metrics = ["Accuracy", "Precision", "Recall", "F-score"]
before = [0.81,	0.79,	0.80,	0.77]
after = [0.99,	0.99,	0.99,	0.99]

# === Prepare data for radar plot ===
N = len(metrics)
values_before = before + [before[0]]  # close the loop
values_after = after + [after[0]]
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

# === Create radar chart ===
plt.figure(figsize=(7, 7))
ax = plt.subplot(111, polar=True)

# Plot After first so it's behind the 'Before' layer
ax.plot(angles, values_after, color="deepskyblue", linewidth=2, linestyle='-', label="After SISAFS-DIBS", zorder=1)
ax.fill(angles, values_after, color="deepskyblue", alpha=0.4, zorder=1)

# Plot Before on top for clear visibility
ax.plot(angles, values_before, color="lightcoral", linewidth=2, linestyle='-', label="With all features", zorder=2)
ax.fill(angles, values_before, color="lightcoral", alpha=0.4, zorder=2)

# === Customize axes ===
ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics, fontsize=16)
ax.set_yticks([  0.7, 0.8, 0.9, 1.0])
ax.set_yticklabels([  "0.7","0.8", "0.9", "1.0"], fontsize=16)
ax.set_ylim(0.7, 1.0)  # slightly lower limit to ensure full shape is visible

# === Style and legend ===
plt.title("SRBCT", size=36, y=1.08)
plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1), frameon=False, fontsize=14)

# Remove gridlines for cleaner look
ax.spines['polar'].set_visible(False)
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()















############# average similarity image

file_path = r'D:\Papers\SIFS\Plots\ISCFS-DIBS\MLL\performance_historyISCFSDIBS.pkl'

### For Loading    
with open(file_path, 'rb') as f:
    performance_history = pickle.load(f)


import matplotlib.pyplot as plt

if performance_history:
    iteration_values = [h[0] for h in performance_history]
    best_scores = [h[1] for h in performance_history]
    avg_sims = [h[2] for h in performance_history]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # --- Plot Accuracy (still with conditional coloring) ---
    for i in range(1, len(iteration_values)):
        x_segment = iteration_values[i-1:i+1]
        y_segment = best_scores[i-1:i+1]

        color = 'blue'
        if best_scores[i] > best_scores[i-1] and avg_sims[i] < avg_sims[i-1]:
            color = 'orange'

        acc_plot = ax1.plot(x_segment, y_segment, marker='o',
                            linestyle='-', markersize=8, color=color)

    ax1.set_xlabel('Iteration', fontsize=30)
    ax1.set_ylabel('Accuracy', fontsize=30, color='blue')
    ax1.tick_params(axis='y', labelcolor='blue', labelsize=26)
    ax1.tick_params(axis='x', labelsize=26)
    ax1.set_ylim(0.5, 1.01)

    # --- Plot AvgSim (still with conditional coloring) ---
    ax2 = ax1.twinx()

    for i in range(1, len(iteration_values)):
        x_segment = iteration_values[i-1:i+1]
        y_segment = avg_sims[i-1:i+1]

        color = 'red'
        if avg_sims[i] < avg_sims[i-1] and best_scores[i] > best_scores[i-1]:
            color = 'green'

        sim_plot = ax2.plot(x_segment, y_segment, marker='x',
                            linestyle='--', color=color, alpha=0.7)

    ax2.set_ylabel('AvgSim', fontsize=30, color='red')
    ax2.tick_params(axis='y', labelcolor='red', labelsize=26)
    ax2.set_ylim(0, 1)

    # ======================================================
    # TWO LEGEND ENTRIES ONLY (your screenshot style)
    # ======================================================
    legend_handles = [
        plt.Line2D([], [], color='blue', marker='o', linestyle='-', label='Accuracy with SISAFS-DIBS'),
        plt.Line2D([], [], color='red', marker='x', linestyle='--', label='Average feature similarity in the beam')
    ]

    ax1.legend(handles=legend_handles, fontsize=20, loc='lower right')

    ax1.grid(True)
    plt.title('MLL', fontsize=36)
    plt.tight_layout()
    plt.show()




#################  Correlation Analysis
################ Correlation analysis

def fisher_score(X, y):
    scores = []
    classes = np.unique(y)
    overall_mean = X.mean(axis=0)
    
    for col in X.columns:
        numerator = 0
        denominator = 0
        for c in classes:
            x_c = X[y == c][col]
            mean_c = x_c.mean()
            var_c = x_c.var()
            n_c = len(x_c)
            numerator += n_c * (mean_c - overall_mean[col]) ** 2
            denominator += n_c * var_c
        # To avoid divide-by-zero
        score = numerator / (denominator + 1e-6)
        scores.append(score)
    
    return np.array(scores)

# Calculate scores and get top 50 features
scores = fisher_score(X, y)
Fisher = np.argsort(scores)[-50:][::-1]
ranked_features = X.columns[Fisher]

print("Top 50 feature indices:", Fisher)
print("Top 50 feature names:", ranked_features.tolist())



####CPR

scores = np.array(sortings["aggregated"])
feature_names = np.array(X.columns)  # or your list of feature names
# Get indices of top 50 scores
top50_indices = np.argsort(scores)[-50:][::-1]

# Get top 50 feature names
ranked_features = feature_names[top50_indices].tolist()
selected_X=X[ranked_features]


### SFFS

selected_X=X[np.array(ranked_features)[:50]]


import numpy as np
import pandas as pd

# Assuming X is a pandas DataFrame and ranked_features contains the top features
selected_X = X[ranked_features]

# Compute correlation matrix
corr_matrix = selected_X.corr().values

# Extract upper triangle without the diagonal
upper_tri_indices = np.triu_indices_from(corr_matrix, k=1)
upper_tri_values = corr_matrix[upper_tri_indices]

# Calculate average pairwise correlation
avg_pairwise_corr = np.mean(upper_tri_values)
avg_absolute_pairwise_corr = np.mean(np.abs(upper_tri_values))
# print("Average pairwise correlation between top 50 features:", round(avg_pairwise_corr,2))
print("Average ABSOLUTE pairwise correlation between top 50 features:", round(avg_absolute_pairwise_corr, 2))