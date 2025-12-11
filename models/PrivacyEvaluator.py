import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.decomposition import PCA
from collections import Counter
import warnings

class PrivacyEvaluator:
    def __init__(self, real_train_df, real_test_df, synthetic_df, 
                 static_num_cols, static_cat_cols, 
                 temp_num_cols, temp_cat_cols, 
                 max_seq_len=None, n_samples=None, n_components_pca=None):
        
        self.static_num_cols_ = static_num_cols
        self.static_cat_cols_ = static_cat_cols
        self.temp_num_cols_ = temp_num_cols
        self.temp_cat_cols_ = temp_cat_cols
        self.use_pca_ = n_components_pca is not None
        
        self.all_cat_cols_ = list(static_cat_cols) + list(temp_cat_cols)
        
        if n_samples:
            print(f"Sampling data down to {n_samples} patients.")
            real_train_df = real_train_df.sample(n=min(n_samples, len(real_train_df)), random_state=42)
            real_test_df = real_test_df.sample(n=min(n_samples, len(real_test_df)), random_state=42)
            synthetic_df = synthetic_df.sample(n=min(n_samples, len(synthetic_df)), random_state=42)
        
        self.real_train_df = real_train_df
        self.synthetic_df = synthetic_df

        if max_seq_len is None:
            check_col = self.temp_num_cols_[0] if len(self.temp_num_cols_) > 0 else self.temp_cat_cols_[0]
            self.max_seq_len_ = real_train_df[check_col].apply(len).max()
        else:
            self.max_seq_len_ = max_seq_len
        print(f"Using max_seq_len: {self.max_seq_len_}")

        self.static_encoder_ = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        if self.static_cat_cols_:
            static_fit_df = real_train_df[self.static_cat_cols_].explode(self.static_cat_cols_).fillna('missing')
            self.static_encoder_.fit(static_fit_df)
            print(f"Fitted static OneHotEncoder on {len(self.static_encoder_.get_feature_names_out())} features.")

        self.temp_encoder_ = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        if self.temp_cat_cols_:
            temp_fit_df = real_train_df[self.temp_cat_cols_].explode(self.temp_cat_cols_).fillna('missing')
            self.temp_encoder_.fit(temp_fit_df)
            print(f"Fitted temporal OneHotEncoder on {len(self.temp_encoder_.get_feature_names_out())} features.")
            
        print("Flattening and padding training data...")
        self.X_train_flat = self._preprocess_wide_df(real_train_df)
        
        print("Flattening and padding test data...")
        self.X_test_flat = self._preprocess_wide_df(real_test_df)
        
        print("Flattening and padding synthetic data...")
        self.X_synth_flat = self._preprocess_wide_df(synthetic_df)

        if self.use_pca_:
            print(f"Applying PCA, reducing to {n_components_pca} components...")
            self.pca_ = PCA(n_components=n_components_pca)
            self.X_train_flat = self.pca_.fit_transform(self.X_train_flat)
            print(f"PCA explained variance: {np.sum(self.pca_.explained_variance_ratio_):.4f}")
            self.X_test_flat = self.pca_.transform(self.X_test_flat)
            self.X_synth_flat = self.pca_.transform(self.X_synth_flat)
        
        print("Scaling data...")
        self.scaler_ = StandardScaler()
        self.X_train_scaled = self.scaler_.fit_transform(self.X_train_flat)
        self.X_test_scaled = self.scaler_.transform(self.X_test_flat)
        self.X_synth_scaled = self.scaler_.transform(self.X_synth_flat)
        
        print("Data processing complete. Evaluator is ready.")

    def _preprocess_wide_df(self, df):
        all_patient_vectors = []
        
        n_static_num = len(self.static_num_cols_)
        n_static_cat = len(self.static_encoder_.get_feature_names_out()) if self.static_cat_cols_ else 0
        n_temp_num = len(self.temp_num_cols_)
        n_temp_cat = len(self.temp_encoder_.get_feature_names_out()) if self.temp_cat_cols_ else 0
        
        df = df.sort_index()
        
        for index, row in df.iterrows():
            
            static_num_vector = [row[col][0] if len(row[col]) > 0 else 0 for col in self.static_num_cols_]
            
            static_cat_vector = []
            if self.static_cat_cols_:
                static_cat_list = [row[col][0] if len(row[col]) > 0 else 'missing' for col in self.static_cat_cols_]
                static_cat_df = pd.DataFrame([static_cat_list], columns=self.static_cat_cols_)
                static_cat_vector = self.static_encoder_.transform(static_cat_df).flatten()

            static_vector = np.hstack([static_num_vector, static_cat_vector])
            
            temp_vector_padded = np.zeros((self.max_seq_len_, n_temp_num + n_temp_cat))
            
            current_max_len = 0
            if self.temp_num_cols_:
                 non_empty_num_cols = [col for col in self.temp_num_cols_ if len(row[col]) > 0]
                 if non_empty_num_cols:
                     current_max_len = max(current_max_len, max(len(row[col]) for col in non_empty_num_cols))
            if self.temp_cat_cols_:
                 non_empty_cat_cols = [col for col in self.temp_cat_cols_ if len(row[col]) > 0]
                 if non_empty_cat_cols:
                      current_max_len = max(current_max_len, max(len(row[col]) for col in non_empty_cat_cols))

            if current_max_len == 0:
                 flat_temp_vector = temp_vector_padded.flatten()
                 final_patient_vector = np.hstack([static_vector, flat_temp_vector])
                 all_patient_vectors.append(final_patient_vector)
                 continue

            padded_num_data_list = []
            if self.temp_num_cols_:
                for col in self.temp_num_cols_:
                    original_list = row[col]
                    padding_needed = current_max_len - len(original_list)
                    padded_list = [np.nan] * padding_needed + list(original_list) 
                    padded_num_data_list.append(padded_list)
                num_data_array = pd.DataFrame(padded_num_data_list).T.fillna(0).values 
            else:
                 num_data_array = np.array([]).reshape(current_max_len, 0)
            
            padded_cat_data_list = []
            if self.temp_cat_cols_:
                for col in self.temp_cat_cols_:
                    original_list = row[col]
                    padding_needed = current_max_len - len(original_list)
                    padded_list = ['missing'] * padding_needed + list(original_list)
                    padded_cat_data_list.append(padded_list)
                cat_data_df = pd.DataFrame(padded_cat_data_list).T.fillna('missing') 
                cat_data_encoded = self.temp_encoder_.transform(cat_data_df)
            else:
                 cat_data_encoded = np.array([]).reshape(current_max_len, 0)
                 
            seq_len = current_max_len

            if seq_len > 0:
                combined_temp_features = np.hstack([num_data_array, cat_data_encoded])
                
                actual_len = min(seq_len, self.max_seq_len_)
                temp_vector_padded[-actual_len:] = combined_temp_features[-actual_len:] 
            
            flat_temp_vector = temp_vector_padded.flatten()
            
            final_patient_vector = np.hstack([static_vector, flat_temp_vector])
            all_patient_vectors.append(final_patient_vector)
                
        return np.vstack(all_patient_vectors)

    
    def membership_inference_attack(self, n_neighbors=1):
        if self.X_train_scaled.size == 0 or self.X_synth_scaled.size == 0:
            return 0.5

        nn_model = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
        nn_model.fit(self.X_synth_scaled, np.zeros(len(self.X_synth_scaled)))
        
        distances_train, _ = nn_model.kneighbors(self.X_train_scaled)
        d_train = np.mean(distances_train, axis=1)

        distances_test, _ = nn_model.kneighbors(self.X_test_scaled)
        d_test = np.mean(distances_test, axis=1)

        X_attack = np.concatenate([d_train, d_test]).reshape(-1, 1)
        y_attack = np.concatenate([np.ones(len(d_train)), np.zeros(len(d_test))])
        
        try:
            X_train_mi, X_test_mi, y_train_mi, y_test_mi = train_test_split(
                X_attack, y_attack, test_size=0.3, random_state=42, stratify=y_attack
            )
        except ValueError: # Handle cases with only one class in y_attack
             X_train_mi, X_test_mi, y_train_mi, y_test_mi = train_test_split(
                X_attack, y_attack, test_size=0.3, random_state=42
            )
        
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X_train_mi, y_train_mi)
        
        try:
            if len(np.unique(y_test_mi)) > 1:
                y_pred_proba = clf.predict_proba(X_test_mi)[:, 1]
                auc = roc_auc_score(y_test_mi, y_pred_proba)
            else: # Only one class in test set
                auc = 0.5
        except Exception:
            y_pred = clf.predict(X_test_mi)
            auc = accuracy_score(y_test_mi, y_pred)
            
        return auc

    def re_identification_attack(self, k=1):
        if self.X_train_scaled.size == 0 or self.X_synth_scaled.size == 0:
            return 0.0, 0.0
        
        nn_model = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        nn_model.fit(self.X_train_scaled, np.zeros(len(self.X_train_scaled)))
        
        synth_distances, _ = nn_model.kneighbors(self.X_synth_scaled)
        min_synth_distances = synth_distances[:, 0]

        test_distances, _ = nn_model.kneighbors(self.X_test_scaled)
        min_test_distances = test_distances[:, 0]

        non_zero_test_dist = min_test_distances[min_test_distances > 1e-6]
        if len(non_zero_test_dist) > 0:
            distance_threshold = np.percentile(non_zero_test_dist, 5)
        else:
            distance_threshold = 1e-6

        attack_success_rate = np.sum(min_synth_distances < distance_threshold) / len(min_synth_distances)
        
        baseline_success_rate = np.sum(min_test_distances < distance_threshold) / len(min_test_distances)
        
        return attack_success_rate, baseline_success_rate
    
    def attribute_inference_attack(self, target_attribute_col):
        if target_attribute_col not in self.static_cat_cols_:
             print(f"Warning: Attribute '{target_attribute_col}' is not in static_cat_cols. Skipping.")
             return {'real_data_auc': 0.5, 'synthetic_data_auc': 0.5}

        def get_target_vector(df):
            return df[target_attribute_col].apply(
                lambda x: next((item for item in x if pd.notna(item) and item != 'missing'), 'missing')
                if (isinstance(x, (list, np.ndarray)) and len(x) > 0) else 'missing'
            ).fillna('missing').values

        try:
            y_real = get_target_vector(self.real_train_df)
            y_synth = get_target_vector(self.synthetic_df)
            
            X_real = self.X_train_scaled
            X_synth = self.X_synth_scaled

        except (KeyError, IndexError):
            print(f"Warning: Could not extract attribute {target_attribute_col}.")
            return {'real_data_auc': 0.5, 'synthetic_data_auc': 0.5}

        unique_values = np.unique(y_real)
        if len(unique_values) <= 1:
             print(f"Warning: Attribute {target_attribute_col} has only one class.")
             return {'real_data_auc': 0.5, 'synthetic_data_auc': 0.5}
        elif len(unique_values) > 2:
            most_common = Counter(y_real).most_common(1)[0][0]
            y_real_binary = (y_real == most_common).astype(int)
            y_synth_binary = (y_synth == most_common).astype(int)
        else:
            le = LabelEncoder().fit(y_real)
            y_real_binary = le.transform(y_real)
            y_synth_binary = le.transform(y_synth)
            
        try:
            X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
                X_real, y_real_binary, test_size=0.3, random_state=42, stratify=y_real_binary
            )
        except ValueError:
             X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
                X_real, y_real_binary, test_size=0.3, random_state=42
            )
        
        clf_real = KNeighborsClassifier(n_neighbors=5)
        clf_real.fit(X_train_real, y_train_real)
        try:
            if len(np.unique(y_test_real)) > 1:
                y_pred_real = clf_real.predict_proba(X_test_real)[:, 1]
                auc_real = roc_auc_score(y_test_real, y_pred_real)
            else:
                auc_real = 0.5
        except Exception:
            auc_real = 0.5

        clf_synth = KNeighborsClassifier(n_neighbors=5)
        clf_synth.fit(X_synth, y_synth_binary)
        try:
            if len(np.unique(y_test_real)) > 1:
                y_pred_synth = clf_synth.predict_proba(X_test_real)[:, 1] 
                auc_synth = roc_auc_score(y_test_real, y_pred_synth)
            else:
                auc_synth = 0.5
        except Exception:
            auc_synth = 0.5
        
        return {
            'real_data_auc': auc_real,
            'synthetic_data_auc': auc_synth
        }

    def evaluate_all_attacks(self, sensitive_attribute_cols):
        print("\n" + "="*30)
        print("Running Privacy Evaluation...")
        print("="*30)
        
        mi_auc = self.membership_inference_attack()
        print(f"\n--- 1. Membership Inference Attack ---")
        print(f"Attack Classifier AUC: {mi_auc:.4f}")
        print(f"(Ideal value is 0.5. Higher indicates a privacy leak.)")
        
        reid_attack, reid_baseline = self.re_identification_attack()
        print(f"\n--- 2. Re-identification Attack ---")
        print(f"Attack Success Rate: {reid_attack:.4f}")
        print(f"Baseline (Ideal) Rate: {reid_baseline:.4f} (5th percentile of test set)")
        print(f"(Lower is better. Attack score should be close to baseline.)")

        print(f"\n--- 3. Attribute Inference Attack ---")
        
        results = {
            'membership_inference_auc': mi_auc,
            're_identification_risk': reid_attack,
            're_identification_baseline': reid_baseline,
            'attribute_inference': {}
        }
        
        for attr_col in sensitive_attribute_cols:
            if attr_col not in self.all_cat_cols_:
                print(f"Warning: Sensitive column '{attr_col}' not found. Skipping.")
                continue
                
            try:
                attr_result = self.attribute_inference_attack(attr_col)
                print(f"\nTarget: '{attr_col}'")
                print(f"  Baseline (Train on Real) AUC: {attr_result['real_data_auc']:.4f}")
                print(f"  Attack (Train on Synth) AUC: {attr_result['synthetic_data_auc']:.4f}")
                print(f"  (Privacy is preserved if 'Attack' AUC is not higher than 'Baseline' AUC.)")
                results['attribute_inference'][attr_col] = attr_result
            except Exception as e:
                print(f"Could not evaluate '{attr_col}': {e}")
                results['attribute_inference'][attr_col] = {'real_data_auc': 0.0, 'synthetic_data_auc': 0.0}

        print("="*30)
        print("Privacy Evaluation Complete.")
        print("="*30)
        
        return results