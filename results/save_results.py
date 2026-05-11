import pandas as pd
import os

# Test set results — already computed
test_results = {
    'Strategy': [
        'EfficientNet_S1', 'EfficientNet_S2', 'EfficientNet_S3',
        'ResNet18_S2', 'ResNet18_S3'
    ],
    'Balanced_Accuracy': [0.4998, 0.6183, 0.6553, 0.5844, 0.7330],
    'Macro_F1':          [0.3634, 0.4614, 0.4862, 0.4232, 0.6010],
    'Macro_AUC':         [0.8775, 0.9122, 0.9242, 0.8999, 0.9451]
}

results_df = pd.DataFrame(test_results)
os.makedirs('results', exist_ok=True)
results_df.to_csv('results/final_test_results.csv', index=False)
print(" Saved to results/final_test_results.csv")
print(results_df.to_string(index=False))