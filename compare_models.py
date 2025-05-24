import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import time

print("="*80)
print("MODEL COMPARISON FOR FLIPKART CUSTOMER SUPPORT DATA")
print("="*80)

# Function to run a model script and capture its accuracy
def run_model(script_name):
    print(f"\nRunning {script_name}...")
    start_time = time.time()
    
    # Run the script and capture output
    process = subprocess.Popen(['python', script_name], 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE,
                              universal_newlines=True)
    
    stdout, stderr = process.communicate()
    
    # Extract accuracy from output
    accuracy = None
    for line in stdout.split('\n'):
        if 'Accuracy:' in line:
            try:
                accuracy = float(line.split('Accuracy:')[1].strip())
                break
            except:
                pass
    
    end_time = time.time()
    runtime = end_time - start_time
    
    return {
        'model': script_name.replace('.py', '').replace('_', ' ').title(),
        'accuracy': accuracy,
        'runtime': runtime,
        'stdout': stdout,
        'stderr': stderr
    }

# List of model scripts to run
model_scripts = [
    'logistic_regression_model.py',
    'random_forest_model.py',
    'gradient_boosting_model.py'
]

# Run each model and collect results
results = []
for script in model_scripts:
    result = run_model(script)
    results.append(result)
    print(f"Completed {script} with accuracy: {result['accuracy']}")

# Create results dataframe
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('accuracy', ascending=False)

print("\n" + "="*80)
print("MODEL COMPARISON RESULTS")
print("="*80)
print("\nModel Performance Ranking:")
for i, row in results_df.iterrows():
    print(f"{i+1}. {row['model']}: Accuracy = {row['accuracy']:.4f}, Runtime = {row['runtime']:.2f} seconds")

# Create comparison chart
plt.figure(figsize=(10, 6))
bars = plt.barh(results_df['model'], results_df['accuracy'], color=['#1f77b4', '#ff7f0e', '#2ca02c'])

# Add accuracy values on bars
for i, bar in enumerate(bars):
    plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
             f'{results_df["accuracy"].iloc[i]:.4f}', 
             va='center')

plt.xlabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.xlim(0, 1)
plt.tight_layout()
plt.savefig('model_comparison.png')
print("\nComparison chart saved as 'model_comparison.png'")

# Analyze results
best_model = results_df.iloc[0]['model']
worst_model = results_df.iloc[-1]['model']
acc_diff = results_df.iloc[0]['accuracy'] - results_df.iloc[-1]['accuracy']

print("\nANALYSIS:")
print(f"• The best performing model is {best_model} with an accuracy of {results_df.iloc[0]['accuracy']:.4f}")
print(f"• The difference between the best and worst model is {acc_diff:.4f} ({acc_diff*100:.2f}%)")

if acc_diff < 0.05:
    print("• The performance difference between models is relatively small (<5%)")
    print("  Consider using the simpler model for better interpretability or faster inference.")
else:
    print(f"• {best_model} outperforms other models by a significant margin")
    print("  This model would be recommended for production use.")

print("\nNOTE: For a more robust comparison, consider using cross-validation to ensure")
print("the results are stable across different data splits.")

print("\nModel comparison complete!")
