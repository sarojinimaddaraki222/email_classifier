import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("="*60)
print("MODEL COMPARISON")
print("="*60)

# Load all results
results = []

# Logistic Regression
try:
    lr_results = pd.read_csv("../results/logistic_regression_results.csv")
    results.append(lr_results)
    print("✅ Loaded Logistic Regression results")
except:
    print("❌ Logistic Regression results not found")

# Naive Bayes
try:
    nb_results = pd.read_csv("../results/naive_bayes_results.csv")
    results.append(nb_results)
    print("✅ Loaded Naive Bayes results")
except:
    print("❌ Naive Bayes results not found")

# Transformer
try:
    bert_results = pd.read_csv("../results/transformer_results.csv")
    results.append(bert_results)
    print("✅ Loaded Transformer results")
except:
    print("⚠️  Transformer results not found (run bert_classifier.py)")

if results:
    # Combine all results
    comparison_df = pd.concat(results, ignore_index=True)
    
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    print("\n", comparison_df.to_string(index=False))
    
    # Save comparison
    comparison_df.to_csv("../results/model_comparison.csv", index=False)
    print("\n✅ Comparison saved: ../results/model_comparison.csv")
    
    # Visualize comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Category accuracy comparison
    axes[0].bar(comparison_df['Model'], comparison_df['Category Accuracy'], color='steelblue')
    axes[0].set_title('Category Classification Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_ylim(0, 1)
    axes[0].grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(comparison_df['Category Accuracy']):
        axes[0].text(i, v + 0.02, f"{v:.3f}", ha='center', fontweight='bold')
    
    # Urgency accuracy comparison
    axes[1].bar(comparison_df['Model'], comparison_df['Urgency Accuracy'], color='coral')
    axes[1].set_title('Urgency Classification Accuracy', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_ylim(0, 1)
    axes[1].grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(comparison_df['Urgency Accuracy']):
        axes[1].text(i, v + 0.02, f"{v:.3f}", ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("../results/model_comparison.png", dpi=300, bbox_inches='tight')
    print("✅ Visualization saved: ../results/model_comparison.png")
    plt.show()
    
    # Find best models
    best_category_idx = comparison_df['Category Accuracy'].idxmax()
    best_urgency_idx = comparison_df['Urgency Accuracy'].idxmax()
    
    print("\n" + "="*60)
    print("BEST MODELS")
    print("="*60)
    print(f"\n🏆 Best for Category: {comparison_df.loc[best_category_idx, 'Model']}")
    print(f"   Accuracy: {comparison_df.loc[best_category_idx, 'Category Accuracy']:.4f}")
    print(f"\n🏆 Best for Urgency: {comparison_df.loc[best_urgency_idx, 'Model']}")
    print(f"   Accuracy: {comparison_df.loc[best_urgency_idx, 'Urgency Accuracy']:.4f}")

else:
    print("\n❌ No results found! Please run model training scripts first.")
    print("   Run: python logistic_regression.py")
    print("   Run: python naive_bayes.py")
    print("   Run: python bert_classifier.py")

print("\n" + "="*60)