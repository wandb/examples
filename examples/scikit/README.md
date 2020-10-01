## 📍 Scikit
You can use wandb to visualize and compare your `scikit-learn` models' performance with just a few lines of code.
```python
import wandb
wandb.init(project="visualize-sklearn")

# Model training here

# Log classifier visualizations
wandb.sklearn.plot_classifier(clf, X_train, X_test, y_train, y_test, y_pred, y_probas, labels, model_name='SVC', feature_names=None)

# Log regression visualizations
wandb.sklearn.plot_regressor(reg, X_train, X_test, y_train, y_test,  model_name='Ridge')

# Log clustering visualizations
wandb.sklearn.plot_clusterer(kmeans, X_train, cluster_labels, labels=None, model_name='KMeans')
```

- **[Try in a colab →](https://colab.research.google.com/drive/1dxWV5uulLOQvMoBBaJy2dZ3ZONr4Mqlo?usp=sharing)**
- [Docs](https://docs.wandb.com/library/integrations/scikit)
