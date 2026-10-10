# Machine Learning: Glass Classification

This was my **first machine learning project**, built for my university Artificial Intelligence course. It compares three classic classifiers on the [Glass Identification dataset](https://archive.ics.uci.edu/dataset/42/glass+identification) (214 samples, 9 chemical features, 6 glass types).

| Notebook | Model |
| --- | --- |
| `KNN.ipynb` | k-Nearest Neighbors, comparing k = 1 to 10 |
| `Support_Vector_Machine.ipynb` | Support Vector Machine (linear and RBF kernels) |
| `Decision_Tree.ipynb` | Decision Tree, with a Graphviz visualization of the tree |

## Looking back

I'm keeping this project as it was to show where I started. Since then I've learned that it has a few mistakes:

- **Scaling bug in the SVM notebook.** The model is trained on unscaled data but evaluated on scaled data, which is why its precision and recall come out very low. Features should be scaled the same way for training and testing, ideally inside a scikit-learn `Pipeline`.
- **Results change on every run.** `train_test_split` is called without `random_state`, so each run uses a different split.
- **Unreliable evaluation for a small dataset.** With only 214 samples and very few examples in some classes, a single train/test split is noisy. Stratified k-fold cross-validation would give more reliable numbers.

For my later work with more careful evaluation, see [phishing-website-detection-tool](https://gitlab.com/fireball3473/phishing-website-detection-tool) and [brain-tumor-detector](https://gitlab.com/fireball3473/brain-tumor-detector).

## Running

```bash
pip install pandas scikit-learn matplotlib graphviz
jupyter notebook
```
