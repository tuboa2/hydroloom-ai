# 🤝 Open Source Contribution Guidelines

I actively encourage the community to break the models and help me refine the clustering logic!

## How to Contribute
1. **Discover Anomalies**: Run `sandbox.py` and experiment with extreme inputs.
2. **Document Findings**: If you trigger the `🚨 ANOMALY DETECTED! 🚨` alert, note the exact feature inputs and the reported distances.
3. **Submit a Pull Request (PR)**:
   - Create a new branch (e.g., `fix/anomaly-landscape-index`).
   - Propose changes to the scaling logic in `src/features/extract_features.py` or the model hyper-parameters in `src/model/train_clusterer.py` to better account for your discovered edge case.
   - Include the sandbox execution log in your PR description as proof of the anomaly.
