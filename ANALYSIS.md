# HydroLoom AI: Unsupervised Learning Analysis

## Executive Summary
This document provides a cohesive technical report on the findings from the Service B (Unsupervised Learning) pipeline of the HydroLoom AI project. By applying clustering techniques to engineered behavioral features, we have successfully segmented consumers into four distinct archetypes. This segmentation enables targeted interventions and provides a critical categorical feature for downstream predictive and prescriptive modeling phases.

## Methodology
The unsupervised pipeline utilizes K-Means clustering to discover latent behavioral patterns in consumer water usage data. 

1. **Feature Engineering**: After rigorous analysis and feature selection, four key metrics survived as the foundational inputs for our model:
   - `log_per_capita_usage`: Captures baseline volumetric consumption (log-transformed for normalcy).
   - `dry_day_spike_factor`: Measures responsiveness (spikes in usage) during consecutive dry days.
   - `efficiency_penalty_ratio`: Quantifies how often a consumer operates outside of optimal appliance efficiency windows.
   - `landscape_demand_index`: Represents the proportion of usage likely dedicated to outdoor/landscape needs.

2. **Clustering Algorithm**: We applied K-Means clustering. The optimal number of clusters was determined to be **$k=4$** based on historical Silhouette score evaluations.

3. **Regional Training**: Separate models were trained for the North and South Hemispheres to account for distinct seasonal and climatic variations.

## Archetype Profiles

Based on the latest K-Means centroid logic and visual distributions in the generated box plots, we have identified the following four behavioral clusters (Clusters 0-3 depending on initialization, mapped below by feature dominance):

### 1. Conservationists (Low Volume)
* **Profile:** This cluster is anchored by the absolute minimum `log_per_capita_usage`.
* **Real-World Interpretation:** Highly efficient households with minimal waste and strict adherence to conservation practices. These users likely have modern, water-efficient appliances and do not engage in excessive outdoor watering.

### 2. Heavy Users (High Volume)
* **Profile:** Defined by the maximum `log_per_capita_usage`. They may also exhibit high `dry_day_spike_factor` and `efficiency_penalty_ratio` metrics.
* **Real-World Interpretation:** "Weather-Reactive Guzzlers." These are the system's largest consumers, heavily driving peak demand. Their usage is likely inelastic to minor price changes but highly reactive to weather patterns.

### 3. Outdoor/Landscape Heavy
* **Profile:** Shows moderate overall usage but stands out with a significantly higher `landscape_demand_index` compared to standard users.
* **Real-World Interpretation:** Households with significant outdoor footprint (large gardens, pools, or agricultural components). Their usage is highly seasonal and tied to irrigation needs rather than indoor appliance use.

### 4. Standard Average Consumers
* **Profile:** Characterized by moderate `log_per_capita_usage` and lower `landscape_demand_index`.
* **Real-World Interpretation:** The typical residential baseline. They exhibit average indoor water use (showers, laundry, cooking) with minimal outdoor/landscape inflation.

### Visual Distributions
*Note: Depending on the hemisphere, the relative distances between clusters might vary slightly due to climatic context.*

**Further Exploration:** The images below highlight the primary cluster formations. **It is highly encouraged to explore the numerous other findings available in the `figures/` directory**, which contains extensive visual data for both raw distributions (`figures/raw/`) and engineered features (`figures/processed/`) across both the North and South Hemispheres.

![North Archetype Profiles](figures/processed/North%20Archetype%20Profiles.png)
![South Archetype Profiles](figures/processed/South%20Archetype%20Profiles.png)

![North Cluster Analysis](figures/processed/Cluster%20Analysis%20(North%20Hemisphere).png)
![South Cluster Analysis](figures/processed/Cluster%20Analysis%20(South%20Hemisphere).png)

## The "So What?"

Identifying these archetypes is not just an analytical exercise; it is a foundational step for the entire HydroLoom AI architecture. 

* **Service A (Supervised XGBoost Predictor):** The assigned cluster ID (0-3) will be ingested as a high-value, engineered categorical feature. Knowing whether a household is a "Heavy User" versus a "Conservationist" dramatically reduces the variance in forecasting their future demand, enabling the XGBoost model to specialize its tree splits based on fundamental behavior rather than just raw volume.

* **Service C (Reinforcement Learning):** In our prescriptive phase, the RL agent will use these archetypes to tailor its policy actions. For example, the agent might learn that sending "drought warnings" works well for the *Outdoor/Landscape Heavy* cluster, while "efficiency appliance rebates" yield the highest ROI when targeted at *Standard Average Consumers* with high penalty ratios. 

Ultimately, this unsupervised segmentation bridges the gap between raw telemetry and actionable, personalized water management policies.
