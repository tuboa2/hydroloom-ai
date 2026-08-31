export type Hemisphere = 'north' | 'south';

export type WQIClassification = 'Excellent' | 'Good' | 'Moderate' | 'Poor' | 'Hazardous';

export type ModelFamily = 'ensemble' | 'lightgbm' | 'xgboost' | 'catboost' | 'linear';

export interface WQIPredictionRequest {
  hemisphere: Hemisphere;
  daily_rainfall_mm: number;
  daily_max_temp: number;
  antecedent_moisture_condition: number;
  total_suspended_solids_mg_l: number;
  nutrient_load_index: number;
  consecutive_dry_days: number;
  consumer_demand_cluster: number;
  day_of_year: number;
  include_shap?: boolean;
  model_family?: ModelFamily;
}

export interface WQIPredictionResponse {
  wqi_score: number;
  classification: WQIClassification;
  confidence_interval: [number, number];
  model_used: string;
  hemisphere: string;
  shap_breakdown: Record<string, number>;
  advisories: string[];
}

export interface ClusterPredictionRequest {
  hemisphere: Hemisphere;
  log_per_capita_usage: number;
  dry_day_spike_factor: number;
  efficiency_penalty_ratio: number;
  landscape_demand_index: number;
}

export interface ClusterPredictionResponse {
  cluster_id: number;
  archetype_name: string;
  description: string;
  behavior_scores: Record<string, number>;
  recommendations: string[];
}

export interface SimulationRequest {
  hemisphere: Hemisphere;
  days: number;
  scenario: 'baseline' | 'drought_heatwave' | 'intense_storm' | 'conservation_mandate';
}

export interface SimulationDay {
  day: number;
  rainfall_mm: number;
  max_temp: number;
  consecutive_dry_days: number;
  tss_mg_l: number;
  wqi_score: number;
}

export interface SimulationResponse {
  scenario: string;
  hemisphere: string;
  days: number;
  timeline: SimulationDay[];
  summary: {
    avg_wqi: number;
    min_wqi: number;
    max_wqi: number;
    total_rainfall_mm: number;
    storm_events: number;
  };
}

export interface ModelMetadata {
  leaderboard: Array<{
    model: string;
    family: string;
    rmse: number;
    mae: number;
    r2: number;
    status: string;
  }>;
  top_features: Array<{
    feature: string;
    importance: number;
    family: string;
  }>;
  version: string;
  last_trained: string;
}

export interface FeedbackSubmission {
  user_name?: string;
  user_email?: string;
  category: 'forecast_accuracy' | 'feature_request' | 'scenario_preset' | 'bug_report' | 'general';
  rating: number;
  comment: string;
  scenario_payload?: any;
}

export interface FeedbackItem {
  id: string;
  user_name: string;
  category: string;
  rating: number;
  comment: string;
  scenario_payload?: any;
  received_at: string;
}

export type CloudStatus = 'connected' | 'waking_up' | 'offline_hybrid';
