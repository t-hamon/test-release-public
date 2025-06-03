#!/bin/bash

for depth in 5 10 15; do
  for n_estimators in 50 100 200; do
    docker exec reco-app python src/models/train_model.py \
      --input_csv data/movies_cleaned.csv \
      --rf_n_estimators $n_estimators \
      --rf_max_depth $depth \
      --rf_min_samples_split 5 \
      --rf_min_samples_leaf 5 \
      --test_size 0.3 \
      --random_state 42
  done
done
