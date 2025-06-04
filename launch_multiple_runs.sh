#!/bin/bash

for depth in 5 10 15; do
  for n_estimators in 50 100 200; do
    for lr_C in 0.1 1.0 10.0; do
      for knn_k in 3 5 7; do
        docker compose exec app python src/models/train_model.py \
          --input_csv data/movies_cleaned.csv \
          --rf_n_estimators $n_estimators \
          --rf_max_depth $depth \
          --rf_min_samples_split 5 \
          --rf_min_samples_leaf 5 \
          --lr_C $lr_C \
          --knn_n_neighbors $knn_k \
          --test_size 0.3 \
          --random_state 42
      done
    done
  done
done

