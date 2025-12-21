#!/usr/bin/env python3
from hrews_model import HealthRiskPredictor
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
import json
import os
import joblib

def main():
    p = HealthRiskPredictor()
    print('Loading data...')
    p.load_data('Health_Risk_Dataset.csv')
    print('Preprocessing...')
    p.preprocess_data()

    print('Training models (this may take a short while)...')
    results = p.train_models()

    print('\nModel performance summary:')
    model_metrics = {}
    for name, info in results.items():
        model = info['model']
        accuracy = info.get('accuracy', None)
        preds = info.get('predictions', None)
        print('\n' + '='*60)
        print(f'Model: {name}')
        if accuracy is not None:
            print(f'  Accuracy: {accuracy*100:.2f}%')
        if preds is not None:
            y_true = p.y_test
            y_pred = preds
            print('\n  Classification Report:')
            print(classification_report(y_true, y_pred, target_names=p.label_encoder.classes_))
            cm = confusion_matrix(y_true, y_pred)
            cm_df = pd.DataFrame(cm, index=p.label_encoder.classes_, columns=p.label_encoder.classes_)
            print('\n  Confusion Matrix:')
            print(cm_df.to_string())
        else:
            print('  No prediction data available for this model.')

        # save metrics per model
        prf = precision_recall_fscore_support(y_true, y_pred, average='weighted') if preds is not None else (None, None, None, None)
        model_metrics[name] = {
            'accuracy': float(accuracy) if accuracy is not None else None,
            'precision': float(prf[0]) if prf[0] is not None else None,
            'recall': float(prf[1]) if prf[1] is not None else None,
            'f1': float(prf[2]) if prf[2] is not None else None,
            'confusion_matrix': cm.tolist() if preds is not None else None
        }

    print('\nDone.')

    # determine best model by highest F1 (from collected metrics)
    best_model_name = None
    best_f1 = -1.0
    for name, vals in model_metrics.items():
        f1 = vals.get('f1')
        if f1 is not None and f1 > best_f1:
            best_f1 = f1
            best_model_name = name

    # fallback: try matching predictor.best_model instance to results
    if best_model_name is None and hasattr(p, 'best_model') and p.best_model is not None:
        for name, info in results.items():
            if info.get('model') is p.best_model:
                best_model_name = name
                break

    # final fallback: first key
    if best_model_name is None and len(results) > 0:
        best_model_name = list(results.keys())[0]

    # persist trained predictor so the Streamlit app can load it directly
    try:
        joblib.dump(p, 'hrews_model.pkl')
        print("Saved trained predictor to 'hrews_model.pkl'")
    except Exception as e:
        print(f"⚠️  Warning: Failed to save predictor pickle: {e}")

    # persist metrics to JSON for UI consumption
    out_path = 'model_metrics.json'
    with open(out_path, 'w') as fh:
        json.dump({'models': model_metrics, 'best_model': best_model_name}, fh, indent=2)
    print(f'Model metrics written to {out_path} (best_model: {best_model_name})')

if __name__ == '__main__':
    main()
