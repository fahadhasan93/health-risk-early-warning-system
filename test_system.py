#!/usr/bin/env python3
"""
Test Script for Health Risk Early Warning System (HREWS)
This script tests the core functionality of the system.
"""

import os
import sys
import pandas as pd
import numpy as np
from hrews_model import HealthRiskPredictor

def test_data_loading():
    """Test if the dataset can be loaded correctly"""
    print("🧪 Testing data loading...")
    
    try:
        data = pd.read_csv('Health_Risk_Dataset.csv')
        print(f"✅ Dataset loaded successfully: {data.shape}")
        
        # Check expected columns
        expected_columns = [
            'Patient_ID', 'Respiratory_Rate', 'Oxygen_Saturation', 'O2_Scale',
            'Systolic_BP', 'Heart_Rate', 'Temperature', 'Consciousness',
            'On_Oxygen', 'Risk_Level'
        ]
        
        missing_columns = set(expected_columns) - set(data.columns)
        if missing_columns:
            print(f"❌ Missing columns: {missing_columns}")
            return False
        
        print("✅ All expected columns present")
        return True
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return False

def test_model_initialization():
    """Test if the model can be initialized"""
    print("\n🧪 Testing model initialization...")
    
    try:
        predictor = HealthRiskPredictor()
        print("✅ HealthRiskPredictor initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Error initializing model: {e}")
        return False

def test_data_preprocessing():
    """Test data preprocessing functionality"""
    print("\n🧪 Testing data preprocessing...")
    
    try:
        predictor = HealthRiskPredictor()
        data = predictor.load_data('Health_Risk_Dataset.csv')
        
        # Test preprocessing
        X_train, X_test, y_train, y_test = predictor.preprocess_data()
        
        print(f"✅ Data preprocessing successful:")
        print(f"  Training set: {X_train.shape}")
        print(f"  Test set: {X_test.shape}")
        print(f"  Features: {len(predictor.feature_names)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in data preprocessing: {e}")
        return False

def test_model_training():
    """Test model training functionality"""
    print("\n🧪 Testing model training...")
    
    try:
        predictor = HealthRiskPredictor()
        data = predictor.load_data('Health_Risk_Dataset.csv')
        predictor.preprocess_data()
        
        # Train models
        results = predictor.train_models()
        
        print(f"✅ Model training successful:")
        print(f"  Models trained: {len(results)}")
        print(f"  Best model: {type(predictor.best_model).__name__}")
        
        # Check if best model is selected
        if predictor.best_model is not None:
            print("✅ Best model selected successfully")
        else:
            print("❌ No best model selected")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error in model training: {e}")
        return False

def test_prediction():
    """Test prediction functionality"""
    print("\n🧪 Testing prediction functionality...")
    
    try:
        predictor = HealthRiskPredictor()
        data = predictor.load_data('Health_Risk_Dataset.csv')
        predictor.preprocess_data()
        predictor.train_models()
        
        # Create sample patient data
        sample_patient = {
            'Respiratory_Rate': 20,
            'Oxygen_Saturation': 95,
            'O2_Scale': 1,
            'Systolic_BP': 120,
            'Heart_Rate': 80,
            'Temperature': 37.0,
            'Consciousness': 'A',
            'On_Oxygen': 0
        }
        
        # Make prediction
        risk_result = predictor.predict_risk(sample_patient)
        
        if risk_result:
            print("✅ Prediction successful:")
            print(f"  Predicted risk: {risk_result['predicted_risk']}")
            print(f"  Probabilities: {risk_result['probabilities']}")
            return True
        else:
            print("❌ Prediction failed")
            return False
        
    except Exception as e:
        print(f"❌ Error in prediction: {e}")
        return False

def test_feature_importance():
    """Test feature importance functionality"""
    print("\n🧪 Testing feature importance...")
    
    try:
        predictor = HealthRiskPredictor()
        data = predictor.load_data('Health_Risk_Dataset.csv')
        predictor.preprocess_data()
        predictor.train_models()
        
        # Get feature importance
        importance = predictor.get_feature_importance()
        
        if importance:
            print("✅ Feature importance retrieved successfully:")
            print(f"  Top 5 features:")
            for i, (feature, imp) in enumerate(list(importance.items())[:5]):
                print(f"    {i+1}. {feature}: {imp:.4f}")
            return True
        else:
            print("❌ Feature importance not available")
            return False
        
    except Exception as e:
        print(f"❌ Error in feature importance: {e}")
        return False

def test_model_saving_loading():
    """Test model saving and loading functionality"""
    print("\n🧪 Testing model saving and loading...")
    
    try:
        # Train and save model
        predictor1 = HealthRiskPredictor()
        data = predictor1.load_data('Health_Risk_Dataset.csv')
        predictor1.preprocess_data()
        predictor1.train_models()
        
        # Save model
        predictor1.save_model('test_model.pkl')
        print("✅ Model saved successfully")
        
        # Load model in new predictor
        predictor2 = HealthRiskPredictor()
        predictor2.load_model('test_model.pkl')
        print("✅ Model loaded successfully")
        
        # Test prediction with loaded model
        sample_patient = {
            'Respiratory_Rate': 25,
            'Oxygen_Saturation': 90,
            'O2_Scale': 2,
            'Systolic_BP': 110,
            'Heart_Rate': 100,
            'Temperature': 38.5,
            'Consciousness': 'A',
            'On_Oxygen': 1
        }
        
        risk_result = predictor2.predict_risk(sample_patient)
        if risk_result:
            print("✅ Loaded model prediction successful")
        else:
            print("❌ Loaded model prediction failed")
            return False
        
        # Clean up test file
        if os.path.exists('test_model.pkl'):
            os.remove('test_model.pkl')
            print("✅ Test model file cleaned up")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in model saving/loading: {e}")
        return False

def run_all_tests():
    """Run all tests and report results"""
    print("🚑 HEALTH RISK EARLY WARNING SYSTEM - SYSTEM TESTING")
    print("="*60)
    
    tests = [
        ("Data Loading", test_data_loading),
        ("Model Initialization", test_model_initialization),
        ("Data Preprocessing", test_data_preprocessing),
        ("Model Training", test_model_training),
        ("Prediction", test_prediction),
        ("Feature Importance", test_feature_importance),
        ("Model Save/Load", test_model_saving_loading)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Report results
    print("\n" + "="*60)
    print("📊 TEST RESULTS SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return False

def main():
    """Main function to run tests"""
    success = run_all_tests()
    
    if success:
        print("\n🚀 Next steps:")
        print("1. Run 'python hrews_model.py' to train the full model")
        print("2. Run 'streamlit run app.py' to launch the web application")
        print("3. Run 'python explore_data.py' to analyze the dataset")
    else:
        print("\n❌ System testing failed. Please fix the issues before proceeding.")
        sys.exit(1)

if __name__ == "__main__":
    main()
