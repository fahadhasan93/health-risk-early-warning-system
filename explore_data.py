#!/usr/bin/env python3
"""
Data Exploration Script for Health Risk Early Warning System (HREWS)
This script provides comprehensive analysis of the patient dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def load_and_explore_data():
    """Load the dataset and perform initial exploration"""
    print("🚑 Loading Health Risk Dataset...")
    
    try:
        data = pd.read_csv('Health_Risk_Dataset.csv')
        print(f"✅ Dataset loaded successfully!")
        print(f"📊 Shape: {data.shape}")
        print(f"👥 Total Patients: {len(data)}")
        print(f"🔍 Features: {len(data.columns)}")
        
        return data
    except FileNotFoundError:
        print("❌ Error: Health_Risk_Dataset.csv not found!")
        return None
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None

def basic_data_info(data):
    """Display basic information about the dataset"""
    print("\n" + "="*60)
    print("📋 BASIC DATASET INFORMATION")
    print("="*60)
    
    # Data types and missing values
    print("\n📊 Data Types:")
    print(data.dtypes)
    
    print("\n🔍 Missing Values:")
    missing_values = data.isnull().sum()
    if missing_values.sum() == 0:
        print("✅ No missing values found!")
    else:
        print(missing_values[missing_values > 0])
    
    # Unique values in categorical columns
    print("\n🏷️ Categorical Features:")
    categorical_cols = data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        unique_vals = data[col].nunique()
        print(f"  {col}: {unique_vals} unique values")
        if unique_vals <= 10:  # Show values if not too many
            print(f"    Values: {sorted(data[col].unique())}")
    
    # Numerical features summary
    print("\n🔢 Numerical Features Summary:")
    numerical_cols = data.select_dtypes(include=[np.number]).columns
    print(data[numerical_cols].describe())

def analyze_target_variable(data):
    """Analyze the target variable (Risk_Level)"""
    print("\n" + "="*60)
    print("🎯 TARGET VARIABLE ANALYSIS")
    print("="*60)
    
    risk_counts = data['Risk_Level'].value_counts()
    risk_percentages = data['Risk_Level'].value_counts(normalize=True) * 100
    
    print("\n📊 Risk Level Distribution:")
    for risk, count in risk_counts.items():
        percentage = risk_percentages[risk]
        print(f"  {risk}: {count} patients ({percentage:.1f}%)")
    
    # Check for class imbalance
    print(f"\n⚖️ Class Balance:")
    min_count = risk_counts.min()
    max_count = risk_counts.max()
    imbalance_ratio = max_count / min_count
    print(f"  Most frequent class: {risk_counts.idxmax()} ({max_count})")
    print(f"  Least frequent class: {risk_counts.idxmin()} ({min_count})")
    print(f"  Imbalance ratio: {imbalance_ratio:.2f}:1")
    
    if imbalance_ratio > 2:
        print("  ⚠️  Note: Significant class imbalance detected")
    else:
        print("  ✅ Classes are relatively balanced")

def analyze_vital_signs(data):
    """Analyze vital signs and their distributions"""
    print("\n" + "="*60)
    print("💓 VITAL SIGNS ANALYSIS")
    print("="*60)
    
    vital_cols = ['Respiratory_Rate', 'Oxygen_Saturation', 'Systolic_BP', 'Heart_Rate', 'Temperature']
    
    print("\n📈 Vital Signs Statistics:")
    vital_stats = data[vital_cols].describe()
    print(vital_stats)
    
    # Check for outliers using IQR method
    print("\n🚨 Outlier Detection (IQR Method):")
    for col in vital_cols:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
        outlier_count = len(outliers)
        outlier_percentage = (outlier_count / len(data)) * 100
        
        print(f"  {col}: {outlier_count} outliers ({outlier_percentage:.1f}%)")
        if outlier_count > 0:
            print(f"    Range: [{data[col].min():.2f}, {data[col].max():.2f}]")
            print(f"    Normal range: [{lower_bound:.2f}, {upper_bound:.2f}]")
    
    # Risk level analysis by vital signs
    print("\n🔍 Risk Level Analysis by Vital Signs:")
    for col in vital_cols:
        print(f"\n  {col}:")
        risk_analysis = data.groupby('Risk_Level')[col].agg(['mean', 'std', 'min', 'max'])
        print(risk_analysis)

def analyze_categorical_features(data):
    """Analyze categorical features"""
    print("\n" + "="*60)
    print("🏷️ CATEGORICAL FEATURES ANALYSIS")
    print("="*60)
    
    # Consciousness analysis
    print("\n🧠 Consciousness Level Analysis:")
    consciousness_counts = data['Consciousness'].value_counts()
    print(consciousness_counts)
    
    # Consciousness vs Risk Level
    print("\n🧠 Consciousness vs Risk Level:")
    consciousness_risk = pd.crosstab(data['Consciousness'], data['Risk_Level'], normalize='index') * 100
    print(consciousness_risk.round(2))
    
    # Oxygen therapy analysis
    print("\n🫁 Oxygen Therapy Analysis:")
    oxygen_counts = data['On_Oxygen'].value_counts()
    print(oxygen_counts)
    
    # Oxygen therapy vs Risk Level
    print("\n🫁 Oxygen Therapy vs Risk Level:")
    oxygen_risk = pd.crosstab(data['On_Oxygen'], data['Risk_Level'], normalize='index') * 100
    print(oxygen_risk.round(2))
    
    # O2 Scale analysis
    print("\n📏 O2 Scale Analysis:")
    o2_scale_counts = data['O2_Scale'].value_counts().sort_index()
    print(o2_scale_counts)
    
    # O2 Scale vs Risk Level
    print("\n📏 O2 Scale vs Risk Level:")
    o2_scale_risk = pd.crosstab(data['O2_Scale'], data['Risk_Level'], normalize='index') * 100
    print(o2_scale_risk.round(2))

def correlation_analysis(data):
    """Perform correlation analysis"""
    print("\n" + "="*60)
    print("🔗 CORRELATION ANALYSIS")
    print("="*60)
    
    # Prepare data for correlation
    corr_data = data.copy()
    
    # Encode categorical variables
    le = LabelEncoder()
    corr_data['Risk_Level_Encoded'] = le.fit_transform(data['Risk_Level'])
    corr_data['Consciousness_Encoded'] = le.fit_transform(data['Consciousness'])
    corr_data['On_Oxygen_Encoded'] = data['On_Oxygen']
    
    # Select numerical columns for correlation
    numerical_cols = ['Respiratory_Rate', 'Oxygen_Saturation', 'O2_Scale', 
                     'Systolic_BP', 'Heart_Rate', 'Temperature', 'On_Oxygen_Encoded',
                     'Consciousness_Encoded', 'Risk_Level_Encoded']
    
    correlation_matrix = corr_data[numerical_cols].corr()
    
    print("\n📊 Correlation with Risk Level:")
    risk_correlations = correlation_matrix['Risk_Level_Encoded'].sort_values(ascending=False)
    for feature, corr in risk_correlations.items():
        if feature != 'Risk_Level_Encoded':
            print(f"  {feature}: {corr:.3f}")
    
    # Strong correlations
    print("\n🔍 Strong Correlations (|r| > 0.3):")
    strong_corr = []
    for i in range(len(numerical_cols)):
        for j in range(i+1, len(numerical_cols)):
            corr_val = correlation_matrix.iloc[i, j]
            if abs(corr_val) > 0.3:
                strong_corr.append((numerical_cols[i], numerical_cols[j], corr_val))
    
    for feat1, feat2, corr_val in strong_corr:
        print(f"  {feat1} ↔ {feat2}: {corr_val:.3f}")

def generate_insights(data):
    """Generate key insights from the data"""
    print("\n" + "="*60)
    print("💡 KEY INSIGHTS & RECOMMENDATIONS")
    print("="*60)
    
    insights = []
    
    # Risk level distribution insights
    risk_counts = data['Risk_Level'].value_counts()
    high_risk_patients = risk_counts.get('High', 0)
    high_risk_percentage = (high_risk_patients / len(data)) * 100
    
    insights.append(f"📊 {high_risk_percentage:.1f}% of patients are classified as High Risk")
    
    # Vital signs insights
    vital_cols = ['Respiratory_Rate', 'Oxygen_Saturation', 'Systolic_BP', 'Heart_Rate', 'Temperature']
    
    for col in vital_cols:
        high_risk_data = data[data['Risk_Level'] == 'High'][col]
        normal_data = data[data['Risk_Level'] == 'Normal'][col]
        
        if len(high_risk_data) > 0 and len(normal_data) > 0:
            high_mean = high_risk_data.mean()
            normal_mean = normal_data.mean()
            diff = high_mean - normal_mean
            
            if abs(diff) > normal_data.std():
                direction = "higher" if diff > 0 else "lower"
                insights.append(f"🔍 High-risk patients have {direction} {col} (diff: {abs(diff):.2f})")
    
    # Oxygen therapy insights
    oxygen_high_risk = data[(data['Risk_Level'] == 'High') & (data['On_Oxygen'] == 1)]
    oxygen_high_risk_pct = (len(oxygen_high_risk) / len(data[data['Risk_Level'] == 'High'])) * 100
    insights.append(f"🫁 {oxygen_high_risk_pct:.1f}% of high-risk patients are on oxygen therapy")
    
    # Consciousness insights
    consciousness_high_risk = data[data['Risk_Level'] == 'High']['Consciousness'].value_counts()
    if len(consciousness_high_risk) > 0:
        most_common_consciousness = consciousness_high_risk.index[0]
        insights.append(f"🧠 Most common consciousness level in high-risk patients: {most_common_consciousness}")
    
    # Print insights
    for i, insight in enumerate(insights, 1):
        print(f"{i}. {insight}")
    
    # Recommendations
    print("\n🎯 RECOMMENDATIONS:")
    print("1. Focus monitoring on patients with abnormal vital signs")
    print("2. Pay special attention to patients on oxygen therapy")
    print("3. Consider consciousness level as a key risk indicator")
    print("4. Implement early warning thresholds for critical vital signs")
    print("5. Regular monitoring for patients with medium risk levels")

def create_visualizations(data):
    """Create and save visualizations"""
    print("\n" + "="*60)
    print("📊 CREATING VISUALIZATIONS")
    print("="*60)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Health Risk Dataset Analysis', fontsize=16, fontweight='bold')
    
    # 1. Risk Level Distribution
    risk_counts = data['Risk_Level'].value_counts()
    axes[0, 0].pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%', startangle=90)
    axes[0, 0].set_title('Risk Level Distribution')
    
    # 2. Vital Signs Distribution
    vital_cols = ['Respiratory_Rate', 'Oxygen_Saturation', 'Systolic_BP', 'Heart_Rate', 'Temperature']
    for i, col in enumerate(vital_cols):
        row = i // 3
        col_pos = i % 3
        axes[row, col_pos].hist(data[col], bins=20, alpha=0.7, edgecolor='black')
        axes[row, col_pos].set_title(f'{col} Distribution')
        axes[row, col_pos].set_xlabel(col)
        axes[row, col_pos].set_ylabel('Frequency')
    
    # Remove empty subplot
    axes[1, 2].remove()
    
    plt.tight_layout()
    plt.savefig('data_analysis_plots.png', dpi=300, bbox_inches='tight')
    print("✅ Visualizations saved as 'data_analysis_plots.png'")
    
    # Create correlation heatmap
    plt.figure(figsize=(10, 8))
    
    # Prepare correlation data
    corr_data = data.copy()
    le = LabelEncoder()
    corr_data['Risk_Level_Encoded'] = le.fit_transform(data['Risk_Level'])
    corr_data['Consciousness_Encoded'] = le.fit_transform(data['Consciousness'])
    
    numerical_cols = ['Respiratory_Rate', 'Oxygen_Saturation', 'O2_Scale', 
                     'Systolic_BP', 'Heart_Rate', 'Temperature', 'On_Oxygen',
                     'Consciousness_Encoded', 'Risk_Level_Encoded']
    
    correlation_matrix = corr_data[numerical_cols].corr()
    
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5)
    plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("✅ Correlation heatmap saved as 'correlation_heatmap.png'")

def main():
    """Main function to run the data exploration"""
    print("🚑 HEALTH RISK EARLY WARNING SYSTEM - DATA EXPLORATION")
    print("="*60)
    
    # Load data
    data = load_and_explore_data()
    if data is None:
        return
    
    # Perform analysis
    basic_data_info(data)
    analyze_target_variable(data)
    analyze_vital_signs(data)
    analyze_categorical_features(data)
    correlation_analysis(data)
    generate_insights(data)
    
    # Create visualizations
    try:
        create_visualizations(data)
    except Exception as e:
        print(f"⚠️  Warning: Could not create visualizations: {e}")
    
    print("\n" + "="*60)
    print("✅ DATA EXPLORATION COMPLETED!")
    print("="*60)
    print("\n📁 Generated files:")
    print("  - data_analysis_plots.png")
    print("  - correlation_heatmap.png")
    print("\n🚀 Next steps:")
    print("  1. Run 'python hrews_model.py' to train the ML model")
    print("  2. Run 'streamlit run app.py' to launch the web application")

if __name__ == "__main__":
    main()
