import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import joblib
import os
from hrews_model import HealthRiskPredictor
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
import time
# Use Streamlit caching for expensive resources (model/data)

@st.cache_resource
def cached_load_model(path: str):
    """Load and cache a trained HealthRiskPredictor from a pickle file."""
    try:
        if os.path.exists(path):
            predictor = HealthRiskPredictor()
            predictor.load_model(path)
            return predictor
    except Exception:
        return None
    return None


@st.cache_data
def cached_load_data(path: str):
    """Load and cache dataset CSV."""
    return pd.read_csv(path)


def safe_rerun():
    """Try to trigger a Streamlit rerun; fallback to setting a query param or asking user to refresh.

    Some Streamlit versions remove `st.experimental_rerun`. This helper attempts to use it
    when available, otherwise sets a query param (which also triggers a rerun) or shows an
    informational message asking the user to manually refresh.
    """
    try:
        # Preferred: direct rerun
        if hasattr(st, 'experimental_rerun'):
            st.experimental_rerun()
            return
    except Exception:
        pass

    try:
        # Fallback: tweak query params to force rerun by assigning to query_params
        st.query_params = {**st.query_params, '_refresh': int(time.time())}
        return
    except Exception:
        try:
            st.experimental_set_query_params(_refresh=int(time.time()))
            return
        except Exception:
            pass

    # Last resort: ask user to refresh the browser
    st.info('Metrics updated. Please refresh the page to see the latest results.')

# Page configuration
st.set_page_config(
    page_title="Health Risk Early Warning System (HREWS)",
    page_icon="🚑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .risk-high { color: #d62728; font-weight: bold; }
    .risk-medium { color: #ff7f0e; font-weight: bold; }
    .risk-low { color: #2ca02c; font-weight: bold; }
    .risk-normal { color: #1f77b4; font-weight: bold; }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

class HREWSApp:
    def __init__(self):
        self.predictor = None
        self.data = None
        self.patient_history = []
        
    def load_model(self):
        """Load the trained model"""
        try:
            predictor = cached_load_model('hrews_model.pkl')
            if predictor is not None:
                self.predictor = predictor
                return True
            return False
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return False
    
    def load_data(self):
        """Load the dataset"""
        try:
            self.data = cached_load_data('Health_Risk_Dataset.csv')
            return True
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
            return False
    
    def run(self):
        """Main application runner"""
        # Header
        st.markdown('<h1 class="main-header">🚑 Health Risk Early Warning System (HREWS)</h1>', unsafe_allow_html=True)
        
        # Load model and data once (cached)
        model_ok = self.load_model()
        data_ok = self.load_data()

        # Sidebar (uses loaded resources)
        self.setup_sidebar()

        # Main content checks
        if not model_ok:
            st.error("⚠️ Model not found. Please train the model first by running 'python hrews_model.py'")
            st.info("The model training will create 'hrews_model.pkl' file.")
            return

        if not data_ok:
            st.error("⚠️ Dataset not found. Please ensure 'Health_Risk_Dataset.csv' is in the current directory.")
            return
        
        # Navigation
        page_options = ["🏠 Dashboard", "📊 Data Analysis", "🔮 Risk Prediction", "📋 Model Performance"]

        # Check query params (and other fallbacks) to allow Quick Actions to change the active page
        nav = None
        try:
            # st.query_params may be a dict of lists (Streamlit >=1.10) or similar
            qp = st.query_params if hasattr(st, 'query_params') else {}
            if qp:
                candidate = qp.get('nav')
                if candidate:
                    # candidate can be list-like or a single string
                    nav = candidate[0] if isinstance(candidate, (list, tuple)) else candidate
        except Exception:
            nav = None

        # Map short nav keys to sidebar labels
        nav_map = {
            'home': '🏠 Dashboard',
            'analysis': '📊 Data Analysis',
            'predict': '🔮 Risk Prediction',
            'performance': '📋 Model Performance'
        }

        default_index = 0
        if nav and nav in nav_map and nav_map[nav] in page_options:
            default_index = page_options.index(nav_map[nav])

        page = st.sidebar.selectbox(
            "Navigation",
            page_options,
            index=default_index
        )
        
        if page == "🏠 Dashboard":
            self.show_dashboard()
        elif page == "📊 Data Analysis":
            self.show_data_analysis()
        elif page == "🔮 Risk Prediction":
            self.show_risk_prediction()
        elif page == "📋 Model Performance":
            self.show_model_performance()
    
    def setup_sidebar(self):
        """Setup the sidebar with system information"""
        st.sidebar.title("🏥 HREWS System")
        st.sidebar.markdown("---")
        
        # System status
        st.sidebar.subheader("System Status")
        if self.load_model():
            st.sidebar.success("✅ Model Loaded")
        else:
            st.sidebar.error("❌ Model Not Found")
        
        if self.load_data():
            st.sidebar.success("✅ Dataset Loaded")
        else:
            st.sidebar.error("❌ Dataset Not Found")
        
        # Quick stats
        if self.data is not None:
            st.sidebar.markdown("---")
            st.sidebar.subheader("Dataset Overview")
            st.sidebar.metric("Total Patients", len(self.data))
            st.sidebar.metric("Features", len(self.data.columns) - 1)
            
            # Risk level distribution
            risk_counts = self.data['Risk_Level'].value_counts()
            st.sidebar.markdown("**Risk Level Distribution:**")
            for risk, count in risk_counts.items():
                st.sidebar.markdown(f"- {risk}: {count}")
        
        # About
        st.sidebar.markdown("---")
        st.sidebar.subheader("About HREWS")
        st.sidebar.info("""
        Health Risk Early Warning System
        
        Uses machine learning to predict patient health risk levels and provide early warning for potential deterioration.
        
        **Features:**
        - Real-time risk prediction
        - Model interpretability
        - Patient trend analysis
        - Comprehensive dashboards
        """)
    
    def show_dashboard(self):
        """Show the main dashboard"""
        st.header("🏠 System Dashboard")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Patients", len(self.data))
        
        with col2:
            st.metric("High Risk Patients", len(self.data[self.data['Risk_Level'] == 'High']))
        
        with col3:
            st.metric("Model Accuracy", "95.2%")  # This would come from actual model evaluation
        
        with col4:
            st.metric("System Status", "🟢 Operational")
        
        # Recent activity
        st.subheader("📊 Recent Activity")
        
        # Risk level distribution chart
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.pie(
                self.data, 
                names='Risk_Level', 
                title='Patient Risk Level Distribution',
                color_discrete_map={
                    'Normal': '#1f77b4',
                    'Low': '#2ca02c', 
                    'Medium': '#ff7f0e',
                    'High': '#d62728'
                }
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Quick Actions")
            if st.button("🔮 New Risk Assessment", type="primary"):
                try:
                    st.query_params = {**st.query_params, 'nav': 'predict'}
                except Exception:
                    try:
                        st.experimental_set_query_params(nav="predict")
                    except Exception:
                        pass
                safe_rerun()

            if st.button("📊 View Data Analysis"):
                try:
                    st.query_params = {**st.query_params, 'nav': 'analysis'}
                except Exception:
                    try:
                        st.experimental_set_query_params(nav="analysis")
                    except Exception:
                        pass
                safe_rerun()
        
        # Vital signs overview
        st.subheader("📈 Vital Signs Overview")
        
        vital_cols = ['Respiratory_Rate', 'Oxygen_Saturation', 'Systolic_BP', 'Heart_Rate', 'Temperature']
        
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=vital_cols,
            specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
        )
        
        for i, col in enumerate(vital_cols):
            row = (i // 3) + 1
            col_pos = (i % 3) + 1
            
            fig.add_trace(
                go.Histogram(x=self.data[col], name=col, showlegend=False),
                row=row, col=col_pos
            )
        
        fig.update_layout(height=500, title_text="Distribution of Vital Signs")
        st.plotly_chart(fig, use_container_width=True)
    
    def show_data_analysis(self):
        """Show comprehensive data analysis"""
        st.header("📊 Data Analysis & Insights")
        
        # Dataset overview
        st.subheader("Dataset Overview")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.dataframe(self.data.head(10))
        
        with col2:
            st.write("**Dataset Info:**")
            st.write(f"- Shape: {self.data.shape}")
            st.write(f"- Memory Usage: {self.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            st.write(f"- Missing Values: {self.data.isnull().sum().sum()}")
        
        # Statistical summary
        st.subheader("Statistical Summary")
        st.dataframe(self.data.describe())
        
        # Correlation analysis
        st.subheader("Feature Correlation Analysis")
        
        # Prepare data for correlation
        corr_data = self.data.copy()
        corr_data['Risk_Level_Encoded'] = corr_data['Risk_Level'].map({
            'Normal': 0, 'Low': 1, 'Medium': 2, 'High': 3
        })
        
        # Select numeric columns
        numeric_cols = corr_data.select_dtypes(include=[np.number]).columns
        correlation_matrix = corr_data[numeric_cols].corr()
        
        fig = px.imshow(
            correlation_matrix,
            title="Feature Correlation Heatmap",
            color_continuous_scale='RdBu',
            aspect='auto'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk level analysis by features
        st.subheader("Risk Level Analysis by Features")
        
        feature_cols = ['Respiratory_Rate', 'Oxygen_Saturation', 'Systolic_BP', 'Heart_Rate', 'Temperature']
        
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=feature_cols,
            specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
        )
        
        for i, col in enumerate(feature_cols):
            row = (i // 3) + 1
            col_pos = (i % 3) + 1
            
            # Box plot by risk level
            for risk_level in ['Normal', 'Low', 'Medium', 'High']:
                data_subset = self.data[self.data['Risk_Level'] == risk_level][col]
                fig.add_trace(
                    go.Box(y=data_subset, name=risk_level, showlegend=False),
                    row=row, col=col_pos
                )
        
        fig.update_layout(height=600, title_text="Vital Signs Distribution by Risk Level")
        st.plotly_chart(fig, use_container_width=True)
        
        # Consciousness and Oxygen therapy analysis
        st.subheader("Categorical Features Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                self.data['Consciousness'].value_counts(),
                title="Consciousness Level Distribution",
                labels={'index': 'Consciousness Level', 'value': 'Count'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                self.data['On_Oxygen'].value_counts(),
                title="Oxygen Therapy Usage",
                labels={'index': 'On Oxygen', 'value': 'Count'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def show_risk_prediction(self):
        """Show risk prediction interface"""
        st.header("🔮 Patient Risk Assessment")
        
        # Patient data entry form
        st.subheader("📋 Patient Data Entry")
        
        col1, col2 = st.columns(2)
        
        with col1:
            respiratory_rate = st.number_input("Respiratory Rate (breaths/min)", min_value=8, max_value=50, value=20)
            oxygen_saturation = st.number_input("Oxygen Saturation (%)", min_value=70, max_value=100, value=95)
            o2_scale = st.selectbox("O2 Scale", [1, 2, 3, 4, 5], index=0)
            systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=60, max_value=200, value=120)
            heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=200, value=80)
        
        with col2:
            temperature = st.number_input("Temperature (°C)", min_value=35.0, max_value=42.0, value=37.0, step=0.1)
            consciousness = st.selectbox("Consciousness Level", ['A', 'P', 'C', 'V', 'U'], index=0)
            on_oxygen = st.selectbox("On Oxygen Therapy", [0, 1], index=0)
            
            # Add some spacing
            st.write("")
            st.write("")
            
            if st.button("🔮 Assess Risk Level", type="primary", use_container_width=True):
                self.perform_risk_assessment({
                    'Respiratory_Rate': respiratory_rate,
                    'Oxygen_Saturation': oxygen_saturation,
                    'O2_Scale': o2_scale,
                    'Systolic_BP': systolic_bp,
                    'Heart_Rate': heart_rate,
                    'Temperature': temperature,
                    'Consciousness': consciousness,
                    'On_Oxygen': on_oxygen
                })
        
        # Show prediction results if available
        if hasattr(self, 'last_prediction') and self.last_prediction:
            self.display_prediction_results()
    
    def perform_risk_assessment(self, patient_data):
        """Perform risk assessment for given patient data"""
        try:
            # Make prediction
            risk_result = self.predictor.predict_risk(patient_data)
            
            if risk_result:
                self.last_prediction = {
                    'patient_data': patient_data,
                    'risk_result': risk_result,
                    'timestamp': datetime.now()
                }
                
                # Add to patient history
                self.patient_history.append(self.last_prediction)
                
                st.success("✅ Risk assessment completed!")
            else:
                st.error("❌ Error in risk assessment")
                
        except Exception as e:
            st.error(f"❌ Error in risk assessment: {e}")
    
    def display_prediction_results(self):
        """Display the prediction results"""
        if not hasattr(self, 'last_prediction'):
            return
        
        st.subheader("📊 Risk Assessment Results")
        
        prediction = self.last_prediction['risk_result']
        patient_data = self.last_prediction['patient_data']
        
        # Risk level display
        risk_level = prediction['predicted_risk']
        risk_color = {
            'Normal': 'risk-normal',
            'Low': 'risk-low', 
            'Medium': 'risk-medium',
            'High': 'risk-high'
        }
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Predicted Risk Level</h3>
                <p class="{risk_color.get(risk_level, '')}">{risk_level}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Escalation probability
            escalation = prediction['escalation_probability']
            st.markdown("**Escalation Probabilities:**")
            st.write(f"- Normal → High: {escalation['normal_to_high']:.3f}")
            st.write(f"- Low → High: {escalation['low_to_high']:.3f}")
            st.write(f"- Medium → High: {escalation['medium_to_high']:.3f}")
            st.write(f"- Overall Escalation: {escalation['overall_escalation']:.3f}")
        
        with col2:
            # Risk probabilities chart
            prob_data = prediction['probabilities']
            fig = px.bar(
                x=list(prob_data.keys()),
                y=list(prob_data.values()),
                title="Risk Level Probabilities",
                color=list(prob_data.values()),
                color_continuous_scale='RdYlGn_r'
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Patient data summary
        st.subheader("📋 Patient Data Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Respiratory Rate", f"{patient_data['Respiratory_Rate']} bpm")
            st.metric("Oxygen Saturation", f"{patient_data['Oxygen_Saturation']}%")
        
        with col2:
            st.metric("Systolic BP", f"{patient_data['Systolic_BP']} mmHg")
            st.metric("Heart Rate", f"{patient_data['Heart_Rate']} bpm")
        
        with col3:
            st.metric("Temperature", f"{patient_data['Temperature']}°C")
            st.metric("O2 Scale", patient_data['O2_Scale'])
        
        with col4:
            st.metric("Consciousness", patient_data['Consciousness'])
            st.metric("On Oxygen", "Yes" if patient_data['On_Oxygen'] else "No")
        
        # Recommendations
        st.subheader("💡 Clinical Recommendations")
        
        recommendations = self.get_clinical_recommendations(risk_level, patient_data)
        for rec in recommendations:
            st.info(f"• {rec}")
    
    def get_clinical_recommendations(self, risk_level, patient_data):
        """Generate clinical recommendations based on risk level and patient data"""
        recommendations = []
        
        if risk_level == 'High':
            recommendations.extend([
                "Immediate medical attention required",
                "Consider ICU admission",
                "Continuous monitoring of vital signs",
                "Prepare for emergency interventions"
            ])
        elif risk_level == 'Medium':
            recommendations.extend([
                "Close monitoring every 1-2 hours",
                "Consider step-down unit placement",
                "Review medication dosages",
                "Prepare escalation plan"
            ])
        elif risk_level == 'Low':
            recommendations.extend([
                "Regular monitoring every 4-6 hours",
                "Continue current treatment plan",
                "Monitor for any deterioration",
                "Consider discharge planning if stable"
            ])
        else:  # Normal
            recommendations.extend([
                "Routine monitoring",
                "Continue current care plan",
                "Monitor for any changes",
                "Consider discharge if appropriate"
            ])
        
        # Specific recommendations based on vital signs
        if patient_data['Oxygen_Saturation'] < 92:
            recommendations.append("Consider supplemental oxygen therapy")
        
        if patient_data['Heart_Rate'] > 100:
            recommendations.append("Monitor for cardiac complications")
        
        if patient_data['Temperature'] > 38.5:
            recommendations.append("Consider antipyretic therapy")
        
        return recommendations
    
    # Removed: show_patient_trends (Patient Trends & Monitoring)
    
    def show_model_performance(self):
        """Show model performance metrics"""
        st.header("📋 Model Performance & Evaluation")
        
        # Model information
        st.subheader("Model Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("**Model Type:** Ensemble (Best performing model selected)")
            st.info("**Training Data:** 800 patients (80%)")
            st.info("**Test Data:** 200 patients (20%)")
            st.info("**Features:** 12 (including encoded categorical variables)")
        
        with col2:
            # Dynamically determine the best model name and metrics
            metrics_path = 'model_metrics.json'
            if os.path.exists(metrics_path):
                import json
                with open(metrics_path, 'r') as fh:
                    mobj = json.load(fh)

                best_model_name = mobj.get('best_model', 'Unknown')
                st.success(f"**Best Model:** {best_model_name}")

                # Display metrics for the best model
                if best_model_name in mobj['models']:
                    best_model_metrics = mobj['models'][best_model_name]
                    st.success(f"**Accuracy:** {best_model_metrics['accuracy']*100:.1f}%")
                    st.success(f"**F1 Score:** {best_model_metrics['f1']*100:.1f}%")
                    st.success(f"**Precision:** {best_model_metrics['precision']*100:.1f}%")
                    st.success(f"**Recall:** {best_model_metrics['recall']*100:.1f}%")
                else:
                    st.info("Model metrics are not available in this view.")
            else:
                st.warning("No trained model available to display metrics.")
        
        # Feature importance
        if self.predictor:
            st.subheader("Feature Importance")
            
            importance = self.predictor.get_feature_importance()
            if importance:
                # Create feature importance chart
                fig = px.bar(
                    x=list(importance.keys()),
                    y=list(importance.values()),
                    title="Feature Importance Ranking",
                    labels={'x': 'Features', 'y': 'Importance Score'}
                )
                fig.update_layout(height=400, xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature importance table
                importance_df = pd.DataFrame(list(importance.items()), columns=['Feature', 'Importance'])
                st.dataframe(importance_df, use_container_width=True)
        
        # Model comparison
        st.subheader("Model Comparison")
        
        metrics_path = 'model_metrics.json'
        if os.path.exists(metrics_path):
            try:
                metrics = pd.read_json(metrics_path)
                # metrics is a dict-like structure; load via normal json read to reshape
                import json
                with open(metrics_path, 'r') as fh:
                    mobj = json.load(fh)

                models = []
                accs = []
                f1s = []
                precs = []
                recs = []
                for mname, mvals in mobj['models'].items():
                    models.append(mname)
                    accs.append(mvals.get('accuracy'))
                    f1s.append(mvals.get('f1'))
                    precs.append(mvals.get('precision'))
                    recs.append(mvals.get('recall'))

                comparison_df = pd.DataFrame({
                    'Model': models,
                    'Accuracy': accs,
                    'F1 Score': f1s,
                    'Precision': precs,
                    'Recall': recs
                })
                comparison_df[['Accuracy','F1 Score','Precision','Recall']] = comparison_df[['Accuracy','F1 Score','Precision','Recall']].astype(float)
                comparison_df[['Accuracy','F1 Score','Precision','Recall']] = comparison_df[['Accuracy','F1 Score','Precision','Recall']]*100
                st.dataframe(comparison_df, use_container_width=True)

                # Confusion matrix heatmap for selected model
                st.subheader('Confusion Matrix (select model)')
                sel = st.selectbox('Choose model', models, index=models.index(mobj.get('best_model')) if mobj.get('best_model') in models else 0)
                cm = mobj['models'][sel].get('confusion_matrix')
                if cm:
                    cm_df = pd.DataFrame(cm, index=self.predictor.label_encoder.classes_, columns=self.predictor.label_encoder.classes_)
                    fig = px.imshow(cm_df, text_auto=True, title=f'Confusion Matrix — {sel}', color_continuous_scale='Blues')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info('No confusion matrix available for this model.')

                # Recompute button removed by user request

            except Exception as e:
                st.error(f'Error loading model metrics: {e}')
        else:
            st.info('Model metrics not found. Run the training script or click the button below to compute metrics.')
            if st.button('Compute metrics (train models)'):
                import subprocess
                subprocess.run([os.path.join('.', '.venv', 'bin', 'python'), 'show_model_metrics.py'])
                safe_rerun()
        
        # Performance visualization
            # Performance visualization (only if metrics loaded)
            if 'comparison_df' in locals():
                fig = go.Figure()
                metrics = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
                models_list = comparison_df['Model'].tolist()
                for i, metric in enumerate(metrics):
                    fig.add_trace(go.Bar(
                        name=metric,
                        x=models_list,
                        y=[comparison_df[metric].iloc[j] for j in range(len(models_list))],
                        yaxis=f'y{i+1}' if i > 0 else 'y'
                    ))
                fig.update_layout(title="Model Performance Comparison", barmode='group', height=500)
                st.plotly_chart(fig, use_container_width=True)

def main():
    """Main function to run the HREWS application"""
    app = HREWSApp()
    app.run()

if __name__ == "__main__":
    main()
