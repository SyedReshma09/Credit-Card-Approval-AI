from flask import Flask, render_template, request, jsonify, session, send_from_directory
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
import plotly
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import random
import hashlib
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = 'super-secret-key-2024-credit-card-ai'
app.permanent_session_lifetime = timedelta(days=7)

# Try to import the predictor
try:
    from model.predict import CreditCardPredictor
    predictor = CreditCardPredictor()
    if predictor.model is None:
        print("[WARNING] Predictor model is None. Falling back to demo mode.")
        predictor = None
except Exception as e:
    print(f"[WARNING] Could not load predictor: {e}")
    predictor = None

# Sample data for demo mode
DEMO_DATA = {
    'approval_rate': 68.5,
    'total_applications': 15247,
    'avg_income': 187500,
    'avg_age': 42,
    'recent_activity': [
        {'time': 'Just now', 'user': 'John D.', 'status': 'Approved', 'amount': '$45,000'},
        {'time': '2 min ago', 'user': 'Sarah M.', 'status': 'Pending', 'amount': '$32,000'},
        {'time': '5 min ago', 'user': 'Robert K.', 'status': 'Approved', 'amount': '$78,000'},
        {'time': '12 min ago', 'user': 'Emily R.', 'status': 'Declined', 'amount': '$25,000'},
        {'time': '20 min ago', 'user': 'Michael S.', 'status': 'Approved', 'amount': '$92,000'},
    ]
}

@app.route('/')
def index():
    """Home page with animations"""
    return render_template('index.html', demo=DEMO_DATA)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Prediction page with interactive form"""
    if request.method == 'POST':
        try:
            # Get form data
            form_data = {
                'CODE_GENDER': request.form.get('gender'),
                'FLAG_OWN_CAR': request.form.get('own_car'),
                'FLAG_OWN_REALTY': request.form.get('own_property'),
                'CNT_CHILDREN': int(request.form.get('children', 0)),
                'AMT_INCOME_TOTAL': float(request.form.get('income', 0)),
                'NAME_EDUCATION_TYPE': request.form.get('education'),
                'NAME_FAMILY_STATUS': request.form.get('family_status'),
                'NAME_HOUSING_TYPE': request.form.get('housing_type'),
                'DAYS_BIRTH': -int(request.form.get('age', 0)) * 365,
                'DAYS_EMPLOYED': -int(request.form.get('employed_days', 0)),
                'FLAG_WORK_PHONE': int(request.form.get('work_phone', 0)),
                'FLAG_PHONE': int(request.form.get('phone', 0)),
                'FLAG_EMAIL': int(request.form.get('email', 0)),
                'OCCUPATION_TYPE': request.form.get('occupation'),
                'CNT_FAM_MEMBERS': int(request.form.get('family_members', 1))
            }
            
            # Generate a unique application ID
            app_id = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8].upper()
            
            if predictor is None:
                # Demo prediction
                import random
                status = random.choice(['Approved', 'Not Approved'])
                confidence = f"{random.uniform(65, 95):.1f}%"
                return render_template('result.html', 
                                     prediction=status,
                                     probability=confidence,
                                     form_data=form_data,
                                     app_id=app_id,
                                     timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            
            # Make prediction
            status, confidence = predictor.predict(form_data)
            
            return render_template('result.html', 
                                 prediction=status,
                                 probability=confidence,
                                 form_data=form_data,
                                 app_id=app_id,
                                 timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        except Exception as e:
            print(f"Prediction error: {e}")
            return render_template('predict.html', error=str(e))
    
    return render_template('predict.html')

@app.route('/dashboard')
def dashboard():
    """Interactive dashboard with live data"""
    graphs = []
    stats = {}
    error = None
    
    try:
        # Check if dataset files exist
        if not os.path.exists('dataset/application_record.csv'):
            error = "Dataset file not found. Using demo data."
            return render_template('dashboard.html', graphs=graphs, stats=stats, error=error)
        
        # Load datasets
        app_df = pd.read_csv('dataset/application_record.csv')
        
        # Calculate stats
        stats = {
            'total_applicants': len(app_df),
            'avg_income': f"${app_df['AMT_INCOME_TOTAL'].mean():,.0f}",
            'avg_age': f"{(-app_df['DAYS_BIRTH'].mean() / 365):.1f}",
            'car_owners': f"{app_df['FLAG_OWN_CAR'].value_counts().get('Y', 0):,}",
            'property_owners': f"{app_df['FLAG_OWN_REALTY'].value_counts().get('Y', 0):,}",
        }
        
        # Create visualizations
        graphs = []
        
        # 1. Income Distribution with histogram
        fig1 = go.Figure()
        fig1.add_trace(go.Histogram(
            x=app_df['AMT_INCOME_TOTAL'],
            nbinsx=50,
            marker_color='#6C63FF',
            opacity=0.7,
            name='Income Distribution'
        ))
        fig1.update_layout(
            title='💰 Income Distribution',
            xaxis_title='Annual Income ($)',
            yaxis_title='Frequency',
            template='plotly_white',
            height=400,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        graphs.append(json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder))
        
        # 2. Education vs Income - Box plot
        fig2 = go.Figure()
        for edu in app_df['NAME_EDUCATION_TYPE'].unique()[:5]:
            fig2.add_trace(go.Box(
                y=app_df[app_df['NAME_EDUCATION_TYPE'] == edu]['AMT_INCOME_TOTAL'],
                name=edu,
                boxmean='sd'
            ))
        fig2.update_layout(
            title='📊 Income by Education Level',
            xaxis_title='Education Level',
            yaxis_title='Income ($)',
            template='plotly_white',
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        graphs.append(json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder))
        
        # 3. Age vs Income Scatter
        app_df['AGE'] = -app_df['DAYS_BIRTH'] / 365
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=app_df['AGE'],
            y=app_df['AMT_INCOME_TOTAL'],
            mode='markers',
            marker=dict(
                size=8,
                color=app_df['AMT_INCOME_TOTAL'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Income")
            ),
            text=app_df['CODE_GENDER'],
            hovertemplate='Age: %{x:.1f}<br>Income: $%{y:,.0f}<br>Gender: %{text}<extra></extra>'
        ))
        fig3.update_layout(
            title='📈 Age vs Income Analysis',
            xaxis_title='Age (Years)',
            yaxis_title='Annual Income ($)',
            template='plotly_white',
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        graphs.append(json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder))
        
        # 4. Family Status - Donut Chart
        family_counts = app_df['NAME_FAMILY_STATUS'].value_counts()
        fig4 = go.Figure(data=[go.Pie(
            labels=family_counts.index,
            values=family_counts.values,
            hole=.4,
            marker=dict(colors=['#6C63FF', '#FF6584', '#FFC857', '#4ECDC4', '#45B7D1'])
        )])
        fig4.update_layout(
            title='👨‍👩‍👧‍👦 Family Status Distribution',
            template='plotly_white',
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        graphs.append(json.dumps(fig4, cls=plotly.utils.PlotlyJSONEncoder))
        
        # 5. Housing Type Bar Chart
        housing_counts = app_df['NAME_HOUSING_TYPE'].value_counts()
        fig5 = go.Figure(data=[
            go.Bar(
                x=housing_counts.index,
                y=housing_counts.values,
                marker_color=['#6C63FF', '#FF6584', '#FFC857', '#4ECDC4'],
                text=housing_counts.values,
                textposition='auto'
            )
        ])
        fig5.update_layout(
            title='🏠 Housing Type Distribution',
            xaxis_title='Housing Type',
            yaxis_title='Count',
            template='plotly_white',
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        graphs.append(json.dumps(fig5, cls=plotly.utils.PlotlyJSONEncoder))
        
        # 6. Occupation Distribution
        occ_counts = app_df['OCCUPATION_TYPE'].value_counts().head(10)
        fig6 = go.Figure(data=[
            go.Bar(
                x=occ_counts.values,
                y=occ_counts.index,
                orientation='h',
                marker_color='#4ECDC4',
                text=occ_counts.values,
                textposition='auto'
            )
        ])
        fig6.update_layout(
            title='💼 Top 10 Occupations',
            xaxis_title='Count',
            yaxis_title='Occupation',
            template='plotly_white',
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        graphs.append(json.dumps(fig6, cls=plotly.utils.PlotlyJSONEncoder))
        
    except Exception as e:
        error = f"Error loading dashboard: {str(e)}"
        print(f"Error: {error}")
        stats = DEMO_DATA
    
    return render_template('dashboard.html', graphs=graphs, stats=stats, error=error)

@app.route('/api/live-stats')
def live_stats():
    """API endpoint for live statistics"""
    try:
        # Generate random live data
        return jsonify({
            'timestamp': datetime.now().isoformat(),
            'applications_today': random.randint(50, 200),
            'approval_rate': f"{random.uniform(60, 80):.1f}%",
            'avg_response_time': f"{random.uniform(1.5, 4.5):.1f}s",
            'active_users': random.randint(20, 50)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/recent-applications')
def recent_applications():
    """API endpoint for recent applications"""
    try:
        statuses = ['Approved', 'Approved', 'Pending', 'Approved', 'Declined', 'Approved', 'Pending']
        names = ['James Wilson', 'Maria Garcia', 'Robert Taylor', 'Jennifer Lee', 
                'William Brown', 'Patricia Davis', 'Thomas Martinez']
        amounts = ['$45,000', '$32,000', '$78,000', '$25,000', '$92,000', '$67,000', '$54,000']
        
        recent = []
        for i in range(7):
            recent.append({
                'time': f"{random.randint(1, 30)} min ago",
                'user': random.choice(names),
                'status': random.choice(statuses),
                'amount': random.choice(amounts)
            })
        
        return jsonify(recent)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    print("Starting Credit Card Approval AI Application...")
    print("Visit http://127.0.0.1:5000 to access the application")
    print("Interactive features enabled!")
    app.run(debug=True, host='127.0.0.1', port=5000)