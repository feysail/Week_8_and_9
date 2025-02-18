import dash
from dash import dcc, html
import requests
import pandas as pd
import plotly.express as px

# Initialize Dash app
app = dash.Dash(__name__)


app.layout = html.Div([
    html.H1("Fraud Detection Dashboard"),
    
    html.Div(id='summary-box'),
    
    dcc.Graph(id='fraud-trends'),
    
    dcc.Graph(id='device-browser-comparison'),
])


@app.callback(
    dash.dependencies.Output('summary-box', 'children'),
    [dash.dependencies.Input('summary-box', 'id')]
)
def update_summary_box(_):
    response = requests.get('http://localhost:5000/summary')
    summary = response.json()
    
    return html.Div([
        html.Div(f"Total Transactions: {summary['total_transactions']}"),
        html.Div(f"Total Fraud Cases: {summary['total_fraud_cases']}"),
        html.Div(f"Fraud Percentage: {summary['fraud_percentage']:.2f}%")
    ])


@app.callback(
    dash.dependencies.Output('fraud-trends', 'figure'),
    [dash.dependencies.Input('fraud-trends', 'id')]
)
def update_fraud_trends(_):
    response = requests.get('http://localhost:5000/fraud_trends')
    trends = pd.read_json(response.text)
    
    fig = px.line(trends, x='date', y='fraud_cases', title='Fraud Cases Over Time')
    return fig


if __name__ == '__main__':
    app.run_server(debug=True, port=8050)