"""
Advanced Analytics Module with Enhanced Features
Implements comprehensive analytics, reporting, and Excel export functionality
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import asyncio
from pathlib import Path
import io

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.chart import LineChart, Reference, BarChart, PieChart
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

@dataclass
class QueryMetrics:
    """Detailed metrics for each query"""
    timestamp: datetime
    session_id: str
    query: str
    response_time: float
    chunks_retrieved: int
    confidence: float
    context_size: int
    response_length: int
    user_satisfaction: Optional[float] = None
    topic_category: Optional[str] = None
    error_occurred: bool = False

class AdvancedAnalytics:
    """Enhanced analytics engine with predictive capabilities"""
    
    def __init__(self):
        self.query_history: List[QueryMetrics] = []
        self.session_analytics = defaultdict(dict)
        self.performance_thresholds = {
            'response_time': {'good': 2.0, 'acceptable': 5.0},
            'confidence': {'good': 0.8, 'acceptable': 0.6},
            'chunks': {'optimal': 5, 'max': 10}
        }
        self._cache = {}
        self._cache_ttl = 300  # 5 minutes
        
    def addQueryMetric(self, metric: QueryMetrics):
        """Add a query metric to history"""
        self.query_history.append(metric)
        
        # Update session analytics
        session = self.session_analytics[metric.session_id]
        if 'start_time' not in session:
            session['start_time'] = metric.timestamp
        session['last_activity'] = metric.timestamp
        session['query_count'] = session.get('query_count', 0) + 1
        session['total_response_time'] = session.get('total_response_time', 0) + metric.response_time
        
    def generateInteractiveReport(self) -> Dict[str, Any]:
        """Generate comprehensive interactive analytics report with enhanced error handling"""
        try:
            # Return empty report structure if no data or insufficient data
            if not self.query_history or len(self.query_history) < 2:
                return self._getEmptyReportStructure()
            
            # Validate data quality
            valid_queries = [q for q in self.query_history if q.response_time > 0]
            if len(valid_queries) < 2:
                return self._getEmptyReportStructure()
            
            # Check cache
            cache_key = f"report_{len(self.query_history)}"
            if cache_key in self._cache:
                cached_time, cached_data = self._cache[cache_key]
                if datetime.now() - cached_time < timedelta(seconds=self._cache_ttl):
                    return cached_data
            
            # Generate report components
            summary = self._generateExecutiveSummary()
            insights = self._generateInsights()
            recommendations = self._generateRecommendations()
            visualizations = self._createVisualizations()
            predictions = self._generatePredictions()
            
            report = {
                'generated_at': datetime.now().isoformat(),
                'summary': summary,
                'insights': insights,
                'recommendations': recommendations,
                'visualizations': visualizations,
                'predictions': predictions,
                'detailed_metrics': self._getDetailedMetrics()
            }
            
            # Cache the report
            self._cache[cache_key] = (datetime.now(), report)
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return self._getEmptyReportStructure()
    
    def _getEmptyReportStructure(self) -> Dict[str, Any]:
        """Return empty report structure when no data is available"""
        empty_fig = go.Figure()
        empty_fig.add_annotation(
            text="No data available yet. Start chatting to generate analytics!",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        empty_fig.update_layout(
            height=400,
            showlegend=False,
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, showticklabels=False)
        )
        
        return {
            'generated_at': datetime.now().isoformat(),
            'summary': {
                'total_queries': 0,
                'avg_response_time': 0,
                'success_rate': 0,
                'user_satisfaction_score': 0,
                'total_sessions': 0,
                'avg_session_duration': 0,
                'avg_confidence': 0,
                'total_chunks_processed': 0,
                'queries_per_session': 0
            },
            'insights': ["No insights available - insufficient data"],
            'recommendations': ["Start using the chatbot to generate analytics"],
            'visualizations': {
                'time_series': empty_fig,
                'performance': empty_fig,
                'engagement': empty_fig,
                'topics': empty_fig,
                'confidence': empty_fig
            },
            'predictions': {'message': 'Insufficient data for predictions'},
            'detailed_metrics': {}
        }
    
    def _generateExecutiveSummary(self) -> Dict[str, float]:
        """Generate executive summary statistics"""
        if not self.query_history:
            return {}
        
        df = pd.DataFrame([
            {
                'timestamp': q.timestamp,
                'response_time': q.response_time,
                'confidence': q.confidence,
                'chunks_retrieved': q.chunks_retrieved,
                'error_occurred': q.error_occurred
            }
            for q in self.query_history
        ])
        
        success_rate = (1 - df['error_occurred'].mean()) * 100
        
        # Calculate user satisfaction score based on multiple factors
        satisfaction_components = []
        
        # Response time component (40% weight)
        avg_response = df['response_time'].mean()
        if avg_response < self.performance_thresholds['response_time']['good']:
            satisfaction_components.append(100 * 0.4)
        elif avg_response < self.performance_thresholds['response_time']['acceptable']:
            satisfaction_components.append(70 * 0.4)
        else:
            satisfaction_components.append(40 * 0.4)
        
        # Confidence component (30% weight)
        avg_confidence = df['confidence'].mean()
        satisfaction_components.append(avg_confidence * 100 * 0.3)
        
        # Success rate component (30% weight)
        satisfaction_components.append(success_rate * 0.3)
        
        user_satisfaction = sum(satisfaction_components)
        
        return {
            'total_queries': len(self.query_history),
            'avg_response_time': round(avg_response, 2),
            'success_rate': round(success_rate, 1),
            'user_satisfaction_score': round(user_satisfaction, 1),
            'total_sessions': len(self.session_analytics),
            'avg_session_duration': self._calculateAvgSessionDuration(),
            'avg_confidence': round(avg_confidence, 2),
            'total_chunks_processed': df['chunks_retrieved'].sum(),
            'queries_per_session': round(len(self.query_history) / max(len(self.session_analytics), 1), 1)
        }
    
    def _calculateAvgSessionDuration(self) -> float:
        """Calculate average session duration in minutes"""
        if not self.session_analytics:
            return 0
        
        durations = []
        for session in self.session_analytics.values():
            if 'start_time' in session and 'last_activity' in session:
                duration = (session['last_activity'] - session['start_time']).total_seconds() / 60
                durations.append(duration)
        
        return round(sum(durations) / len(durations), 1) if durations else 0
    
    def _generateInsights(self) -> List[str]:
        """Generate AI-powered insights from analytics data"""
        insights = []
        
        if not self.query_history:
            return ["Insufficient data for insights"]
        
        df = pd.DataFrame([
            {
                'hour': q.timestamp.hour,
                'response_time': q.response_time,
                'confidence': q.confidence,
                'chunks': q.chunks_retrieved
            }
            for q in self.query_history
        ])
        
        # Peak usage hours
        if len(df) > 0:
            peak_hour = df.groupby('hour').size().idxmax() if df.groupby('hour').size().any() else 0
            insights.append(f"📊 Peak usage occurs at {peak_hour}:00 hours")
        
        # Response time trends
        if len(df) > 10:
            recent_avg = df.tail(5)['response_time'].mean()
            overall_avg = df['response_time'].mean()
            if recent_avg < overall_avg * 0.8:
                insights.append("⚡ Response times have improved by 20% recently")
            elif recent_avg > overall_avg * 1.2:
                insights.append("⚠️ Response times have increased by 20% recently")
        
        # Confidence analysis
        low_confidence_pct = (df['confidence'] < 0.6).mean() * 100
        if low_confidence_pct > 20:
            insights.append(f"🎯 {low_confidence_pct:.0f}% of queries have low confidence - consider adding more documents")
        
        # Chunk optimization
        avg_chunks = df['chunks'].mean()
        if avg_chunks > 7:
            insights.append("📚 High chunk retrieval count - consider optimizing chunk size")
        
        # Session patterns
        if len(self.session_analytics) > 5:
            avg_queries_per_session = len(self.query_history) / len(self.session_analytics)
            if avg_queries_per_session > 10:
                insights.append("💬 High engagement with average 10+ queries per session")
        
        return insights if insights else ["Analyzing patterns..."]
    
    def _generateRecommendations(self) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if not self.query_history:
            return ["Upload documents and start chatting to receive personalized recommendations"]
        
        df = pd.DataFrame([
            {
                'response_time': q.response_time,
                'confidence': q.confidence,
                'chunks': q.chunks_retrieved,
                'response_length': q.response_length
            }
            for q in self.query_history
        ])
        
        # Performance recommendations
        avg_response = df['response_time'].mean()
        if avg_response > self.performance_thresholds['response_time']['acceptable']:
            recommendations.append("🚀 Enable caching to improve response times")
            recommendations.append("📉 Reduce chunk size or retrieval count for faster responses")
        
        # Confidence recommendations
        avg_confidence = df['confidence'].mean()
        if avg_confidence < self.performance_thresholds['confidence']['acceptable']:
            recommendations.append("📚 Add more relevant documents to improve answer quality")
            recommendations.append("🔧 Adjust similarity threshold in settings")
        
        # Optimization recommendations
        if df['chunks'].mean() > self.performance_thresholds['chunks']['optimal']:
            recommendations.append("⚙️ Optimize chunk size for better retrieval efficiency")
        
        # Response length optimization
        if df['response_length'].mean() > 2000:
            recommendations.append("📝 Consider reducing max output tokens for conciseness")
        
        return recommendations if recommendations else ["System performing optimally"]
    
    def _createVisualizations(self) -> Dict[str, go.Figure]:
        """Create interactive Plotly visualizations with proper error handling - FIXED"""
        visualizations = {}
        
        try:
            if not self.query_history or len(self.query_history) < 2:
                # Return empty visualizations as Figure objects
                empty_fig = self._createEmptyChart("No data available")
                return {
                    'time_series': empty_fig,
                    'performance': empty_fig,
                    'engagement': empty_fig,
                    'topics': empty_fig,
                    'confidence': empty_fig
                }
            
            # Time series analysis
            visualizations['time_series'] = self._createTimeSeriesChart()
            
            # Performance metrics
            visualizations['performance'] = self._createPerformanceChart()
            
            # User engagement heatmap
            visualizations['engagement'] = self._createEngagementHeatmap()
            
            # Topic distribution
            visualizations['topics'] = self._createTopicDistribution()
            
            # Confidence distribution
            visualizations['confidence'] = self._createConfidenceChart()
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
            empty_fig = self._createEmptyChart("Error creating visualization")
            for key in ['time_series', 'performance', 'engagement', 'topics', 'confidence']:
                if key not in visualizations:
                    visualizations[key] = empty_fig
        
        return visualizations
    
    def _createEmptyChart(self, message: str) -> go.Figure:
        """Create empty chart with message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color="gray")
        )
        fig.update_layout(
            height=400,
            showlegend=False,
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, showticklabels=False)
        )
        return fig
    
    def _createTimeSeriesChart(self) -> go.Figure:
        """Create time series chart of queries and response times - Returns Figure object"""
        if len(self.query_history) < 2:
            return self._createEmptyChart("Insufficient data for time series")
        
        df = pd.DataFrame([
            {
                'timestamp': q.timestamp,
                'response_time': q.response_time,
                'confidence': q.confidence
            }
            for q in self.query_history
        ])
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Query Volume Over Time', 'Average Response Time'),
            vertical_spacing=0.15
        )
        
        # Query volume
        query_counts = df.set_index('timestamp').resample('1H').size()
        if len(query_counts) > 0:
            fig.add_trace(
                go.Scatter(
                    x=query_counts.index,
                    y=query_counts.values,
                    mode='lines+markers',
                    name='Queries',
                    line=dict(color='#667eea', width=2),
                    marker=dict(size=8)
                ),
                row=1, col=1
            )
        
        # Response time
        response_avg = df.set_index('timestamp').resample('1H')['response_time'].mean()
        if len(response_avg) > 0:
            fig.add_trace(
                go.Scatter(
                    x=response_avg.index,
                    y=response_avg.values,
                    mode='lines+markers',
                    name='Avg Response Time',
                    line=dict(color='#764ba2', width=2),
                    marker=dict(size=8)
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            height=600,
            showlegend=True,
            title_text="Time Series Analysis",
            hovermode='x unified'
        )
        
        return fig
    
    def _createPerformanceChart(self) -> go.Figure:
        """Create performance metrics visualization - Returns Figure object"""
        if not self.query_history:
            return self._createEmptyChart("No performance data")
        
        df = pd.DataFrame([
            {
                'response_time': q.response_time,
                'confidence': q.confidence,
                'chunks': q.chunks_retrieved
            }
            for q in self.query_history
        ])
        
        # Create performance gauge charts
        fig = make_subplots(
            rows=1, cols=3,
            specs=[[
                {'type': 'indicator'},
                {'type': 'indicator'},
                {'type': 'indicator'}
            ]],
            subplot_titles=('Response Time', 'Confidence Score', 'Retrieval Efficiency')
        )
        
        # Response time gauge
        avg_response = df['response_time'].mean()
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=avg_response,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Avg (seconds)"},
                delta={'reference': self.performance_thresholds['response_time']['good']},
                gauge={
                    'axis': {'range': [None, 10]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 2], 'color': "lightgreen"},
                        {'range': [2, 5], 'color': "yellow"},
                        {'range': [5, 10], 'color': "lightgray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': self.performance_thresholds['response_time']['acceptable']
                    }
                }
            ),
            row=1, col=1
        )
        
        # Confidence gauge
        avg_confidence = df['confidence'].mean()
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=avg_confidence,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Average"},
                delta={'reference': self.performance_thresholds['confidence']['good']},
                gauge={
                    'axis': {'range': [0, 1]},
                    'bar': {'color': "darkgreen"},
                    'steps': [
                        {'range': [0, 0.6], 'color': "lightgray"},
                        {'range': [0.6, 0.8], 'color': "yellow"},
                        {'range': [0.8, 1], 'color': "lightgreen"}
                    ]
                }
            ),
            row=1, col=2
        )
        
        # Chunk efficiency gauge
        avg_chunks = df['chunks'].mean()
        efficiency = max(0, 1 - (avg_chunks - 5) / 10)  # Optimal at 5 chunks
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=efficiency,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Score"},
                gauge={
                    'axis': {'range': [0, 1]},
                    'bar': {'color': "purple"},
                    'steps': [
                        {'range': [0, 0.5], 'color': "lightgray"},
                        {'range': [0.5, 0.8], 'color': "yellow"},
                        {'range': [0.8, 1], 'color': "lightgreen"}
                    ]
                }
            ),
            row=1, col=3
        )
        
        fig.update_layout(height=400, title_text="Performance Metrics Dashboard")
        
        return fig
    
    def _createEngagementHeatmap(self) -> go.Figure:
        """Create user engagement heatmap - Returns Figure object"""
        if len(self.query_history) < 2:
            return self._createEmptyChart("Insufficient engagement data")
        
        # Prepare data for heatmap
        df = pd.DataFrame([
            {
                'hour': q.timestamp.hour,
                'day': q.timestamp.strftime('%A'),
                'day_num': q.timestamp.weekday()
            }
            for q in self.query_history
        ])
        
        # Create pivot table
        heatmap_data = df.pivot_table(
            values='hour',
            index='day',
            columns='hour',
            aggfunc='size',
            fill_value=0
        )
        
        # Ensure all hours are present
        for hour in range(24):
            if hour not in heatmap_data.columns:
                heatmap_data[hour] = 0
        
        heatmap_data = heatmap_data.reindex(columns=range(24), fill_value=0)
        
        # Sort by day of week
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        existing_days = [day for day in day_order if day in heatmap_data.index]
        if existing_days:
            heatmap_data = heatmap_data.reindex(existing_days, fill_value=0)
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=list(range(24)),
            y=heatmap_data.index if len(heatmap_data.index) > 0 else ['No Data'],
            colorscale='Viridis',
            text=heatmap_data.values,
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Queries")
        ))
        
        fig.update_layout(
            title="User Engagement Heatmap",
            xaxis_title="Hour of Day",
            yaxis_title="Day of Week",
            height=400
        )
        
        return fig
    
    def _createTopicDistribution(self) -> go.Figure:
        """Create topic distribution chart using clustering - Returns Figure object"""
        if len(self.query_history) < 5:
            return self._createEmptyChart("Insufficient data for topic analysis")
        
        # Simple topic categorization based on query patterns
        topics = defaultdict(int)
        
        for q in self.query_history:
            query_lower = q.query.lower()
            
            if any(word in query_lower for word in ['what', 'define', 'explain']):
                topics['Informational'] += 1
            elif any(word in query_lower for word in ['how', 'step', 'process']):
                topics['Procedural'] += 1
            elif any(word in query_lower for word in ['why', 'reason', 'cause']):
                topics['Analytical'] += 1
            elif any(word in query_lower for word in ['compare', 'difference', 'versus']):
                topics['Comparative'] += 1
            else:
                topics['General'] += 1
        
        fig = go.Figure(data=[
            go.Pie(
                labels=list(topics.keys()),
                values=list(topics.values()),
                hole=0.3,
                marker=dict(
                    colors=['#667eea', '#764ba2', '#48bb78', '#f6ad55', '#fc8181']
                )
            )
        ])
        
        fig.update_layout(
            title="Query Topic Distribution",
            height=400,
            showlegend=True
        )
        
        return fig
    
    def _createConfidenceChart(self) -> go.Figure:
        """Create confidence score distribution - Returns Figure object"""
        if not self.query_history:
            return self._createEmptyChart("No confidence data")
        
        df = pd.DataFrame([
            {'confidence': q.confidence}
            for q in self.query_history
        ])
        
        fig = go.Figure()
        
        # Histogram
        fig.add_trace(go.Histogram(
            x=df['confidence'],
            nbinsx=20,
            name='Frequency',
            marker=dict(color='#667eea'),
            opacity=0.7
        ))
        
        # Add threshold lines
        fig.add_vline(
            x=self.performance_thresholds['confidence']['good'],
            line_dash="dash",
            line_color="green",
            annotation_text="Good"
        )
        
        fig.add_vline(
            x=self.performance_thresholds['confidence']['acceptable'],
            line_dash="dash",
            line_color="orange",
            annotation_text="Acceptable"
        )
        
        fig.update_layout(
            title="Confidence Score Distribution",
            xaxis_title="Confidence Score",
            yaxis_title="Frequency",
            height=400,
            showlegend=True
        )
        
        return fig
    
    def _generatePredictions(self) -> Dict[str, Any]:
        """Generate predictive analytics"""
        if len(self.query_history) < 20:
            return {'message': 'Insufficient data for predictions (need 20+ queries)'}
        
        # Prepare time series data
        df = pd.DataFrame([
            {
                'timestamp': q.timestamp,
                'hour': q.timestamp.hour,
                'response_time': q.response_time
            }
            for q in self.query_history
        ])
        
        # Simple predictions
        predictions = {
            'next_peak_hour': self._predictPeakHour(df),
            'expected_queries_next_hour': self._predictNextHourQueries(df),
            'response_time_trend': self._predictResponseTimeTrend(df)
        }
        
        return predictions
    
    def _predictPeakHour(self, df: pd.DataFrame) -> int:
        """Predict next peak usage hour"""
        if len(df) == 0:
            return 0
        hourly_counts = df.groupby('hour').size()
        return int(hourly_counts.idxmax()) if len(hourly_counts) > 0 else 0
    
    def _predictNextHourQueries(self, df: pd.DataFrame) -> int:
        """Predict number of queries in next hour"""
        recent_rate = len(df[df['timestamp'] > datetime.now() - timedelta(hours=1)])
        return max(1, int(recent_rate * 1.1))  # Simple growth factor
    
    def _predictResponseTimeTrend(self, df: pd.DataFrame) -> str:
        """Predict response time trend"""
        if len(df) < 10:
            return "stable"
        
        recent = df.tail(5)['response_time'].mean()
        older = df.head(5)['response_time'].mean()
        
        if recent < older * 0.9:
            return "improving"
        elif recent > older * 1.1:
            return "degrading"
        return "stable"
    
    def _getDetailedMetrics(self) -> Dict[str, Any]:
        """Get detailed metrics for export"""
        if not self.query_history:
            return {}
        
        response_times = [q.response_time for q in self.query_history]
        confidences = [q.confidence for q in self.query_history]
        
        return {
            'total_queries': len(self.query_history),
            'unique_sessions': len(self.session_analytics),
            'error_rate': sum(1 for q in self.query_history if q.error_occurred) / len(self.query_history),
            'percentiles': {
                'response_time_p50': np.percentile(response_times, 50) if response_times else 0,
                'response_time_p95': np.percentile(response_times, 95) if response_times else 0,
                'confidence_p50': np.percentile(confidences, 50) if confidences else 0,
                'confidence_p95': np.percentile(confidences, 95) if confidences else 0
            }
        }
    
    def exportToExcel(self, query_history: List[Dict]) -> bytes:
        """Export analytics to Excel with multiple sheets and formatting"""
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Executive Summary Sheet
            summary_df = pd.DataFrame([self._generateExecutiveSummary()])
            summary_df.to_excel(writer, sheet_name='Executive Summary', index=False)
            
            # Query History Sheet
            if query_history:
                history_df = pd.DataFrame(query_history)
                history_df.to_excel(writer, sheet_name='Query History', index=False)
            
            # Performance Metrics Sheet
            if self.query_history:
                metrics_data = []
                for q in self.query_history:
                    metrics_data.append({
                        'Timestamp': q.timestamp,
                        'Response Time (s)': q.response_time,
                        'Confidence': q.confidence,
                        'Chunks Retrieved': q.chunks_retrieved,
                        'Context Size': q.context_size,
                        'Response Length': q.response_length
                    })
                
                if metrics_data:
                    metrics_df = pd.DataFrame(metrics_data)
                    metrics_df.to_excel(writer, sheet_name='Performance Metrics', index=False)
            
            # Insights & Recommendations Sheet
            insights = self._generateInsights()
            recommendations = self._generateRecommendations()
            max_len = max(len(insights), len(recommendations))
            
            # Pad lists to same length
            insights.extend([''] * (max_len - len(insights)))
            recommendations.extend([''] * (max_len - len(recommendations)))
            
            insights_df = pd.DataFrame({
                'Insights': insights,
                'Recommendations': recommendations
            })
            insights_df.to_excel(writer, sheet_name='Insights', index=False)
            
            # Format the Excel file
            workbook = writer.book
            
            # Format Executive Summary
            if 'Executive Summary' in workbook.sheetnames:
                summary_sheet = workbook['Executive Summary']
                for cell in summary_sheet[1]:
                    cell.font = Font(bold=True, color="FFFFFF")
                    cell.fill = PatternFill(start_color="667EEA", end_color="667EEA", fill_type="solid")
                    cell.alignment = Alignment(horizontal="center")
            
            # Auto-adjust column widths
            for sheet in workbook.worksheets:
                for column in sheet.columns:
                    max_length = 0
                    column_letter = get_column_letter(column[0].column)
                    
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    
                    adjusted_width = min(50, max(12, max_length + 2))
                    sheet.column_dimensions[column_letter].width = adjusted_width
            
            # Apply conditional formatting to Performance Metrics
            if 'Performance Metrics' in workbook.sheetnames:
                metrics_sheet = workbook['Performance Metrics']
                
                # Add data bars for response time
                from openpyxl.formatting.rule import DataBarRule
                rule = DataBarRule(
                    start_type='min',
                    end_type='max',
                    color="FF638EC6"
                )
                
                # Find response time column
                for col_idx, cell in enumerate(metrics_sheet[1], 1):
                    if cell.value == 'Response Time (s)':
                        col_letter = get_column_letter(col_idx)
                        if metrics_sheet.max_row > 1:
                            metrics_sheet.conditional_formatting.add(
                                f'{col_letter}2:{col_letter}{metrics_sheet.max_row}',
                                rule
                            )
                        break
        
        output.seek(0)
        return output.read()