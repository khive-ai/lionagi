#!/usr/bin/env python3
"""
GitHub Queue Optimization Dashboard

Real-time monitoring and analytics dashboard for the GitHub queue optimization system.
Provides comprehensive insights into queue performance, bottlenecks, predictions,
and optimization effectiveness.

Features:
- Real-time queue status monitoring
- Performance analytics and trending
- Bottleneck detection and alerts
- ML prediction visualization
- ROI tracking and analysis
- Interactive optimization controls

Created: 2025-07-03
Author: GitHub Queue Optimization Specialist
"""

import asyncio
import json
import logging
import sqlite3
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import Input, Output, dcc, html
from github_queue_optimizer import (
    EventState,
    IssueMetrics,
    Priority,
    QueueOptimizer,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DashboardData:
    """Data management for dashboard"""

    def __init__(self, optimizer: QueueOptimizer):
        self.optimizer = optimizer
        self.db_path = Path(".cache/dashboard.db")
        self.db_path.parent.mkdir(exist_ok=True)
        self.init_database()

        # Real-time data
        self.current_metrics = {}
        self.historical_data = []
        self.predictions = {}
        self.alerts = deque(maxlen=100)

        # Update intervals
        self.last_update = datetime.now()
        self.update_interval = 60  # seconds

    def init_database(self):
        """Initialize dashboard database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS dashboard_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                metric_type TEXT NOT NULL,
                metric_value REAL NOT NULL,
                metadata TEXT
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS optimization_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                event_type TEXT NOT NULL,
                description TEXT NOT NULL,
                impact_score REAL,
                metadata TEXT
            )
        """
        )

        conn.commit()
        conn.close()

    async def update_data(self):
        """Update dashboard data"""
        try:
            # Get current queue status
            self.current_metrics = await self._collect_current_metrics()

            # Update predictions
            self.predictions = await self._collect_predictions()

            # Check for alerts
            await self._check_alerts()

            # Store historical data
            await self._store_historical_data()

            self.last_update = datetime.now()

        except Exception as e:
            logger.error(f"Dashboard data update failed: {e}")

    async def _collect_current_metrics(self) -> dict[str, Any]:
        """Collect current system metrics"""
        issues = list(self.optimizer.issue_cache.values())

        # Basic counts
        total_issues = len(issues)
        state_distribution = defaultdict(int)
        priority_distribution = defaultdict(int)

        for issue in issues:
            state_distribution[issue.state.value] += 1
            priority_distribution[issue.priority.name] += 1

        # Performance metrics
        avg_processing_time = np.mean(
            [i.processing_time for i in issues if i.processing_time]
        )
        avg_quality = np.mean(
            [i.deliverable_quality for i in issues if i.deliverable_quality]
        )
        avg_roi = np.mean([i.roi_estimate for i in issues if i.roi_estimate])

        # Bottleneck analysis
        bottlenecks = self.optimizer.predict_bottlenecks()

        # Parallel processing efficiency
        parallel_efficiency = self.optimizer._calculate_parallel_efficiency()

        return {
            "timestamp": datetime.now().isoformat(),
            "queue_status": {
                "total_issues": total_issues,
                "state_distribution": dict(state_distribution),
                "priority_distribution": dict(priority_distribution),
            },
            "performance": {
                "avg_processing_time": (
                    avg_processing_time
                    if not np.isnan(avg_processing_time)
                    else 0
                ),
                "avg_quality": avg_quality if not np.isnan(avg_quality) else 0,
                "avg_roi": avg_roi if not np.isnan(avg_roi) else 0,
                "parallel_efficiency": parallel_efficiency,
            },
            "bottlenecks": bottlenecks,
            "optimization_status": {
                "ml_enabled": self.optimizer.enable_ml,
                "sync_enabled": self.optimizer.sync_enabled,
                "last_sync": (
                    self.optimizer.last_sync.isoformat()
                    if self.optimizer.last_sync
                    else None
                ),
            },
        }

    async def _collect_predictions(self) -> dict[str, Any]:
        """Collect ML predictions"""
        predictions = {}

        if (
            self.optimizer.enable_ml
            and self.optimizer.priority_predictor.is_trained
        ):
            # Completion time predictions
            completion_predictions = {}
            for issue in self.optimizer.issue_cache.values():
                if issue.state != EventState.KNOWLEDGE_CAPTURED:
                    pred_time = self.optimizer.priority_predictor.predict_completion_time(
                        issue
                    )
                    completion_predictions[issue.number] = pred_time

            predictions["completion_times"] = completion_predictions

            # Bottleneck predictions
            bottleneck_pred = (
                self.optimizer.priority_predictor.predict_bottlenecks(
                    list(self.optimizer.issue_cache.values())
                )
            )
            predictions["future_bottlenecks"] = bottleneck_pred

        return predictions

    async def _check_alerts(self):
        """Check for system alerts"""
        # Check for bottlenecks
        bottlenecks = self.optimizer.predict_bottlenecks()

        for stage, info in bottlenecks.get("stages", {}).items():
            if info.get("severity", 0) > 1.5:
                alert = {
                    "timestamp": datetime.now().isoformat(),
                    "type": "bottleneck",
                    "severity": (
                        "warning" if info["severity"] < 2.0 else "critical"
                    ),
                    "message": f"Bottleneck detected in {stage}: {info['count']} issues",
                    "metadata": info,
                }
                self.alerts.append(alert)

        # Check for stale issues
        for issue in self.optimizer.issue_cache.values():
            age_hours = (
                datetime.now() - issue.created_at
            ).total_seconds() / 3600
            if age_hours > 168:  # 1 week
                alert = {
                    "timestamp": datetime.now().isoformat(),
                    "type": "stale_issue",
                    "severity": "warning",
                    "message": f"Issue #{issue.number} is stale ({age_hours:.1f} hours old)",
                    "metadata": {
                        "issue_number": issue.number,
                        "age_hours": age_hours,
                    },
                }
                self.alerts.append(alert)

    async def _store_historical_data(self):
        """Store current metrics as historical data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Store key metrics
            metrics_to_store = [
                (
                    "total_issues",
                    self.current_metrics["queue_status"]["total_issues"],
                ),
                (
                    "avg_processing_time",
                    self.current_metrics["performance"]["avg_processing_time"],
                ),
                (
                    "avg_quality",
                    self.current_metrics["performance"]["avg_quality"],
                ),
                ("avg_roi", self.current_metrics["performance"]["avg_roi"]),
                (
                    "parallel_efficiency",
                    self.current_metrics["performance"]["parallel_efficiency"],
                ),
            ]

            for metric_type, value in metrics_to_store:
                cursor.execute(
                    """
                    INSERT INTO dashboard_metrics (metric_type, metric_value, metadata)
                    VALUES (?, ?, ?)
                """,
                    (metric_type, value, json.dumps(self.current_metrics)),
                )

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to store historical data: {e}")

    def get_historical_data(self, hours: int = 24) -> pd.DataFrame:
        """Get historical metrics data"""
        try:
            conn = sqlite3.connect(self.db_path)

            query = f"""
                SELECT timestamp, metric_type, metric_value 
                FROM dashboard_metrics 
                WHERE timestamp > datetime('now', '-{hours} hours')
                ORDER BY timestamp
            """

            df = pd.read_sql_query(query, conn)
            conn.close()

            return df

        except Exception as e:
            logger.error(f"Failed to get historical data: {e}")
            return pd.DataFrame()


class OptimizationDashboard:
    """Main dashboard application"""

    def __init__(self, optimizer: QueueOptimizer):
        self.optimizer = optimizer
        self.data = DashboardData(optimizer)

        # Initialize Dash app
        self.app = dash.Dash(
            __name__, external_stylesheets=[dbc.themes.BOOTSTRAP]
        )
        self.app.title = "GitHub Queue Optimization Dashboard"

        # Setup layout
        self.setup_layout()
        self.setup_callbacks()

        # Background data updates
        self.update_task = None

    def setup_layout(self):
        """Setup dashboard layout"""
        self.app.layout = dbc.Container(
            [
                # Header
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H1(
                                    "🔍 GitHub Queue Optimization Dashboard",
                                    className="text-center mb-4",
                                ),
                                html.Hr(),
                            ]
                        )
                    ]
                ),
                # Status Cards
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                            [
                                                html.H4(
                                                    "Total Issues",
                                                    className="card-title",
                                                ),
                                                html.H2(
                                                    id="total-issues",
                                                    children="0",
                                                    className="text-primary",
                                                ),
                                            ]
                                        )
                                    ]
                                )
                            ],
                            width=3,
                        ),
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                            [
                                                html.H4(
                                                    "Avg Processing Time",
                                                    className="card-title",
                                                ),
                                                html.H2(
                                                    id="avg-processing-time",
                                                    children="0m",
                                                    className="text-info",
                                                ),
                                            ]
                                        )
                                    ]
                                )
                            ],
                            width=3,
                        ),
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                            [
                                                html.H4(
                                                    "Quality Score",
                                                    className="card-title",
                                                ),
                                                html.H2(
                                                    id="quality-score",
                                                    children="0%",
                                                    className="text-success",
                                                ),
                                            ]
                                        )
                                    ]
                                )
                            ],
                            width=3,
                        ),
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                            [
                                                html.H4(
                                                    "ROI Average",
                                                    className="card-title",
                                                ),
                                                html.H2(
                                                    id="roi-average",
                                                    children="0.0x",
                                                    className="text-warning",
                                                ),
                                            ]
                                        )
                                    ]
                                )
                            ],
                            width=3,
                        ),
                    ],
                    className="mb-4",
                ),
                # Main Content Tabs
                dbc.Tabs(
                    [
                        dbc.Tab(label="Queue Status", tab_id="queue-status"),
                        dbc.Tab(
                            label="Performance Analytics", tab_id="performance"
                        ),
                        dbc.Tab(
                            label="Bottleneck Analysis", tab_id="bottlenecks"
                        ),
                        dbc.Tab(label="Predictions", tab_id="predictions"),
                        dbc.Tab(
                            label="Optimization Controls", tab_id="controls"
                        ),
                    ],
                    id="main-tabs",
                    active_tab="queue-status",
                ),
                # Tab Content
                html.Div(id="tab-content", className="mt-4"),
                # Alerts Section
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader(
                                            html.H5("🚨 System Alerts")
                                        ),
                                        dbc.CardBody(id="alerts-content"),
                                    ]
                                )
                            ]
                        )
                    ],
                    className="mt-4",
                ),
                # Auto-refresh interval
                dcc.Interval(
                    id="interval-component",
                    interval=30 * 1000,  # 30 seconds
                    n_intervals=0,
                ),
            ],
            fluid=True,
        )

    def setup_callbacks(self):
        """Setup dashboard callbacks"""

        @self.app.callback(
            [
                Output("total-issues", "children"),
                Output("avg-processing-time", "children"),
                Output("quality-score", "children"),
                Output("roi-average", "children"),
            ],
            [Input("interval-component", "n_intervals")],
        )
        async def update_status_cards(n):
            """Update status cards"""
            await self.data.update_data()
            metrics = self.data.current_metrics

            total_issues = metrics.get("queue_status", {}).get(
                "total_issues", 0
            )
            avg_time = metrics.get("performance", {}).get(
                "avg_processing_time", 0
            )
            avg_quality = metrics.get("performance", {}).get("avg_quality", 0)
            avg_roi = metrics.get("performance", {}).get("avg_roi", 0)

            return (
                str(total_issues),
                f"{avg_time / 60:.1f}m",
                f"{avg_quality * 100:.1f}%",
                f"{avg_roi:.1f}x",
            )

        @self.app.callback(
            Output("tab-content", "children"),
            [Input("main-tabs", "active_tab")],
        )
        def update_tab_content(active_tab):
            """Update tab content based on selection"""
            if active_tab == "queue-status":
                return self.create_queue_status_tab()
            if active_tab == "performance":
                return self.create_performance_tab()
            if active_tab == "bottlenecks":
                return self.create_bottlenecks_tab()
            if active_tab == "predictions":
                return self.create_predictions_tab()
            if active_tab == "controls":
                return self.create_controls_tab()

            return html.Div("Select a tab")

        @self.app.callback(
            Output("alerts-content", "children"),
            [Input("interval-component", "n_intervals")],
        )
        def update_alerts(n):
            """Update alerts section"""
            alerts = list(self.data.alerts)[-10:]  # Last 10 alerts

            if not alerts:
                return html.P("No active alerts", className="text-muted")

            alert_items = []
            for alert in reversed(alerts):
                severity_color = {
                    "critical": "danger",
                    "warning": "warning",
                    "info": "info",
                }.get(alert["severity"], "secondary")

                alert_items.append(
                    dbc.Alert(
                        [
                            html.Strong(
                                f"{alert['type'].replace('_', ' ').title()}: "
                            ),
                            alert["message"],
                            html.Small(
                                f" - {alert['timestamp']}",
                                className="text-muted",
                            ),
                        ],
                        color=severity_color,
                        className="mb-2",
                    )
                )

            return alert_items

    def create_queue_status_tab(self) -> html.Div:
        """Create queue status tab content"""
        return html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader(
                                            html.H5(
                                                "📊 Queue Distribution by State"
                                            )
                                        ),
                                        dbc.CardBody(
                                            [
                                                dcc.Graph(
                                                    id="state-distribution-chart"
                                                )
                                            ]
                                        ),
                                    ]
                                )
                            ],
                            width=6,
                        ),
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader(
                                            html.H5("🎯 Priority Distribution")
                                        ),
                                        dbc.CardBody(
                                            [
                                                dcc.Graph(
                                                    id="priority-distribution-chart"
                                                )
                                            ]
                                        ),
                                    ]
                                )
                            ],
                            width=6,
                        ),
                    ],
                    className="mb-4",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader(
                                            html.H5("📋 Current Issues")
                                        ),
                                        dbc.CardBody(
                                            [html.Div(id="issues-table")]
                                        ),
                                    ]
                                )
                            ]
                        )
                    ]
                ),
            ]
        )

    def create_performance_tab(self) -> html.Div:
        """Create performance analytics tab"""
        return html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader(
                                            html.H5("📈 Performance Trends")
                                        ),
                                        dbc.CardBody(
                                            [
                                                dcc.Graph(
                                                    id="performance-trends-chart"
                                                )
                                            ]
                                        ),
                                    ]
                                )
                            ]
                        )
                    ],
                    className="mb-4",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader(
                                            html.H5(
                                                "⚡ Parallel Processing Efficiency"
                                            )
                                        ),
                                        dbc.CardBody(
                                            [
                                                dcc.Graph(
                                                    id="parallel-efficiency-chart"
                                                )
                                            ]
                                        ),
                                    ]
                                )
                            ],
                            width=6,
                        ),
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader(
                                            html.H5("💰 ROI Analysis")
                                        ),
                                        dbc.CardBody(
                                            [
                                                dcc.Graph(
                                                    id="roi-analysis-chart"
                                                )
                                            ]
                                        ),
                                    ]
                                )
                            ],
                            width=6,
                        ),
                    ]
                ),
            ]
        )

    def create_bottlenecks_tab(self) -> html.Div:
        """Create bottleneck analysis tab"""
        return html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader(
                                            html.H5("🚧 Current Bottlenecks")
                                        ),
                                        dbc.CardBody(
                                            [
                                                html.Div(
                                                    id="bottlenecks-analysis"
                                                )
                                            ]
                                        ),
                                    ]
                                )
                            ]
                        )
                    ],
                    className="mb-4",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader(
                                            html.H5("🔗 Dependency Graph")
                                        ),
                                        dbc.CardBody(
                                            [dcc.Graph(id="dependency-graph")]
                                        ),
                                    ]
                                )
                            ],
                            width=6,
                        ),
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader(
                                            html.H5("📊 Bottleneck History")
                                        ),
                                        dbc.CardBody(
                                            [
                                                dcc.Graph(
                                                    id="bottleneck-history-chart"
                                                )
                                            ]
                                        ),
                                    ]
                                )
                            ],
                            width=6,
                        ),
                    ]
                ),
            ]
        )

    def create_predictions_tab(self) -> html.Div:
        """Create predictions tab"""
        return html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader(
                                            html.H5(
                                                "🔮 Completion Time Predictions"
                                            )
                                        ),
                                        dbc.CardBody(
                                            [
                                                dcc.Graph(
                                                    id="completion-predictions-chart"
                                                )
                                            ]
                                        ),
                                    ]
                                )
                            ]
                        )
                    ],
                    className="mb-4",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader(
                                            html.H5(
                                                "⚠️ Future Bottleneck Predictions"
                                            )
                                        ),
                                        dbc.CardBody(
                                            [html.Div(id="future-bottlenecks")]
                                        ),
                                    ]
                                )
                            ],
                            width=6,
                        ),
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader(
                                            html.H5("📊 Prediction Accuracy")
                                        ),
                                        dbc.CardBody(
                                            [
                                                dcc.Graph(
                                                    id="prediction-accuracy-chart"
                                                )
                                            ]
                                        ),
                                    ]
                                )
                            ],
                            width=6,
                        ),
                    ]
                ),
            ]
        )

    def create_controls_tab(self) -> html.Div:
        """Create optimization controls tab"""
        return html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader(
                                            html.H5("🎛️ Optimization Controls")
                                        ),
                                        dbc.CardBody(
                                            [
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            [
                                                                dbc.Button(
                                                                    "🔄 Trigger Optimization",
                                                                    id="trigger-optimization-btn",
                                                                    color="primary",
                                                                    className="mb-2 w-100",
                                                                ),
                                                                dbc.Button(
                                                                    "⚡ Rebalance Queues",
                                                                    id="rebalance-queues-btn",
                                                                    color="info",
                                                                    className="mb-2 w-100",
                                                                ),
                                                                dbc.Button(
                                                                    "🔧 Auto-transition Issues",
                                                                    id="auto-transition-btn",
                                                                    color="success",
                                                                    className="mb-2 w-100",
                                                                ),
                                                            ],
                                                            width=6,
                                                        ),
                                                        dbc.Col(
                                                            [
                                                                dbc.Button(
                                                                    "📊 Generate Report",
                                                                    id="generate-report-btn",
                                                                    color="warning",
                                                                    className="mb-2 w-100",
                                                                ),
                                                                dbc.Button(
                                                                    "🔍 Predict Bottlenecks",
                                                                    id="predict-bottlenecks-btn",
                                                                    color="secondary",
                                                                    className="mb-2 w-100",
                                                                ),
                                                                dbc.Button(
                                                                    "🧠 Retrain ML Models",
                                                                    id="retrain-models-btn",
                                                                    color="dark",
                                                                    className="mb-2 w-100",
                                                                ),
                                                            ],
                                                            width=6,
                                                        ),
                                                    ]
                                                )
                                            ]
                                        ),
                                    ]
                                )
                            ],
                            width=6,
                        ),
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader(
                                            html.H5("⚙️ Configuration")
                                        ),
                                        dbc.CardBody(
                                            [
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            [
                                                                dbc.Label(
                                                                    "ML Optimization"
                                                                ),
                                                                dbc.Switch(
                                                                    id="ml-optimization-switch",
                                                                    value=True,
                                                                ),
                                                            ],
                                                            width=12,
                                                            className="mb-3",
                                                        ),
                                                        dbc.Col(
                                                            [
                                                                dbc.Label(
                                                                    "Real-time Sync"
                                                                ),
                                                                dbc.Switch(
                                                                    id="realtime-sync-switch",
                                                                    value=True,
                                                                ),
                                                            ],
                                                            width=12,
                                                            className="mb-3",
                                                        ),
                                                        dbc.Col(
                                                            [
                                                                dbc.Label(
                                                                    "Auto-transitions"
                                                                ),
                                                                dbc.Switch(
                                                                    id="auto-transitions-switch",
                                                                    value=True,
                                                                ),
                                                            ],
                                                            width=12,
                                                            className="mb-3",
                                                        ),
                                                    ]
                                                )
                                            ]
                                        ),
                                    ]
                                )
                            ],
                            width=6,
                        ),
                    ],
                    className="mb-4",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader(
                                            html.H5("📄 System Status")
                                        ),
                                        dbc.CardBody(
                                            [html.Div(id="system-status")]
                                        ),
                                    ]
                                )
                            ]
                        )
                    ]
                ),
            ]
        )

    async def start_dashboard(
        self, host: str = "0.0.0.0", port: int = 8050, debug: bool = False
    ):
        """Start the dashboard server"""
        logger.info(f"Starting optimization dashboard on {host}:{port}")

        # Start background data updates
        self.update_task = asyncio.create_task(self.background_updates())

        # Run dashboard
        self.app.run_server(host=host, port=port, debug=debug)

    async def background_updates(self):
        """Background task for data updates"""
        while True:
            try:
                await self.data.update_data()
                await asyncio.sleep(60)  # Update every minute
            except Exception as e:
                logger.error(f"Background update error: {e}")
                await asyncio.sleep(30)  # Retry after 30 seconds


def create_sample_data(optimizer: QueueOptimizer) -> QueueOptimizer:
    """Create sample data for dashboard testing"""
    import random
    from datetime import datetime

    # Sample issues
    sample_issues = []
    states = list(EventState)
    priorities = list(Priority)

    for i in range(1, 21):  # 20 sample issues
        issue = IssueMetrics(
            number=i,
            title=f"Sample Issue {i}: Enhancement Request",
            state=random.choice(
                states[:9]
            ),  # Exclude blocked/waiting/abandoned
            priority=random.choice(priorities),
            created_at=datetime.now()
            - timedelta(hours=random.randint(1, 168)),
            updated_at=datetime.now() - timedelta(hours=random.randint(0, 24)),
            dependencies=[],
            assignees=[],
            labels=[],
            complexity_score=random.uniform(0.3, 0.9),
            urgency_score=random.uniform(0.2, 0.8),
            business_value=random.uniform(0.4, 1.0),
            risk_score=random.uniform(0.1, 0.6),
            processing_time=random.uniform(300, 3600),  # 5 minutes to 1 hour
            agent_invocations=random.randint(1, 10),
            deliverable_quality=random.uniform(0.6, 1.0),
            roi_estimate=random.uniform(0.8, 3.5),
        )

        sample_issues.append(issue)
        optimizer.issue_cache[i] = issue

    logger.info(f"Created {len(sample_issues)} sample issues for dashboard")
    return optimizer


async def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="GitHub Queue Optimization Dashboard"
    )
    parser.add_argument("--host", default="0.0.0.0", help="Dashboard host")
    parser.add_argument(
        "--port", type=int, default=8050, help="Dashboard port"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode"
    )
    parser.add_argument(
        "--sample-data", action="store_true", help="Use sample data"
    )

    args = parser.parse_args()

    # Initialize optimizer
    optimizer = QueueOptimizer(enable_ml=True)

    # Add sample data if requested
    if args.sample_data:
        optimizer = create_sample_data(optimizer)

    # Initialize dashboard
    dashboard = OptimizationDashboard(optimizer)

    # Start dashboard
    await dashboard.start_dashboard(
        host=args.host, port=args.port, debug=args.debug
    )


if __name__ == "__main__":
    asyncio.run(main())
