#!/usr/bin/env python3
"""
Real-time GitHub Webhook Server for Queue Optimization

This server provides real-time synchronization between GitHub events and the KB
queue optimization system. It processes webhook events instantly and triggers
optimization actions.

Features:
- Real-time GitHub webhook processing
- Instant event queue synchronization
- Automated state transitions
- Conflict resolution
- Performance monitoring
- Retry mechanisms

Created: 2025-07-03
Author: GitHub Queue Optimization Specialist
"""

import asyncio
import hashlib
import hmac
import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from aiohttp import web
from github_queue_optimizer import (
    EventState,
    IssueMetrics,
    Priority,
    QueueOptimizer,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class WebhookEvent:
    """GitHub webhook event data"""

    event_type: str
    action: str
    payload: dict[str, Any]
    timestamp: datetime
    signature: str | None = None
    processed: bool = False
    retry_count: int = 0


class WebhookServer:
    """Real-time GitHub webhook server"""

    def __init__(
        self,
        port: int = 8080,
        webhook_secret: str | None = None,
        optimizer: QueueOptimizer | None = None,
    ):
        self.port = port
        self.webhook_secret = webhook_secret
        self.optimizer = optimizer or QueueOptimizer()

        # Event processing
        self.event_queue = asyncio.Queue()
        self.processing_tasks = set()
        self.retry_queue = asyncio.Queue()

        # Performance tracking
        self.stats = {
            "events_received": 0,
            "events_processed": 0,
            "events_failed": 0,
            "avg_processing_time": 0.0,
            "last_sync": None,
        }

        # Rate limiting
        self.rate_limiter = RateLimiter(max_requests=100, window_seconds=60)

        # Database for event persistence
        self.db_path = Path(".cache/webhooks.db")
        self.db_path.parent.mkdir(exist_ok=True)
        self.init_database()

        logger.info(f"Webhook server initialized on port {port}")

    def init_database(self):
        """Initialize webhook event database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS webhook_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                action TEXT NOT NULL,
                payload TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                processed BOOLEAN DEFAULT FALSE,
                retry_count INTEGER DEFAULT 0,
                processing_time REAL,
                error_message TEXT
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS sync_state (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                last_sync DATETIME DEFAULT CURRENT_TIMESTAMP,
                issues_synced INTEGER DEFAULT 0,
                conflicts_resolved INTEGER DEFAULT 0
            )
        """
        )

        conn.commit()
        conn.close()

    def verify_signature(self, payload: bytes, signature: str) -> bool:
        """Verify GitHub webhook signature"""
        if not self.webhook_secret:
            return True  # Skip verification if no secret configured

        expected_signature = hmac.new(
            self.webhook_secret.encode("utf-8"), payload, hashlib.sha256
        ).hexdigest()

        return hmac.compare_digest(f"sha256={expected_signature}", signature)

    async def handle_webhook(self, request: web.Request) -> web.Response:
        """Handle incoming GitHub webhook"""
        try:
            # Rate limiting
            if not await self.rate_limiter.allow_request():
                return web.Response(status=429, text="Rate limit exceeded")

            # Get event data
            event_type = request.headers.get("X-GitHub-Event", "unknown")
            signature = request.headers.get("X-Hub-Signature-256", "")
            payload_bytes = await request.read()

            # Verify signature
            if not self.verify_signature(payload_bytes, signature):
                logger.warning("Invalid webhook signature")
                return web.Response(status=401, text="Invalid signature")

            # Parse payload
            payload = json.loads(payload_bytes.decode("utf-8"))
            action = payload.get("action", "unknown")

            # Create webhook event
            event = WebhookEvent(
                event_type=event_type,
                action=action,
                payload=payload,
                timestamp=datetime.now(),
                signature=signature,
            )

            # Store in database
            await self.store_event(event)

            # Queue for processing
            await self.event_queue.put(event)

            self.stats["events_received"] += 1
            logger.info(f"Received webhook: {event_type}.{action}")

            return web.Response(status=200, text="OK")

        except Exception as e:
            logger.error(f"Webhook handling error: {e}")
            return web.Response(status=500, text="Internal server error")

    async def store_event(self, event: WebhookEvent):
        """Store webhook event in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO webhook_events (event_type, action, payload, timestamp)
                VALUES (?, ?, ?, ?)
            """,
                (
                    event.event_type,
                    event.action,
                    json.dumps(event.payload),
                    event.timestamp.isoformat(),
                ),
            )

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to store event: {e}")

    async def process_events(self):
        """Background event processing loop"""
        while True:
            try:
                # Get next event
                event = await self.event_queue.get()

                # Process event
                task = asyncio.create_task(self.process_single_event(event))
                self.processing_tasks.add(task)

                # Clean up completed tasks
                self.processing_tasks = {
                    t for t in self.processing_tasks if not t.done()
                }

            except Exception as e:
                logger.error(f"Event processing error: {e}")
                await asyncio.sleep(1)

    async def process_single_event(self, event: WebhookEvent):
        """Process a single webhook event"""
        start_time = datetime.now()

        try:
            logger.info(f"Processing {event.event_type}.{event.action}")

            # Route to appropriate handler
            if event.event_type == "issues":
                await self.handle_issues_event(event)
            elif event.event_type == "issue_comment":
                await self.handle_issue_comment_event(event)
            elif event.event_type == "pull_request":
                await self.handle_pull_request_event(event)
            else:
                logger.debug(f"Ignoring event type: {event.event_type}")

            # Update processing stats
            processing_time = (datetime.now() - start_time).total_seconds()
            self.stats["events_processed"] += 1
            self.stats["avg_processing_time"] = (
                self.stats["avg_processing_time"]
                * (self.stats["events_processed"] - 1)
                + processing_time
            ) / self.stats["events_processed"]

            # Mark as processed
            event.processed = True
            await self.update_event_status(event, processing_time)

        except Exception as e:
            logger.error(f"Event processing failed: {e}")
            self.stats["events_failed"] += 1
            event.retry_count += 1

            # Retry if under limit
            if event.retry_count < 3:
                await asyncio.sleep(
                    2**event.retry_count
                )  # Exponential backoff
                await self.retry_queue.put(event)
            else:
                await self.update_event_status(event, 0, str(e))

    async def handle_issues_event(self, event: WebhookEvent):
        """Handle GitHub issues webhook events"""
        payload = event.payload
        issue = payload.get("issue", {})
        action = event.action

        issue_number = issue.get("number")
        if not issue_number:
            return

        logger.info(f"Processing issue #{issue_number} - {action}")

        # Convert to IssueMetrics
        issue_metrics = await self.convert_to_issue_metrics(issue)

        if action == "opened":
            await self.handle_issue_opened(issue_metrics)
        elif action == "closed":
            await self.handle_issue_closed(issue_metrics)
        elif action == "labeled" or action == "unlabeled":
            await self.handle_issue_labeled(
                issue_metrics, payload.get("label", {})
            )
        elif action == "assigned" or action == "unassigned":
            await self.handle_issue_assigned(
                issue_metrics, payload.get("assignee", {})
            )

        # Update optimizer cache
        self.optimizer.issue_cache[issue_number] = issue_metrics

        # Trigger optimization analysis
        await self.trigger_optimization_analysis()

    async def handle_issue_comment_event(self, event: WebhookEvent):
        """Handle issue comment webhook events"""
        payload = event.payload
        issue = payload.get("issue", {})
        comment = payload.get("comment", {})
        action = event.action

        issue_number = issue.get("number")
        if not issue_number:
            return

        logger.info(f"Processing comment on issue #{issue_number} - {action}")

        # Check if comment contains deliverable
        comment_body = comment.get("body", "")
        if "Deliverable" in comment_body and "**Event**:" in comment_body:
            await self.handle_deliverable_posted(issue_number, comment_body)

        # Check for agent communication
        if self.is_agent_communication(comment_body):
            await self.handle_agent_communication(issue_number, comment_body)

        # Update issue metrics
        if issue_number in self.optimizer.issue_cache:
            issue_metrics = self.optimizer.issue_cache[issue_number]
            issue_metrics.last_activity = datetime.now()
            issue_metrics.webhook_events.append(event.payload)

    async def handle_pull_request_event(self, event: WebhookEvent):
        """Handle pull request webhook events"""
        payload = event.payload
        pr = payload.get("pull_request", {})
        action = event.action

        pr_number = pr.get("number")
        if not pr_number:
            return

        logger.info(f"Processing PR #{pr_number} - {action}")

        # Link PR to related issues
        linked_issues = await self.extract_linked_issues(pr.get("body", ""))

        for issue_number in linked_issues:
            if issue_number in self.optimizer.issue_cache:
                issue_metrics = self.optimizer.issue_cache[issue_number]
                issue_metrics.webhook_events.append(event.payload)

    async def handle_issue_opened(self, issue: IssueMetrics):
        """Handle new issue creation"""
        logger.info(f"New issue opened: #{issue.number}")

        # Automatically categorize and prioritize
        await self.auto_categorize_issue(issue)

        # Check for urgent issues
        if issue.priority == Priority.CRITICAL:
            await self.handle_urgent_issue(issue)

        # Update dependency graph
        self.optimizer.build_dependency_graph([issue])

    async def handle_issue_closed(self, issue: IssueMetrics):
        """Handle issue closure"""
        logger.info(f"Issue closed: #{issue.number}")

        # Update dependency graph
        if issue.number in self.optimizer.issue_cache:
            del self.optimizer.issue_cache[issue.number]

        # Check for dependent issues that can now proceed
        await self.check_unblocked_dependencies(issue.number)

    async def handle_issue_labeled(self, issue: IssueMetrics, label: dict):
        """Handle issue labeling changes"""
        label_name = label.get("name", "")

        if label_name.startswith("stage:"):
            # State transition detected
            new_state = EventState(label_name)
            old_state = issue.state

            logger.info(
                f"State transition detected: #{issue.number} {old_state.value} → {new_state.value}"
            )

            # Update state
            issue.state = new_state
            issue.transition_history.append((new_state, datetime.now()))

            # Check for automatic next transitions
            next_state = self.optimizer.auto_transition_states(issue)
            if next_state:
                await self.trigger_auto_transition(issue.number, next_state)

        elif label_name.startswith("priority:"):
            # Priority change
            new_priority = Priority[label_name.split(":")[1].upper()]
            issue.priority = new_priority

            # Recalculate queue order
            await self.reorder_processing_queue()

    async def handle_deliverable_posted(
        self, issue_number: int, comment_body: str
    ):
        """Handle deliverable posting"""
        logger.info(f"Deliverable posted for issue #{issue_number}")

        # Extract deliverable quality
        quality_score = self.extract_quality_score(comment_body)

        # Update issue metrics
        if issue_number in self.optimizer.issue_cache:
            issue = self.optimizer.issue_cache[issue_number]
            issue.deliverable_quality = quality_score
            issue.last_activity = datetime.now()

            # Check for auto-transition
            if quality_score >= 0.8:  # High quality threshold
                next_state = self.optimizer.auto_transition_states(issue)
                if next_state:
                    await self.trigger_auto_transition(
                        issue_number, next_state
                    )

    async def trigger_auto_transition(
        self, issue_number: int, next_state: EventState
    ):
        """Trigger automatic state transition"""
        logger.info(
            f"Auto-transitioning issue #{issue_number} to {next_state.value}"
        )

        # Update GitHub issue labels
        await self.update_github_labels(issue_number, next_state)

        # Update local cache
        if issue_number in self.optimizer.issue_cache:
            self.optimizer.issue_cache[issue_number].state = next_state

    async def update_github_labels(
        self, issue_number: int, new_state: EventState
    ):
        """Update GitHub issue labels"""
        try:
            # This would use GitHub API to update labels
            # For now, log the intended action
            logger.info(
                f"Would update issue #{issue_number} to {new_state.value}"
            )

            # In production, would make API call:
            # await self.github_client.update_issue_labels(issue_number, new_state.value)

        except Exception as e:
            logger.error(f"Failed to update GitHub labels: {e}")

    async def trigger_optimization_analysis(self):
        """Trigger optimization analysis"""
        try:
            # Generate optimization report
            report = self.optimizer.generate_optimization_report()

            # Check for critical issues
            if report.get("performance_metrics", {}).get("bottlenecks"):
                await self.handle_bottlenecks(
                    report["performance_metrics"]["bottlenecks"]
                )

            # Update processing queues
            await self.optimizer.rebalance_queues()

        except Exception as e:
            logger.error(f"Optimization analysis failed: {e}")

    async def handle_bottlenecks(self, bottlenecks: dict):
        """Handle detected bottlenecks"""
        logger.warning(f"Bottlenecks detected: {bottlenecks}")

        # Auto-create optimization issue if severe
        for stage, info in bottlenecks.get("stages", {}).items():
            if info.get("severity", 0) > 2.0:  # Severe bottleneck
                await self.create_optimization_issue(stage, info)

    async def create_optimization_issue(self, stage: str, info: dict):
        """Create GitHub issue for optimization"""
        logger.info(f"Creating optimization issue for stage: {stage}")

        # This would create a GitHub issue via API
        # For now, log the intended action
        logger.info(f"Would create optimization issue for {stage}: {info}")

    async def convert_to_issue_metrics(self, issue_data: dict) -> IssueMetrics:
        """Convert GitHub issue data to IssueMetrics"""
        labels = [label["name"] for label in issue_data.get("labels", [])]

        # Extract state
        state = EventState.RESEARCH_REQUESTED  # Default
        for label in labels:
            if label.startswith("stage:"):
                try:
                    state = EventState(label)
                    break
                except ValueError:
                    pass

        # Extract priority
        priority = Priority.MEDIUM  # Default
        for label in labels:
            if label.startswith("priority:"):
                try:
                    priority = Priority[label.split(":")[1].upper()]
                    break
                except (ValueError, KeyError):
                    pass

        # Extract dependencies
        dependencies = []
        body = issue_data.get("body", "")
        import re

        dep_matches = re.findall(r"#(\d+)", body)
        dependencies = [int(match) for match in dep_matches]

        return IssueMetrics(
            number=issue_data["number"],
            title=issue_data["title"],
            state=state,
            priority=priority,
            created_at=datetime.fromisoformat(
                issue_data["created_at"].replace("Z", "+00:00")
            ),
            updated_at=datetime.fromisoformat(
                issue_data["updated_at"].replace("Z", "+00:00")
            ),
            dependencies=dependencies,
            assignees=[
                assignee["login"]
                for assignee in issue_data.get("assignees", [])
            ],
            labels=labels,
        )

    async def auto_categorize_issue(self, issue: IssueMetrics):
        """Automatically categorize and prioritize new issues"""
        # Simple categorization logic
        title_lower = issue.title.lower()

        # Check for urgent keywords
        urgent_keywords = ["critical", "urgent", "blocking", "emergency"]
        if any(keyword in title_lower for keyword in urgent_keywords):
            issue.priority = Priority.CRITICAL

        # Estimate complexity
        if "enhancement" in title_lower:
            issue.complexity_score = 0.7
        elif "bug" in title_lower:
            issue.complexity_score = 0.5
        elif "research" in title_lower:
            issue.complexity_score = 0.8
        else:
            issue.complexity_score = 0.6

    async def handle_urgent_issue(self, issue: IssueMetrics):
        """Handle urgent/critical issues"""
        logger.warning(f"URGENT ISSUE: #{issue.number} - {issue.title}")

        # Move to front of processing queue
        # Implementation depends on queue structure

        # Send notifications
        await self.send_urgent_notification(issue)

    async def send_urgent_notification(self, issue: IssueMetrics):
        """Send urgent issue notification"""
        # This would send notifications via Slack, email, etc.
        logger.info(f"Sending urgent notification for issue #{issue.number}")

    def is_agent_communication(self, comment_body: str) -> bool:
        """Check if comment is agent communication"""
        agent_signatures = [
            "[AGENT-",
            "[RESEARCHER-",
            "[ANALYST-",
            "[ARCHITECT-",
        ]
        return any(sig in comment_body for sig in agent_signatures)

    async def handle_agent_communication(
        self, issue_number: int, comment_body: str
    ):
        """Handle agent-to-agent communication"""
        logger.info(f"Agent communication detected on issue #{issue_number}")

        # Parse agent communication
        # Update coordination state
        # Trigger any necessary actions

    async def extract_linked_issues(self, pr_body: str) -> list[int]:
        """Extract linked issues from PR body"""
        import re

        # Look for "Closes #123", "Fixes #456", etc.
        patterns = [
            r"closes\s+#(\d+)",
            r"fixes\s+#(\d+)",
            r"resolves\s+#(\d+)",
            r"addresses\s+#(\d+)",
        ]

        linked_issues = []
        for pattern in patterns:
            matches = re.findall(pattern, pr_body, re.IGNORECASE)
            linked_issues.extend([int(match) for match in matches])

        return linked_issues

    def extract_quality_score(self, comment_body: str) -> float:
        """Extract quality score from deliverable comment"""
        # Simple quality scoring based on content
        score = 0.5  # Base score

        # Look for quality indicators
        if "Status**: ✅" in comment_body:
            score += 0.3
        if "confidence" in comment_body.lower():
            score += 0.2
        if len(comment_body) > 500:  # Detailed deliverable
            score += 0.1

        return min(score, 1.0)

    async def check_unblocked_dependencies(self, resolved_issue: int):
        """Check for issues that can proceed after dependency resolution"""
        unblocked_issues = []

        for issue in self.optimizer.issue_cache.values():
            if resolved_issue in issue.dependencies:
                # Check if all dependencies are now resolved
                all_resolved = all(
                    dep not in self.optimizer.issue_cache
                    or self.optimizer.issue_cache[dep].state
                    == EventState.KNOWLEDGE_CAPTURED
                    for dep in issue.dependencies
                )

                if all_resolved:
                    unblocked_issues.append(issue)

        # Process unblocked issues
        for issue in unblocked_issues:
            logger.info(f"Issue #{issue.number} is now unblocked")
            await self.handle_issue_unblocked(issue)

    async def handle_issue_unblocked(self, issue: IssueMetrics):
        """Handle newly unblocked issue"""
        # Remove blocked status
        if issue.number in self.optimizer.blocked_issues:
            del self.optimizer.blocked_issues[issue.number]

        # Trigger optimization analysis
        await self.trigger_optimization_analysis()

    async def reorder_processing_queue(self):
        """Reorder processing queue based on updated priorities"""
        # This would reorder the processing queue
        # Implementation depends on queue structure
        logger.info("Reordering processing queue")

    async def update_event_status(
        self,
        event: WebhookEvent,
        processing_time: float,
        error_message: str = None,
    ):
        """Update event processing status in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                UPDATE webhook_events 
                SET processed = ?, processing_time = ?, error_message = ?
                WHERE event_type = ? AND action = ? AND timestamp = ?
            """,
                (
                    event.processed,
                    processing_time,
                    error_message,
                    event.event_type,
                    event.action,
                    event.timestamp.isoformat(),
                ),
            )

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to update event status: {e}")

    async def start_server(self):
        """Start the webhook server"""
        app = web.Application()
        app.router.add_post("/webhook", self.handle_webhook)
        app.router.add_get("/health", self.health_check)
        app.router.add_get("/stats", self.get_stats)

        # Start background tasks
        asyncio.create_task(self.process_events())
        asyncio.create_task(self.retry_failed_events())

        # Start server
        runner = web.AppRunner(app)
        await runner.setup()

        site = web.TCPSite(runner, "0.0.0.0", self.port)
        await site.start()

        logger.info(f"Webhook server started on port {self.port}")

        # Keep server running
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down webhook server")
        finally:
            await runner.cleanup()

    async def health_check(self, request: web.Request) -> web.Response:
        """Health check endpoint"""
        return web.json_response(
            {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "stats": self.stats,
            }
        )

    async def get_stats(self, request: web.Request) -> web.Response:
        """Get server statistics"""
        return web.json_response(self.stats)

    async def retry_failed_events(self):
        """Retry failed events"""
        while True:
            try:
                event = await self.retry_queue.get()
                await self.process_single_event(event)
            except Exception as e:
                logger.error(f"Retry processing error: {e}")
                await asyncio.sleep(5)


class RateLimiter:
    """Simple rate limiter for webhook requests"""

    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = []

    async def allow_request(self) -> bool:
        """Check if request is allowed"""
        now = datetime.now()

        # Remove old requests
        cutoff = now - timedelta(seconds=self.window_seconds)
        self.requests = [
            req_time for req_time in self.requests if req_time > cutoff
        ]

        # Check rate limit
        if len(self.requests) >= self.max_requests:
            return False

        # Add current request
        self.requests.append(now)
        return True


async def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="GitHub Webhook Server")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument("--webhook-secret", help="GitHub webhook secret")
    parser.add_argument(
        "--enable-ml", action="store_true", help="Enable ML optimization"
    )

    args = parser.parse_args()

    # Initialize optimizer
    optimizer = QueueOptimizer(enable_ml=args.enable_ml)

    # Initialize webhook server
    server = WebhookServer(
        port=args.port, webhook_secret=args.webhook_secret, optimizer=optimizer
    )

    # Start server
    await server.start_server()


if __name__ == "__main__":
    asyncio.run(main())
