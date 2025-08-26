# HR Automation System

Multi-agent workflow with feedback loops for comprehensive HR operations.

## Basic HR Workflow

```python
from lionagi import Branch
import json

# Define HR system agents
recruiter = Branch(
    system="You are an HR recruiter. Screen candidates and assess fit for roles.",
    name="recruiter"
)

interviewer = Branch(
    system="You are a technical interviewer. Conduct interviews and evaluate skills.",
    name="interviewer"  
)

manager = Branch(
    system="You are a hiring manager. Make final hiring decisions based on all feedback.",
    name="manager"
)

# Initial candidate screening
candidate_profile = {
    "name": "Alice Johnson",
    "experience": "5 years Python development",
    "skills": ["Python", "FastAPI", "PostgreSQL", "Docker"],
    "role": "Senior Backend Developer"
}

async def hr_workflow(candidate_profile: dict):
    """Complete HR workflow with feedback loops"""
    
    # Step 1: Initial screening
    screening = await recruiter.communicate(
        "Screen this candidate for the role",
        context=candidate_profile
    )
    
    # Step 2: Technical interview
    interview_result = await interviewer.communicate(
        "Conduct technical evaluation based on recruiter screening",
        context={"screening": screening, "candidate": candidate_profile}
    )
    
    # Step 3: Feedback loop - recruiter reviews interview
    recruiter_feedback = await recruiter.communicate(
        "Review the technical interview results and provide additional insights",
        context=interview_result
    )
    
    # Step 4: Manager decision
    final_decision = await manager.communicate(
        "Make hiring decision based on all feedback",
        context={
            "candidate": candidate_profile,
            "screening": screening,
            "interview": interview_result,
            "recruiter_feedback": recruiter_feedback
        }
    )
    
    return {
        "screening": screening,
        "interview": interview_result,
        "recruiter_feedback": recruiter_feedback,
        "final_decision": final_decision
    }

# Execute workflow
result = await hr_workflow(candidate_profile)
```

## Iterative Improvement Workflow

```python
from pydantic import BaseModel
from typing import Literal

class CandidateEvaluation(BaseModel):
    score: int  # 1-10
    strengths: list[str]
    concerns: list[str]
    recommendation: Literal["hire", "reject", "second_interview"]

async def iterative_hiring_process(candidate_profile: dict):
    """Multi-round evaluation with feedback improvements"""
    
    # Round 1: Initial assessments
    recruiter_eval = await recruiter.operate(
        instruction="Evaluate candidate for role fit and cultural alignment",
        context=candidate_profile,
        response_format=CandidateEvaluation
    )
    
    interviewer_eval = await interviewer.operate(
        instruction="Technical skills assessment and problem-solving evaluation", 
        context=candidate_profile,
        response_format=CandidateEvaluation
    )
    
    # Round 2: Cross-feedback and refinement
    recruiter_refined = await recruiter.communicate(
        "Review technical assessment and refine your evaluation",
        context={
            "your_evaluation": recruiter_eval.model_dump(),
            "technical_assessment": interviewer_eval.model_dump()
        }
    )
    
    interviewer_refined = await interviewer.communicate(
        "Consider cultural fit assessment and adjust technical evaluation",
        context={
            "your_evaluation": interviewer_eval.model_dump(),
            "cultural_assessment": recruiter_eval.model_dump()
        }
    )
    
    # Round 3: Final synthesis
    final_assessment = await manager.operate(
        instruction="Synthesize all evaluations into final hiring decision",
        context={
            "candidate": candidate_profile,
            "recruiter_initial": recruiter_eval.model_dump(),
            "interviewer_initial": interviewer_eval.model_dump(),
            "recruiter_refined": recruiter_refined,
            "interviewer_refined": interviewer_refined
        },
        response_format=CandidateEvaluation
    )
    
    return final_assessment

# Usage
assessment = await iterative_hiring_process(candidate_profile)
print(f"Final recommendation: {assessment.recommendation}")
print(f"Score: {assessment.score}/10")
```

## Performance Review System

```python
class PerformanceReview(BaseModel):
    employee_id: str
    overall_rating: int  # 1-5
    achievements: list[str]
    areas_for_improvement: list[str]
    goals: list[str]
    manager_feedback: str

# Performance review agents
direct_manager = Branch(
    system="You are a direct manager conducting performance reviews",
    name="direct_manager"
)

peer_reviewer = Branch(
    system="You provide peer feedback for performance reviews",
    name="peer_reviewer"
)

hr_specialist = Branch(
    system="You synthesize performance data and ensure consistency",
    name="hr_specialist"
)

async def performance_review_cycle(employee_data: dict):
    """360-degree performance review with multiple perspectives"""
    
    # Manager assessment
    manager_review = await direct_manager.operate(
        instruction="Conduct comprehensive performance review",
        context=employee_data,
        response_format=PerformanceReview
    )
    
    # Peer feedback
    peer_feedback = await peer_reviewer.communicate(
        "Provide peer perspective on performance and collaboration",
        context=employee_data
    )
    
    # HR synthesis with manager input
    hr_synthesis = await hr_specialist.communicate(
        "Review manager assessment and peer feedback for consistency",
        context={
            "employee": employee_data,
            "manager_review": manager_review.model_dump(),
            "peer_feedback": peer_feedback
        }
    )
    
    # Manager refinement based on HR feedback
    final_review = await direct_manager.communicate(
        "Refine review based on HR synthesis and peer feedback",
        context=hr_synthesis
    )
    
    return {
        "initial_review": manager_review,
        "peer_feedback": peer_feedback,
        "hr_synthesis": hr_synthesis,
        "final_review": final_review
    }

# Employee data
employee = {
    "id": "EMP001",
    "name": "Bob Smith",
    "role": "Software Engineer",
    "tenure": "2 years",
    "recent_projects": ["API refactoring", "Database optimization"],
    "peer_ratings": [4, 5, 4, 3]
}

review_results = await performance_review_cycle(employee)
```

## Policy Consultation System

```python
# HR policy consultant with tools
def lookup_policy(policy_area: str) -> str:
    """Look up HR policies by area"""
    policies = {
        "vacation": "Employees accrue 2 weeks vacation per year...",
        "remote_work": "Remote work approved for up to 3 days per week...",
        "benefits": "Health insurance starts after 30 days...",
        "performance": "Reviews conducted quarterly with annual ratings..."
    }
    return policies.get(policy_area, "Policy not found")

def check_compliance(situation: str) -> str:
    """Check compliance requirements"""
    return f"Compliance check for: {situation}"

# Policy-aware HR agent
policy_agent = Branch(
    system="You are an HR policy expert. Use tools to look up policies and check compliance.",
    tools=[lookup_policy, check_compliance],
    name="policy_agent"
)

# Employee inquiry agent  
employee_support = Branch(
    system="You help employees with HR questions and concerns.",
    name="employee_support"
)

async def hr_inquiry_workflow(employee_question: str):
    """Handle employee inquiries with policy consultation"""
    
    # Initial response from support agent
    initial_response = await employee_support.communicate(employee_question)
    
    # Policy consultation using tools
    policy_guidance = await policy_agent.ReAct(
        instruct={"instruction": f"Research policy guidance for: {employee_question}"},
        max_extensions=3
    )
    
    # Refined response with policy backing
    final_response = await employee_support.communicate(
        "Provide comprehensive response using policy guidance",
        context={
            "original_question": employee_question,
            "initial_response": initial_response,
            "policy_guidance": policy_guidance
        }
    )
    
    return final_response

# Usage
question = "Can I work remotely 4 days per week for family reasons?"
response = await hr_inquiry_workflow(question)
```

## Onboarding Workflow

```python
# Onboarding specialist agents
onboarding_coordinator = Branch(
    system="You coordinate new employee onboarding processes",
    name="coordinator"
)

it_setup = Branch(
    system="You handle IT setup and access provisioning for new employees",
    name="it_setup"
)

buddy_matcher = Branch(
    system="You match new employees with suitable workplace buddies",
    name="buddy_matcher"
)

async def onboarding_process(new_employee: dict):
    """Complete onboarding with multi-agent coordination"""
    
    # Phase 1: Initial planning
    onboarding_plan = await onboarding_coordinator.operate(
        instruction="Create onboarding plan for new employee",
        context=new_employee,
        response_format=dict
    )
    
    # Phase 2: Parallel setup tasks
    it_tasks = await it_setup.communicate(
        "Generate IT setup checklist and timeline",
        context={"employee": new_employee, "plan": onboarding_plan}
    )
    
    buddy_match = await buddy_matcher.communicate(
        "Find suitable workplace buddy based on role and interests",
        context={"employee": new_employee, "plan": onboarding_plan}
    )
    
    # Phase 3: Coordination feedback
    coordination_update = await onboarding_coordinator.communicate(
        "Review setup progress and adjust plan as needed",
        context={
            "original_plan": onboarding_plan,
            "it_progress": it_tasks,
            "buddy_assignment": buddy_match
        }
    )
    
    # Phase 4: Final checklist
    final_checklist = await onboarding_coordinator.communicate(
        "Generate final onboarding checklist with all requirements"
    )
    
    return {
        "plan": onboarding_plan,
        "it_setup": it_tasks,
        "buddy_match": buddy_match,
        "updates": coordination_update,
        "checklist": final_checklist
    }

# New employee data
new_hire = {
    "name": "Carol Davis",
    "role": "Product Manager",
    "start_date": "2024-02-01",
    "department": "Product",
    "manager": "Alice Johnson",
    "location": "Remote",
    "experience_level": "Senior"
}

onboarding_result = await onboarding_process(new_hire)
```

## State Persistence

```python
async def save_hr_workflow_state():
    """Save conversation states for audit and continuity"""
    
    # Save all agent conversations
    agent_states = {}
    
    for agent_name, agent in [
        ("recruiter", recruiter),
        ("interviewer", interviewer), 
        ("manager", manager)
    ]:
        # Convert to dict for persistence
        state = agent.to_dict()
        
        # Save to file or database
        with open(f"hr_data/{agent_name}_state.json", "w") as f:
            json.dump(state, f, indent=2)
        
        agent_states[agent_name] = state
    
    return agent_states

async def load_hr_workflow_state():
    """Restore agent states for continued processing"""
    
    restored_agents = {}
    
    for agent_name in ["recruiter", "interviewer", "manager"]:
        try:
            with open(f"hr_data/{agent_name}_state.json") as f:
                state_data = json.load(f)
                
            # Restore agent from saved state
            agent = Branch.from_dict(state_data)
            restored_agents[agent_name] = agent
            
        except FileNotFoundError:
            print(f"No saved state for {agent_name}")
    
    return restored_agents

# Usage
await save_hr_workflow_state()
restored_agents = await load_hr_workflow_state()
```

## Key Benefits

**Multi-Agent Collaboration:**

- Specialized roles with domain expertise
- Natural feedback loops between agents
- Iterative improvement of decisions

**Feedback Integration:**

- Cross-agent review and refinement
- Multiple perspectives on each decision
- Continuous improvement through iteration

**State Management:**

- Complete audit trail of decisions
- Ability to pause/resume workflows
- Historical analysis of HR patterns

**Production Features:**

- Policy integration with tool usage
- Compliance checking and documentation
- Scalable workflow orchestration
