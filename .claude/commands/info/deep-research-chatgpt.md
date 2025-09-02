# Deep Research ChatGPT Command

## Purpose

Leverage ChatGPT's most advanced model (o3-pro or latest) for deep codebase
analysis and research when their capabilities exceed our current approach
efficiency. Assumes full codebase access and online research capabilities.

## When to Use

**Invoke ChatGPT deep research when:**

- Complex architectural analysis requiring multi-file understanding
- Need comprehensive pattern analysis across large codebases
- Seeking cutting-edge industry practices and latest research
- Time-critical decisions requiring rapid expert-level synthesis
- Cost of ChatGPT analysis < internal multi-session exploration
- Need validation of complex system designs with external perspective

**Don't use when:**

- Simple implementation tasks or debugging
- Security-sensitive internal analysis
- Quick clarifications that Claude can handle directly
- Tasks requiring Ocean's specific business context

## ChatGPT Capabilities Assumed

**Codebase Analysis:**

- Full repository traversal and understanding
- Cross-file dependency analysis
- Pattern recognition across multiple languages
- Architecture comprehension and optimization suggestions

**Research Capabilities:**

- Latest industry practices and research papers
- Real-time access to documentation and examples
- Comparative analysis with other systems
- Best practice recommendations from multiple sources

## Typical ChatGPT Research Flow

### Step-by-Step Process

```bash
1. 🤔 We give ChatGPT a research question
2. ❓ ChatGPT asks for clarifications
3. ✅ We provide clarifications  
4. 📋 Copy-paste ChatGPT's markdown analysis to storage
5. 🔄 Track processing status in our system
```

### Artifact Storage Structure

```
.khive/workspace/chatgpt_research/
├── YYYYMMDD_HHMMSS_[topic]/
│   ├── 01_initial_question.md        # Our original question
│   ├── 02_clarifications.md          # Our clarification responses  
│   ├── 03_chatgpt_analysis.md        # Copy-pasted ChatGPT response
│   ├── 04_processing_status.yaml     # Tracking metadata
│   └── 05_our_synthesis.md           # Our analysis of their analysis
```

### Processing Status Tracking

```yaml
# 04_processing_status.yaml
research_session:
  topic: "[Research topic]"
  date: "2025-01-13"
  status: "unprocessed"  # unprocessed | in_review | processed | implemented
  chatgpt_analysis_received: true
  our_synthesis_complete: false
  implementation_items_created: false
  knowledge_base_updated: false
  
processing_checklist:
  - [ ] Read ChatGPT analysis thoroughly
  - [ ] Extract key insights and recommendations  
  - [ ] Create implementation action items
  - [ ] Update relevant documentation
  - [ ] Share insights with development workflow
```

## Deep Research Framework

### 1. Research Request Template

```markdown
# ChatGPT Deep Research: [TOPIC]

## Research Objective

[Clear 1-2 sentence objective]

## Codebase Context

**Repository**: ohdearquant/fannrs (assume full access) **Focus Areas**:
[specific modules/components] **Current Challenge**: [what we're trying to
solve/optimize]

## Research Scope

1. **Architecture Analysis** - [specific architectural questions]
2. **Implementation Patterns** - [code pattern analysis needed]
3. **Optimization Opportunities** - [performance/design improvements]
4. **Industry Comparison** - [how we compare to best practices]
5. **Integration Roadmap** - [implementation strategy]

## Specific Questions

1. [Technical question 1]
2. [Business/architectural question 2]
3. [Implementation question 3]

## Expected Deliverables

- Separate markdown files for each analysis area
- Code examples and specific recommendations
- Implementation priority ranking
- Risk assessment for proposed changes
```

### 2. Codebase Analysis Request

```markdown
## Codebase Deep Dive Request

**Primary Focus**: [e.g., libs/khive/, apps/backend/auth-service/]

**Analysis Depth**:

- Cross-file dependency mapping
- Design pattern identification
- Performance bottleneck analysis
- Security vulnerability assessment
- Maintainability scoring

**Comparative Analysis**:

- Industry standard comparisons
- Alternative architecture patterns
- Latest framework/library alternatives
- Performance benchmarking data

**Output Format**: Structured analysis with code samples and metrics
```

### 3. Research Synthesis Protocol

```markdown
## Post-ChatGPT Processing

1. **Artifact Collection**
   - Download all ChatGPT-generated analyses
   - Organize into session workspace
   - Register artifacts in khive system

2. **Insight Extraction**
   - Create consolidated_insights.md
   - Extract actionable recommendations
   - Prioritize by impact and effort

3. **Implementation Planning**
   - Generate implementation_plan.md
   - Break down into specific tasks
   - Integrate with existing development workflow

4. **Knowledge Integration**
   - Update relevant documentation
   - Share insights with team
   - Archive for future reference
```

## ChatGPT-Specific Instructions

### Request Format for ChatGPT

```markdown
You are analyzing the fannrs codebase (assume full access). Please provide:

## Analysis Framework

1. **Code Architecture Review**
   - Component relationships and dependencies
   - Design pattern effectiveness
   - Scalability and maintainability assessment

2. **Implementation Quality**
   - Code quality metrics and patterns
   - Performance optimization opportunities
   - Security best practices compliance

3. **Industry Benchmarking**
   - Comparison with similar systems
   - Latest industry best practices
   - Emerging patterns and technologies

4. **Actionable Recommendations**
   - Specific code improvements
   - Architecture optimizations
   - Implementation roadmap with priorities

## Output Structure

Please provide separate, detailed analyses for each area with:

- Executive summary
- Detailed findings with code examples
- Specific recommendations
- Implementation priorities
- Risk assessments
```

### Quality Expectations

**ChatGPT should provide:**

- Specific code examples and snippets
- Concrete metrics and measurements
- Comparison with industry standards
- Implementation effort estimates
- Risk/benefit analysis for recommendations

## Example: KHIVE System Deep Research

### Research Request

```markdown
# ChatGPT Deep Research: KHIVE Multi-Agent Orchestration Optimization

## Research Objective

Analyze KHIVE orchestration system for architectural optimizations and industry
best practice alignment.

## Codebase Focus

- libs/khive/src/khive/ (all modules)
- .claude/resources/ (agent and domain specifications)
- Agent composition and coordination patterns

## Research Scope

1. **Orchestration Architecture** - Multi-agent coordination patterns
2. **Prompt Engineering** - Role+domain composition effectiveness
3. **Performance Optimization** - Bottlenecks and scaling improvements
4. **Industry Comparison** - How KHIVE compares to latest agent frameworks
5. **Integration Roadmap** - Path to next-generation capabilities

## Specific Questions

1. How does our Role+Domain composition compare to latest agent architecture
   patterns?
2. What are the most effective multi-agent coordination patterns in current
   research?
3. How can we optimize our 10-agent planning consensus mechanism?
4. What emerging technologies could enhance our orchestration capabilities?

## Expected Deliverables

- libs/khive/docs/chatgpt-analysis/orchestration-architecture.md
- libs/khive/docs/chatgpt-analysis/prompt-engineering-optimization.md
- libs/khive/docs/chatgpt-analysis/performance-improvements.md
- libs/khive/docs/chatgpt-analysis/industry-comparison.md
- libs/khive/docs/chatgpt-analysis/implementation-roadmap.md
```

## Practical Workflow Commands

### 1. Initialize Research Session

```bash
# Create timestamped research folder
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TOPIC="[research_topic]"
mkdir -p ".khive/workspace/chatgpt_research/${TIMESTAMP}_${TOPIC}"
cd ".khive/workspace/chatgpt_research/${TIMESTAMP}_${TOPIC}"

# Create initial files
echo "# Initial Question to ChatGPT" > 01_initial_question.md
echo "# Clarifications Provided" > 02_clarifications.md  
echo "# ChatGPT Analysis (Copy-Paste Here)" > 03_chatgpt_analysis.md
cp ~/.claude/templates/processing_status.yaml 04_processing_status.yaml
echo "# Our Synthesis and Next Steps" > 05_our_synthesis.md
```

### 2. After ChatGPT Interaction

```bash
# Manual steps:
# 1. Copy-paste ChatGPT's final analysis into 03_chatgpt_analysis.md
# 2. Update 04_processing_status.yaml to mark analysis received
# 3. Begin our synthesis in 05_our_synthesis.md

# Update status
sed -i 's/chatgpt_analysis_received: false/chatgpt_analysis_received: true/' 04_processing_status.yaml
sed -i 's/status: "unprocessed"/status: "in_review"/' 04_processing_status.yaml
```

### 3. Process Analysis

```bash
# Review and synthesize (manual process)
# 1. Read 03_chatgpt_analysis.md thoroughly
# 2. Extract key insights into 05_our_synthesis.md
# 3. Create implementation tasks
# 4. Update knowledge base as needed
# 5. Mark as processed

# Mark processing complete
sed -i 's/our_synthesis_complete: false/our_synthesis_complete: true/' 04_processing_status.yaml
sed -i 's/status: "in_review"/status: "processed"/' 04_processing_status.yaml
```

### 4. Check Processing Status

```bash
# List all unprocessed analyses
find .khive/workspace/chatgpt_research -name "04_processing_status.yaml" \
  -exec grep -l "status: \"unprocessed\"" {} \; | \
  xargs -I {} dirname {} | \
  xargs -I {} basename {}

# Count unprocessed
find .khive/workspace/chatgpt_research -name "04_processing_status.yaml" \
  -exec grep -l "status: \"unprocessed\"" {} \; | wc -l
```

## Quick Reference Commands

### Start New Research

```bash
# Template for quick setup
TOPIC="khive_optimization"  # Replace with your topic
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESEARCH_DIR=".khive/workspace/chatgpt_research/${TIMESTAMP}_${TOPIC}"
mkdir -p "$RESEARCH_DIR"
cd "$RESEARCH_DIR"

# Copy template and setup files
cp .claude/templates/processing_status.yaml 04_processing_status.yaml
echo "# ChatGPT Research: $TOPIC" > 01_initial_question.md
echo "# Clarifications for ChatGPT" > 02_clarifications.md
echo "# ChatGPT Analysis (PASTE HERE)" > 03_chatgpt_analysis.md  
echo "# Our Analysis & Next Steps" > 05_our_synthesis.md

# Update template with actual topic and timestamp
sed -i "s/\[Replace with research topic\]/$TOPIC/" 04_processing_status.yaml
sed -i "s/\[YYYYMMDD_HHMMSS\]/$TIMESTAMP/" 04_processing_status.yaml

echo "📁 Research session initialized: $RESEARCH_DIR"
```

### Check Unprocessed Research

```bash
# Quick check for pending ChatGPT analyses
echo "🔍 Unprocessed ChatGPT research sessions:"
find .khive/workspace/chatgpt_research -name "04_processing_status.yaml" \
  -exec grep -l "status: \"unprocessed\"" {} \; | \
  sed 's|.*/\([^/]*\)/04_processing_status.yaml|\1|' | \
  sort -r
```

### Mark Analysis Complete

```bash
# Run from within research directory after copy-pasting ChatGPT analysis
sed -i 's/chatgpt_analysis_received: false/chatgpt_analysis_received: true/' 04_processing_status.yaml
sed -i 's/status: "unprocessed"/status: "in_review"/' 04_processing_status.yaml
echo "✅ Analysis marked as received and in review"
```

## Success Criteria

**Effective ChatGPT research session:**

- Clear initial question with specific focus
- Productive clarification exchange
- Comprehensive analysis copy-pasted and preserved
- Our synthesis captures actionable insights
- Implementation items identified and tracked
- Status properly maintained throughout process

**Quality indicators:**

- ChatGPT provides specific code examples and recommendations
- Analysis addresses our actual architecture and constraints
- Recommendations include implementation priorities
- Our synthesis translates insights into concrete next steps

## Processing Workflow Summary

```bash
1. Initialize: Create research session folder and files
2. Question: Document initial question in 01_initial_question.md  
3. Clarify: Record clarifications in 02_clarifications.md
4. Receive: Copy-paste ChatGPT analysis to 03_chatgpt_analysis.md
5. Update: Mark analysis received in 04_processing_status.yaml
6. Synthesize: Create our analysis in 05_our_synthesis.md
7. Implement: Extract action items and update knowledge base
8. Archive: Mark status as processed when complete
```

---

_Ocean-only command for strategic technical decisions requiring ChatGPT's o3-pro
advanced analysis capabilities._
