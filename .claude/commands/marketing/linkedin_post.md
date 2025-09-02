# /linkedin-post

## Purpose

Create LinkedIn posts matching Ocean's authentic voice patterns for various
lionagi announcements, demonstrations, and educational content.

## Usage

```bash
/linkedin-post "[brief description of accomplishment/demo]" [--type TYPE] [--metrics "key metrics"]
```

**Types**:

- `demo` (major live demonstrations - uses two-voice format)
- `announcement` (technical feature releases)
- `promo` (live stream promotions)
- `launch` (product launches - strategic/philosophical)
- `educational` (tutorials/code examples)
- `reflection` (personal thoughts/industry commentary)
- `opensource` (sharing educational tools)

## Execution Pattern

### 1. Understand the Content Context

- Read the provided description and any referenced files/directories
- Identify the post type and appropriate voice pattern
- Extract key technical achievements and business impact

### 2. Choose Voice Pattern Based on Context

**For `demo` type (Two-Voice Format)**:

- Ocean: Brief humble technical context (1-2 sentences)
- Lion: Detailed strategic analysis with structured bullets

**For other types**: Use appropriate Ocean voice pattern from the context
patterns above

### 3. Format Requirements

- Use PLAIN TEXT format (no markdown: no **, ###, [], etc.)
- LinkedIn cannot render markdown, so use:
  - Bullet points with • or -
  - Section breaks with ---
  - ALL CAPS for emphasis instead of **bold**
  - Line breaks for readability
  - Hashtags at the end

### 4. Save Location

```
/Users/lion/projects/lionkhive/.khive/marketing/linkedin/YYYY-MM/YYYY_MM_DD_[topic].post.md
```

## Ocean's Voice Patterns (Context-Dependent)

### 1. Major Demonstrations (Two-Voice Format):

**Ocean**: Humble technical context - "It was a slow setup as I needed to
explain everything, but once setup, it's completely autonomous and reusable
patterns" **Lion**: Strategic promotional analysis with structured bullets and
business impact

### 2. Technical Announcements:

- Direct, feature-focused with code examples
- "Just shipped zero-config database persistence for lionagi v0.14.3"
- Emphasis on simplicity: "One line of code. Any Node object. Done."
- Include working code snippets

### 3. Live Stream Promotions:

- Casual, educational framing
- "Starting soon, Join us for Tuesday live with Ocean at Noon EST"
- "I will walk you through how to..."
- Focus on learning outcomes, not selling

### 4. Product Launches (Strategic/Philosophical):

- Paradigm shift messaging
- "Infrastructure for the Post-Database World"
- "The question isn't which database to choose anymore. It's what becomes
  possible when you don't have to choose."
- Deep philosophical implications of technical changes

### 5. Educational Content:

- Humble, helping others understand
- "Using claude code and lionagi to investigate codebase in 4 lines of python"
- Code-first demonstrations
- Focus on enabling others

### 6. Personal Reflections:

- Authentic, thoughtful commentary
- "Taking a week break from vibing for 🖋️ + 📜"
- Industry observations and personal insights
- Vulnerable, human perspective

### 7. Open Source Sharing:

- Educational focus over promotion
- "khive_claude is pure educational, I provide it for people to build their
  own..."
- Emphasis on learning and understanding
- Technical depth with accessibility

## Content Structure Templates

### Demo Type (Two-Voice Format):

```
[Live Stream/Demo Title]

Ocean:
[Brief humble technical context - acknowledging complexity but emphasizing autonomous reusable patterns]

---

Lion:
[Strategic analysis with bullets and business impact]
```

### Technical Announcement:

```
[Feature/Version Title]

[Direct feature description with emphasis on simplicity]

[Code example if applicable]

[Brief technical explanation]

hashtag#lionagi hashtag#AI hashtag#TechAnnouncement
```

### Live Stream Promo:

```
Starting soon, Join us for Tuesday live with Ocean at [TIME].

In today's session, I will walk you through [TOPIC/LEARNING OUTCOME].

[Brief technical context of what will be covered]

join the discord for link
```

### Product Launch (Strategic):

```
[Product/Version]: [Philosophical Hook]

[Paradigm shift observation]

-> [Section with strategic implications]

-> [Technical innovation explanation]

-> [Broader industry implications]

[Philosophical closing question]
```

### Educational Content:

```
[Descriptive title with learning outcome]

[Code example or simple demonstration]

[Brief technical explanation in accessible terms]

hashtag#lionagi hashtag#Education hashtag#AI
```

## Key Branding Rules

- Primary system: "lionagi" (not khive.ai)
- Show format: "Tuesday live with Ocean"
- Ocean's title: "Creator and Founder of the Lion ecosystem"
- Lion's role: Strategic analysis and promotion
- Technical focus: Autonomous multi-agent orchestration
- Business focus: Performance improvements and cost savings

## Example Usage

```bash
# Major live demonstration
/linkedin-post "Memory MCP migration validation with 4 parallel agents in 10 minutes" --type demo --metrics "240K savings, 10x performance improvement"

# Technical feature announcement  
/linkedin-post "lionagi v0.14.3 ships zero-config database persistence" --type announcement

# Live stream promotion
/linkedin-post "Tuesday live session on claude code orchestration with jupyter notebooks" --type promo

# Product launch with strategic messaging
/linkedin-post "Pydapter v1.0.0 infrastructure for post-database world" --type launch

# Educational content
/linkedin-post "Using claude code and lionagi to investigate codebase in 4 lines" --type educational
```

## Success Criteria

- Appropriate voice pattern for context type
- Plain text format suitable for LinkedIn (no markdown)
- Authentic Ocean voice matching examples
- Technical accuracy with appropriate depth
- Proper lionagi/Ocean branding
- Contextually appropriate calls-to-action
- Year-month directory structure followed
