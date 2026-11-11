# Story & Spark Engine

**Live Deployment:** https://story-engine-0pi6.onrender.com/

## Overview
An AI-powered narrative and content generation platform built with a **Python (FastAPI)** backend. The system uses a custom **Spark Engine** that combines randomized and user-selectable creative "sparks" with structured user prompts to turn chaotic ideas into cohesive stories and scripts.

## Architecture & Completed Milestones (Phase 1 — Live)
- **Backend API:** Built asynchronous REST endpoints using **FastAPI (Python)** for prompt orchestration and LLM API integration.
- **Spark Engine Core:** Implemented dynamic combination logic for random and selectable creative sparks.
- **Frontend UI:** Responsive HTML/JS interface for summary input, spark selection, and real-time story generation.
- **Cloud Deployment:** Configured production environment, API key management, and live hosting on **Render**.

## Active Development Roadmap (Phase 2 — In Progress)

### 1. Core Engine Upgrades
- **Semantic 3D Concept Graph:** Replacing the flat random noun/spark list with a weighted 3D semantic graph to generate deeper, contextually linked creative sparks rather than purely chaotic outputs.
- **Prompt Pipeline Optimization:** Refining system prompts, token efficiency, and structured output validation across API requests.
- **Frontend & UI Overhaul:** Separating workspace views for multi-mode generation.

### 2. YouTube Content & Script Engine (New Module)
- **Dedicated Workspace:** Building a separate frontend and FastAPI routing pipeline specifically for YouTube creators.
- **Pre-Generation Research API:** Integrating an automated research material aggregation step that gathers structured context before calling the main content generation API.
- **End-to-End Script Formatting:** Generating hooks, pacing markers, and structured video scripts.
