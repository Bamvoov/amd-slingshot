sequenceDiagram
    autonumber
    actor C as Candidate
    participant GUI as main_window.py
    participant RE as resume_engine.py
    participant VP as voice_pipeline.py
    participant BA as behavioral_analyzer.py
    participant LLM as AI Model (Local/Cloud)
    participant PDF as pdf_exporter.py

    C->>GUI: Uploads Resume PDF
    GUI->>RE: Extract Skills & Experience
    RE->>LLM: Send Context & Target Role
    LLM-->>VP: Generate Initial Question
    VP-->>GUI: Synthesize Speech (TTS)
    GUI->>C: Ask Question (Audio Output)
    
    C->>GUI: Speaks Answer (Audio Input)
    
    par Dual Processing
        GUI->>VP: Send Audio for Transcription (STT)
        VP->>VP: Convert Speech to Text
    and Behavioral Analysis
        GUI->>BA: Stream Audio for Analysis
        BA-->>BA: Detect Stutters & Confidence
    end
    
    VP->>LLM: Send Text Transcript
    LLM-->>VP: Evaluate & Generate Next Question
    
    Note over C, LLM: Conversation Loop Continues
    
    LLM->>GUI: Signal Interview Complete
    GUI->>PDF: Send Transcripts & Behavioral Metrics
    PDF-->>GUI: Compile Final Report
    GUI->>C: Provide Evaluation PDF
