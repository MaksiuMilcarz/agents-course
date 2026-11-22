                      ┌───────────────────────────────────────────────┐
                      │             Langfuse Tracing                  │
                      │  (traces, spans, confidence scores, accuracy) │
                      └──────────────────────┬────────────────────────┘
                                             │
                                  ┌──────────┴──────────┐
                                  │  Lead Clinical      │
                                  │  CodeAgent          │
                                  └────┬───────┬───────┬┘
                                       │       │       │
             ┌─────────────────────────┘       │       └─────────────────────────┐
             ▼                                 ▼                                 ▼
   ┌───────────────────┐             ┌───────────────────┐             ┌───────────────────┐
   │  Literature / Web │             │   Vector Memory   │             │ Clinical Grilling │
   │  Research Agent   │             │   (Embeddings &   │             │ & Differential    │
   │  (DuckDuckGo)     │             │   State Consol.)  │             │ Confidence State  │
   └───────────────────┘             └───────────────────┘             └───────────────────┘

pip install -U smolagents duckduckgo-search langfuse sentence-transformers datasets numpy tqdm


Results:
1. Benchmark 0 shot: lots of fail, need more thinking time, many times has uniform confidence over all answers and answers A. Need different planning and reasoning sinc ethere is no option to grill the user.
2. Interactive mode: 