# Incorporating Chain-of-Context in Self-planning Enhances Interactive Code Generation from Natural Language

## Baseline Results:

**1. Vanilla RAG:** Score = 0.33%
```json
{
    "total_instances": 300,
    "submitted_instances": 300,
    "completed_instances": 77,
    "resolved_instances": 1,
    "unresolved_instances": 76,
    "empty_patch_instances": 1,
    "error_instances": 222
}
```

**2. Self-Planning RAG:** Score = 1.33%
```json
{
    "total_instances": 300,
    "submitted_instances": 300,
    "completed_instances": 42,
    "resolved_instances": 4,
    "unresolved_instances": 38,
    "empty_patch_instances": 2,
    "error_instances": 256
}
```