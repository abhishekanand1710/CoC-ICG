from prompts.coc_prompts import *
from prompts.self_planning_icr_prompts import *

def format_context_chain(context_chain, iteration_analysis):
    '''
    Get formatted chain of context string including task description to include in prompt.
    '''
    context_str = ""
    for iteration, itr_context in context_chain.items():
        analysis = iteration_analysis[iteration]
        itr_context_str = ""
        for context in itr_context:
            content = context['content'].removeprefix('```python')
            content = content.removesuffix('```').strip()
            reason = context["reason"].strip().removeprefix("Reason:").strip()

            name = context["name"]
            if not name.endswith('.py'):
                name = name.split(".")[-1]

            context = CONTEXT_TEMPLATE.format(
                name = name,
                reason = reason,
                file_path = context["file_path"],
                content = content
            )
            itr_context_str += context
        
        context_str += ITER_CONTEXT_TEMPLATE.format(
            iteration = iteration,
            analysis = analysis,
            code = itr_context_str
        )

    return context_str

def format_context_chain_min(context_chain):
    '''
    Get formatted chain of context string excluding task description to include in prompt.
    '''
    context_str = ""
    for context in context_chain:
        content = context['content'].removeprefix('```python')
        content = content.removesuffix('```').strip()

        name = context["name"]
        if not name.endswith('.py'):
            name = name.split(".")[-1]

        context = CONTEXT_TEMPLATE.format(
            name = name,
            file_path = context["file_path"],
            content = content,
            iteration = context["retrieved_at"],
            reason = context["reason"].strip().removeprefix("Reason:").strip()
        )
        context_str += context

    return context_str

def format_candidate_context(context, reason):
    '''
    Get formatted chain of context string for retrieved code for relevance verification.
    '''
    context_str = ""
    content = context['content'].removeprefix('```python')
    content = content.removesuffix('```').strip()

    name = context["name"] 
    if not name.endswith('.py'):
        name = name.split(".")[-1]

    context_str = CANDIDATE_CONTEXT_TEMPLATE.format(
        name = name,
        reason = reason,
        file_path = context["file_path"],
        content = content,
        iteration = context["retrieved_at"]
    )

    return context_str

def format_context_chain_icr(context_chain):
    '''
    Get formatted chain of context string excluding task description to include in prompt for self planning with ICR.
    '''
    context_str = ""
    for context in context_chain:
        content = context['content'].removeprefix('```python')
        content = content.removesuffix('```').strip()

        name = context["name"]
        if not name.endswith('.py'):
            name = name.split(".")[-1]

        context = CONTEXT_TEMPLATE_ICR.format(
            name = name,
            file_path = context["file_path"],
            content = content
        )
        context_str += context

    return context_str