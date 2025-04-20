from langchain_openai import ChatOpenAI

def init_backend(model):
    '''
    Initialize model
    '''
    if model in ['o3-mini', 'gpt-4o-mini', 'gpt-4o']:
        return ChatOpenAI(model=model)
    else:
        raise NotImplementedError(f"Support for {model} is not yet implemented.")
    
def init_backend_greedy(model):
    '''
    Initialize model with greedy inference settings
    '''
    if model in ['o3-mini', 'gpt-4o-mini', 'gpt-4o']:
        if model == 'o3-mini':
            return ChatOpenAI(model=model)
        return ChatOpenAI(model=model, temperature=0.1, top_p=0.1)
    else:
        raise NotImplementedError(f"Support for {model} is not yet implemented.")