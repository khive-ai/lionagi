








class LionOrchestrator:

    def __init__(self, flow_name, model, **kw):
        
        
        self.flow_name = flow_name
        self.session = create_cc_flow_session(flow_name, model)
        self.builder = Builder(flow_name)

    @property
    def orchestrator_branch(self):
        return self.session.default_branch



