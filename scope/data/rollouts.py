from scope.data.waymax_adapter import WaymaxAdapter

def run_rollouts_for_candidates(parsed_scenario, root_time, candidates, config, sim_agent_policy='waymax_reactive'):
    adapter=WaymaxAdapter(config); traces={}
    for c in candidates:
        traces[c.candidate_id]=adapter.rollout_candidate(parsed_scenario, root_time, parsed_scenario['ego_id'], c.ego_future.detach().cpu().numpy(), sim_agent_policy)
    return traces
