from scope.data.scenario_mining import mine_relevant_agents
from scope.data.candidates import generate_ego_candidates
from scope.planning.selector import select_candidate

def closed_loop_step(parsed_scenario, root_time, model, config):
    agents=mine_relevant_agents(parsed_scenario, root_time, parsed_scenario['ego_id'], config.get('data',{}).get('max_relevant_agents',8))
    cands=generate_ego_candidates(parsed_scenario, root_time, config)
    return {'agents':agents,'candidates':cands,'selected_candidate_id':0}
