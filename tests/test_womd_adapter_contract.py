def test_contract(parsed):
    for k in ['scenario_id','timestamps','tracks','track_valid','object_types','ego_id','sdc_track_index','map_polylines','map_valid','traffic_lights','route','metadata']:
        assert k in parsed
    assert parsed['tracks'].shape[-1]==11
