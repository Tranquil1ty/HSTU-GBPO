import numpy as np
from kconf.get_config import get_json_config

def define_effective_view(durations, playtimes):
    wtd_json = get_json_config("reco.model.Fr22Q1WtdDefineLabelsTable")
    keys = [float(x) for x in wtd_json.keys()]
    wtd_dict = dict()
    for k in wtd_json:
        wtd_dict[float(k)] = wtd_json[k]
    assert len(durations) == len(playtimes)
    def search(duration, p):
        for i in range(len(keys) - 1):
            if duration < keys[i+1]:
                return p * 1000.0 >= wtd_dict[keys[i]][0]
        raise ValueError
    labels = np.zeros(len(durations), dtype=np.bool)
    for i in range(len(durations)):
        labels[i] = search(durations[i], playtimes[i])
    return labels

def define_long_view(durations, playtimes):
    wtd_json = get_json_config("reco.model.Fr22Q1WtdDefineLabelsTable")
    keys = [float(x) for x in wtd_json.keys()]
    wtd_dict = dict()
    for k in wtd_json:
        wtd_dict[float(k)] = wtd_json[k]
    assert len(durations) == len(playtimes)
    def search(duration, p):
        for i in range(len(keys) - 1):
            if duration < keys[i+1]:
                return p * 1000.0 >= wtd_dict[keys[i]][1]
        raise ValueError
    labels = np.zeros(len(durations), dtype=np.bool)
    for i in range(len(durations)):
        labels[i] = search(durations[i], playtimes[i])
    return labels


if __name__ == '__main__':
    from colossus.client import Client
    client = Client("grpc_colossusRecoSimItemV3")
    result = client.query(1624755238, timeout=100)
    result['effective_view'] = define_effective_view(result['duration'],result['play_time'])
    result = result[result['effective_view']]
    print(result)
    r = define_long_view([8, 20], [14, 3])
    print(r)
