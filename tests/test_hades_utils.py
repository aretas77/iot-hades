import pytest
import hades_utils

def test_split_segments4():
    topic = "node/global/AA:BB:CC:DD:EE:FF/model/test"

    name, net, mac, end = hades_utils.split_segments4(topic)

    assert name == "node"
    assert net == "global"
    assert mac == "AA:BB:CC:DD:EE:FF"
    assert end == "model"

def test_verify_mac():
    table = {
        "AA:BB:CC:DD:EE:FF": True,
        "AA:BB:CC:DD:EE": False
    }

    for mac in table:
        assert hades_utils.verify_mac(mac) is table[mac]
