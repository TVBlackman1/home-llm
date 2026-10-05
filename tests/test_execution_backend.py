from ollama_runner.execution import ExecutionBackend, execution_capability
from ollama_runner.ha.route import ExecutionRouter
from ollama_runner.request import command_from_result
from ollama_runner.semantic import ResolvedCommand, SemanticCommand, Target
from ollama_runner.types import Command, Result
from tests.test_ha_inventory import _snapshot


class _Quiet:
    def send(self, command: Command) -> Result:
        return Result(ok=True, command=command)


class _FakeHa:
    def __init__(self, state: str = "off") -> None:
        self.calls: list[tuple] = []
        self.state = state
        self.reads = 0

    def get_states(self) -> list:
        self.reads += 1
        return [{
            "entity_id": "media_player.frame",
            "state": self.state,
            "attributes": {"volume_level": 0.4, "source_list": ["TV", "HDMI"]},
        }]

    def call_service(self, domain: str, service: str, data: dict) -> int:
        self.calls.append((domain, service, data))
        return 200


def _resolved(intent: str, value: str = "") -> ResolvedCommand:
    return ResolvedCommand(
        status="resolved",
        intent=intent,
        semantic_target_id="media_player.frame",
        execution_target_id="media_player.frame",
        arguments={"value": value} if value else {},
        candidates=("media_player.frame",),
    )


def _status(result: Result, intent: str) -> str:
    return command_from_result(
        source="deterministic",
        decision="command",
        result=result,
        semantic=SemanticCommand(intent=intent, target=Target(device_type="tv")),
        resolution_status="resolved",
        device_id="media_player.frame",
    ).status


def test_advertised_stop_is_not_an_implementation():
    inventory = _snapshot()
    assert "media.stop" in inventory.registry.get("media_player.frame").capabilities
    assert execution_capability("media.stop") is None
    assert inventory.bindings["media_player.frame"].for_capability("media.stop").entity_id == (
        "media_player.frame"
    )
    client = _FakeHa()
    result = ExecutionRouter(inventory.registry, inventory.bindings, client).execute(
        _resolved("media.stop"),
        _Quiet(),
    )
    assert client.calls == []
    assert result.payload["reason"] == "execution_not_implemented"
    assert _status(result, "media.stop") == "unsupported"


def test_volume_set_uses_the_media_player_backend():
    inventory = _snapshot()
    capability = execution_capability("volume.set")
    assert capability is not None
    assert capability.backend is ExecutionBackend.HOME_ASSISTANT
    binding = inventory.bindings["media_player.frame"]
    assert binding.for_capability("volume.set") == binding.default
    client = _FakeHa()
    result = ExecutionRouter(inventory.registry, inventory.bindings, client).execute(
        _resolved("volume.set", "30 процентов"),
        _Quiet(),
    )
    assert client.calls == [(
        "media_player",
        "volume_set",
        {"entity_id": "media_player.frame", "volume_level": 0.3},
    )]
    assert client.reads == 0
    assert _status(result, "volume.set") == "success"


def test_power_override_toggles_the_remote_from_fresh_state():
    inventory = _snapshot()
    binding = inventory.bindings["media_player.frame"]
    assert binding.override("device.turn_on").entity_id == "remote.frame"
    client = _FakeHa("off")
    result = ExecutionRouter(inventory.registry, inventory.bindings, client).execute(
        _resolved("device.turn_on"),
        _Quiet(),
    )
    assert client.reads == 1
    assert client.calls == [("homeassistant", "toggle", {"entity_id": "remote.frame"})]
    assert result.payload["execution"]["current_state"] == "off"
    assert _status(result, "device.turn_on") == "success"


def test_power_without_an_override_does_not_call_the_media_player():
    inventory = _snapshot()
    binding = inventory.bindings["media_player.frame"]
    bare = type(binding)(binding.default, {}, binding.members)
    bindings = dict(inventory.bindings)
    bindings["media_player.frame"] = bare
    client = _FakeHa("off")
    result = ExecutionRouter(inventory.registry, bindings, client).execute(
        _resolved("device.turn_on"),
        _Quiet(),
    )
    assert client.calls == []
    assert result.payload["reason"] == "tv_power_unavailable"
    assert _status(result, "device.turn_on") == "unsupported"


def test_relative_volume_reads_the_live_level():
    inventory = _snapshot()
    client = _FakeHa()
    result = ExecutionRouter(inventory.registry, inventory.bindings, client).execute(
        _resolved("volume.increase", "10 процентов"),
        _Quiet(),
    )
    assert client.reads == 1
    assert client.calls == [(
        "media_player",
        "volume_set",
        {"entity_id": "media_player.frame", "volume_level": 0.5},
    )]
    assert result.payload["execution"]["current_volume"] == 0.4


def test_missing_device_binding_is_not_a_success():
    inventory = _snapshot()
    client = _FakeHa()
    result = ExecutionRouter(inventory.registry, {}, client).execute(
        _resolved("volume.set", "30 процентов"),
        _Quiet(),
    )
    assert client.calls == []
    assert result.payload["reason"] == "binding_unavailable"
    assert _status(result, "volume.set") == "unsupported"
