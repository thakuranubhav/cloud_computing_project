from app.schemas import ParseResponse


def test_parse_response_contains_required_intent():
    response = ParseResponse(
        intent="find",
        date="2026-11-07",
        time="09:00",
        place="Patna",
    )

    payload = response.model_dump()

    assert "intent" in payload
    assert payload["intent"] == "find"


def test_parse_response_serializes_to_json():
    response = ParseResponse(
        intent="schedule",
        date="2026-11-08",
        time="14:00",
        place="Delhi",
    )

    payload = response.model_dump_json()

    assert '"intent":"schedule"' in payload
    assert '"date":"2026-11-08"' in payload


def test_optional_fields_are_serialized():
    response = ParseResponse(intent="cancel")

    payload = response.model_dump()

    assert payload["intent"] == "cancel"
    assert payload["date"] is None
    assert payload["time"] is None
    assert payload["place"] is None
