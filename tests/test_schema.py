import pytest
from pydantic import ValidationError

from app.schemas import ParseRequest, ParseResponse


def test_parse_request_accepts_text():
    request = ParseRequest(text="schedule a meeting tomorrow")
    assert request.text == "schedule a meeting tomorrow"


def test_parse_request_rejects_missing_text():
    with pytest.raises(ValidationError):
        ParseRequest()


def test_parse_response_accepts_valid_frame():
    response = ParseResponse(
        intent="schedule",
        date="2026-11-07",
        time="10:30",
        place="Delhi",
    )

    assert response.intent == "schedule"
    assert response.date == "2026-11-07"
    assert response.time == "10:30"
    assert response.place == "Delhi"


def test_parse_response_allows_optional_fields():
    response = ParseResponse(intent="cancel")

    assert response.intent == "cancel"
    assert response.date is None
    assert response.time is None
    assert response.place is None
