import json
import stat

import pytest

from auth.profile import ProfileError, ProfileManager


VALID_STATE = {
    "cookies": [{"name": "SID", "value": "secret", "domain": ".google.com"}],
    "origins": [],
}


def test_atomic_profile_install_permissions_and_logout(tmp_path):
    path = tmp_path / "profile" / "storage_state.json"
    manager = ProfileManager(str(path))
    manager.install(
        VALID_STATE, verify=lambda candidate: json.loads(candidate.read_text())
    )
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert manager.exists()
    manager.logout()
    assert not manager.exists()


def test_failed_verification_does_not_replace_existing_profile(tmp_path):
    path = tmp_path / "storage_state.json"
    path.write_text('{"old": true}')
    manager = ProfileManager(str(path))
    with pytest.raises(ProfileError, match="验证失败"):
        manager.install(VALID_STATE, verify=lambda _: False)
    assert json.loads(path.read_text()) == {"old": True}


@pytest.mark.parametrize(
    "state",
    [{}, {"cookies": [], "origins": []}, {"cookies": "secret", "origins": []}],
)
def test_rejects_malformed_or_non_google_state(tmp_path, state):
    with pytest.raises(ProfileError):
        ProfileManager(str(tmp_path / "state.json")).install(
            state, verify=lambda _: True
        )
