"""The sharing contract's names are the ones every model of an audience
uses: MindsHub's artifact rule, a skill's skill_access row and its rule, and
the saves the app sends. A model that renamed one of them would stop matching
what the app sends and MindsHub stores, so each is pinned to AudienceField.
The names and bounds themselves are the ones tests/fixtures/sharing_contract.yaml
records."""

from __future__ import annotations

from pathlib import Path

import yaml

from cowork.models.skill_access import SkillAccess
from cowork.schemas.artifact_access import ArtifactAccessRequest
from cowork.schemas.skills import SkillSharingRequest
from cowork.services import sharing_contract
from cowork.services.artifact_access import ArtifactAudience, ArtifactRule
from cowork.services.sharing_contract import AudienceField
from cowork.services.skill_sharing import SkillAccessRule

CONTRACT = yaml.safe_load((Path(__file__).parent / "fixtures" / "sharing_contract.yaml").read_text())
ORG_AUDIENCE = {AudienceField.ORG_ALLOWED, AudienceField.ALLOWED_TEAM_IDS, AudienceField.ALLOWED_USER_IDS}


def test_every_audience_model_uses_the_contract_names():
    assert [field.value for field in AudienceField] == CONTRACT["audience_fields"]
    assert set(ArtifactRule.model_fields) == set(AudienceField)
    assert set(ArtifactAudience.model_fields) == ORG_AUDIENCE
    assert set(AudienceField) <= set(ArtifactAccessRequest.model_fields)
    assert ORG_AUDIENCE <= set(SkillSharingRequest.model_fields)
    assert ORG_AUDIENCE <= set(SkillAccessRule.__dataclass_fields__)
    assert ORG_AUDIENCE <= set(SkillAccess.model_fields)


def test_the_bounds_are_the_recorded_ones():
    """Each bound under the name the fixture records, and no bound it
    doesn't: one renamed or added here without the fixture fails. The names
    MindsHub's and the app's contract modules share are the same there."""
    bounds = {
        name for name, value in vars(sharing_contract).items()
        if name.isupper() and isinstance(value, int) and not isinstance(value, bool)
    }
    assert bounds == set(CONTRACT["bounds"])
    assert {name: getattr(sharing_contract, name) for name in CONTRACT["bounds"]} == CONTRACT["bounds"]
