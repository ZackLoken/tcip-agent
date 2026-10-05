"""A blank producer refuses at the stamp every save records its annotations' producer through."""

from __future__ import annotations

import pytest


def test_stamped_refuses_a_blank_author():
    from tcip_annotation.json_io import annotation_from_payload, stamped

    with pytest.raises(ValueError, match="producer"):
        stamped([annotation_from_payload({"subject": "fruit", "bbox": [1, 1, 5, 5]})], [],
                author="  ", now="t")


def test_stamped_names_the_author_of_a_new_annotation():
    from tcip_annotation.json_io import annotation_from_payload, stamped

    (new,) = stamped([annotation_from_payload({"subject": "fruit", "bbox": [1, 1, 5, 5]})], [],
                     author="save_annotations", now="t")
    assert (new.created_by, new.created_at) == ("save_annotations", "t")
