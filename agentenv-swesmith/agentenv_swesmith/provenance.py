from __future__ import annotations


class SwesmithProvenanceError(RuntimeError):
    pass


def validate_revision_binding(
    *,
    dataset_revision: str,
    source_revision: str,
    image_dataset_revision: str,
    image_source_revision: str | None,
) -> None:
    """Fail closed unless data, source profiles, and OCI image agree."""

    if image_dataset_revision != dataset_revision:
        raise SwesmithProvenanceError(
            "SWE-smith image and dataset revisions disagree: "
            f"images={image_dataset_revision} dataset={dataset_revision}"
        )
    if image_source_revision is None:
        raise SwesmithProvenanceError(
            "SWE-smith image manifest must attest source_revision for a formal launch"
        )
    if image_source_revision != source_revision:
        raise SwesmithProvenanceError(
            "SWE-smith image and source revisions disagree: "
            f"images={image_source_revision} source={source_revision}"
        )
