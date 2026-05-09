from kedro.pipeline import Pipeline

from alz.pipelines.attribute import create_pipeline as attribute_pipeline
from alz.pipelines.enrich import create_pipeline as enrich_pipeline
from alz.pipelines.ingest_mapping import create_pipeline as ingest_mapping_pipeline
from alz.pipelines.mechanism import create_pipeline as mechanism_pipeline
from alz.pipelines.normalize import create_pipeline as normalize_pipeline
from alz.pipelines.recovery import create_pipeline as recovery_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    ingest_mapping = ingest_mapping_pipeline()
    normalize = normalize_pipeline()
    enrich = enrich_pipeline()
    attribute = attribute_pipeline()
    mechanism = mechanism_pipeline()
    recovery = recovery_pipeline()
    return {
        "__default__": ingest_mapping,
        "ingest_mapping": ingest_mapping,
        "normalize": normalize,
        "enrich": enrich,
        "attribute": attribute,
        "mechanism": mechanism,
        "recovery": recovery,
    }
