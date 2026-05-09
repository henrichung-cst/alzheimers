from kedro.pipeline import Pipeline

from alz.pipelines.attribute import create_pipeline as attribute_pipeline
from alz.pipelines.enrich import create_pipeline as enrich_pipeline
from alz.pipelines.ingest_mapping import create_pipeline as ingest_mapping_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    ingest_mapping = ingest_mapping_pipeline()
    enrich = enrich_pipeline()
    attribute = attribute_pipeline()
    return {
        "__default__": ingest_mapping,
        "ingest_mapping": ingest_mapping,
        "enrich": enrich,
        "attribute": attribute,
    }
