from kedro.pipeline import Pipeline

from alz.pipelines.ingest_mapping import create_pipeline as ingest_mapping_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    ingest_mapping = ingest_mapping_pipeline()
    return {
        "__default__": ingest_mapping,
        "ingest_mapping": ingest_mapping,
    }
