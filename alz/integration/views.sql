CREATE OR REPLACE VIEW receiver_cache AS
SELECT *
FROM read_parquet('receiver_cache/receiver=*/data.parquet', hive_partitioning = true);

CREATE OR REPLACE VIEW pair_metadata AS
SELECT *
FROM read_parquet('pair_metadata.parquet');

CREATE OR REPLACE VIEW backbone_provenance AS
SELECT DISTINCT
  receiver,
  Path,
  Ligand,
  Receptor,
  EM,
  Target
FROM receiver_cache;

CREATE OR REPLACE VIEW contrast_comparison AS
SELECT
  sender,
  receiver,
  contrast,
  count(*) AS n_pathways,
  avg(TPDS) AS mean_tpds,
  avg(PDS) AS mean_pds,
  max(PDS) AS max_pds
FROM receiver_cache
GROUP BY sender, receiver, contrast;

CREATE OR REPLACE VIEW temporal_dynamics AS
SELECT
  sender,
  receiver,
  contrast,
  regexp_extract(contrast, '^(App|Tau|ApTt)', 1) AS genotype_effect,
  regexp_extract(contrast, '(2mo|4mo|6mo)$', 1) AS timepoint,
  count(*) AS n_pathways,
  avg(PDS) AS mean_pds,
  max(PDS) AS max_pds
FROM receiver_cache
GROUP BY sender, receiver, contrast;

CREATE OR REPLACE VIEW hub_matrix_by_contrast AS
SELECT
  sender,
  receiver,
  contrast,
  Ligand AS hub_gene,
  'Ligand' AS hub_role,
  count(*) AS n_paths,
  avg(PDS) AS mean_pds
FROM receiver_cache
GROUP BY sender, receiver, contrast, Ligand
UNION ALL
SELECT
  sender,
  receiver,
  contrast,
  Receptor AS hub_gene,
  'Receptor' AS hub_role,
  count(*) AS n_paths,
  avg(PDS) AS mean_pds
FROM receiver_cache
GROUP BY sender, receiver, contrast, Receptor
UNION ALL
SELECT
  sender,
  receiver,
  contrast,
  EM AS hub_gene,
  'EM' AS hub_role,
  count(*) AS n_paths,
  avg(PDS) AS mean_pds
FROM receiver_cache
GROUP BY sender, receiver, contrast, EM
UNION ALL
SELECT
  sender,
  receiver,
  contrast,
  Target AS hub_gene,
  'Target' AS hub_role,
  count(*) AS n_paths,
  avg(PDS) AS mean_pds
FROM receiver_cache
GROUP BY sender, receiver, contrast, Target;

CREATE OR REPLACE VIEW kinase_tpds_integration AS
SELECT *
FROM receiver_cache
WHERE false;

CREATE OR REPLACE VIEW backbone_recurrence_by_contrast AS
SELECT
  contrast,
  Path,
  count(DISTINCT sender || '->' || receiver) AS n_pairs,
  avg(PDS) AS mean_pds,
  max(PDS) AS max_pds
FROM receiver_cache
GROUP BY contrast, Path;

CREATE OR REPLACE VIEW target_convergence_by_contrast AS
SELECT
  contrast,
  receiver,
  Target,
  count(DISTINCT sender) AS n_senders,
  count(*) AS n_paths,
  avg(PDS) AS mean_pds,
  max(PDS) AS max_pds
FROM receiver_cache
GROUP BY contrast, receiver, Target;
