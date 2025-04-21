# UKBiobank GWAS - Deep Learning


### Data Preparation Pipeline


```mermaid
graph TD
    A[split_vcf] --> B[parse_vcf]
    B --> C[reconstruct_exome]
    C --> D[index_exome]
    D --> E[aggregate]
    E --> F[esmc_embed]
    E --> G[construct_sqlite]
    F --> H[construct_h5]
```
