# GitHub Release Checklist

1. Add a concise repository description and the topics `PBF-AM`, `LPBF`, `EBM`, `additive-manufacturing`, `optical-microscopy`, `semantic-segmentation`, and `U-Net`.
2. Review every image and annotation for permissions, participant identifiers, and proprietary information before publishing.
3. Confirm that `models/reference/` contains only the two intended reference checkpoints.
4. Add a code license and a data-use license. Completed: MIT for code and CC BY 4.0 for data and release artefacts.
5. Add `CITATION.cff` with the final title, authors, affiliations, publication DOI (when available), and repository URL. Completed; add the publication DOI when it is assigned.
7. Create a numbered, tagged GitHub release after the repository is reviewed, for example `v1.0.0`.
8. Cite the repository URL and release tag in the manuscript. Preserve the tagged commit hash in the paper's reproducibility record.
