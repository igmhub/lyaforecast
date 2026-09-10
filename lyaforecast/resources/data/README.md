# Reference inputs

The DESI-2 INIs use the following exact inputs from the DESI-2 SRD forecast dataset.
They were bundled during modernization without changing their contents:

- `dn_dzdr_qso_desi_2.dat`: `qso_target_selection/dn_dzdr_qso_desi_2.dat`
  in the DESI-2 SRD dataset (162,000 bytes).
- `lbg_matched_dndzdr.txt` and `lae_matched_dndzdr.txt`: the corresponding
  `output_lyaforecast/` tables; previously shipped under `examples/desi2/`.
  Both repository copies were byte-identical to the referenced external inputs.

These three-column tables contain redshift, r-band magnitude, and counts per
redshift/magnitude cell per square degree. Tracer loading applies density
normalization and divides by the grid cell widths. Generation settings beyond
this dataset provenance were not supplied; none have been inferred.

SHA-256 checksums:

- `dn_dzdr_qso_desi_2.dat`: `78914d3a9ad0307ad94072d1b39a54b41d4043ebca5a0e4d1d4d202a3d489c8d`
- `lbg_matched_dndzdr.txt`: `995e1b06e09735e4babdcdfb6f0d253371c9d108b06c5826eb92a9774defcdef`
- `lae_matched_dndzdr.txt`: `f56f7d3ba3144ae0fdfaaeba93c557d337e74f8de993110133aaaba3d7f1302e`

The DESI-2 SNR directories are `DESI-2-QSO` and `DESI-2-LBG`. Older DESIQSO,
DESILBG, and density tables remain distinct reference inputs; they are not
interchangeable with the DESI-2 inputs. `z_beta_bias.txt` is a historical
McDonald (2003) comparison table, not the current bias configuration.
