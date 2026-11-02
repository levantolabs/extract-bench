# Extract Bench v0.1 board (eval split)

`extract-bench report` generates this file. Each cell is field accuracy in percent under rule (a), on headline fields. A ✱ means the 95% bootstrap interval of the difference from the reference excludes 0. The means leave out the receipt sets.

## headline

| Dataset | Items | Fields | Sage 2 | Clairic | Δ vs Sage 2 [95% CI] |
|---|---:|---:|---:|---:|---|
| fatura | 62 | 612 | **90.7** | 87.6 | Clairic -3.1 [-5.1, -1.3] ✱ |
| omniai | 95 | 1,556 | **73.5** | 69.7 | Clairic -3.8 [-6.6, -1.0] ✱ |
| docvqa | 145 | 145 | 73.0 | **75.0** | Clairic +2.0 [-3.9, +8.3] |
| vrdu | 131 | 972 | **67.0** | 60.0 | Clairic -7.0 [-10.3, -3.7] ✱ |
| realdoc | 71 | 218 | **41.3** | 39.9 | Clairic -1.4 [-7.8, +6.0] |
| kleister_nda | 31 | 93 | **94.6** | **94.6** | Clairic +0.0 [-5.4, +5.4] |
| kleister_charity | 44 | 352 | 82.4 | **84.9** | Clairic +2.6 [-0.3, +5.7] |
| emails | 45 | 45 | **80.0** | 66.7 | Clairic -13.3 [-33.3, +6.7] |
| tatqa | 24 | 24 | **83.5** | 79.4 | Clairic -4.1 [-19.4, +11.2] |
| squad2 | 57 | 57 | 92.1 | **94.7** | Clairic +2.7 [-3.1, +9.2] |
| sroie (receipts) | 143 | 572 | 59.3 | **60.7** | Clairic +1.4 [-0.9, +3.7] |
| cord (receipts) | 79 | 947 | **53.0** | 50.9 | Clairic -2.1 [-4.5, +0.5] |
| **Mean of 10 sets** | | | **77.8** | **75.3** | Clairic -2.5 [-5.4, +0.2] |
| "Not in the document" correct (7 sets) | | | 72.2 | 59.0 | |

## text

| Dataset | Items | Fields | Sage 2 | Clairic | NuExtract3 | Schematron-3B | GLiNER2.5 | Δ vs Sage 2 [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| fatura | 62 | 612 | **90.7** | 87.6 | 88.9 | 85.0 | 68.0 | Clairic -3.1 [-5.1, -1.3] ✱; NuExtract3 -1.8 [-3.3, -0.3] ✱; Schematron-3B -5.7 [-7.7, -4.0] ✱; GLiNER2.5 -22.7 [-25.8, -19.7] ✱ |
| omniai | 95 | 1,556 | 73.5 | 69.7 | **83.9** | 55.3 | 38.8 | Clairic -3.8 [-6.6, -1.0] ✱; NuExtract3 +10.4 [+5.7, +14.3] ✱; Schematron-3B -18.2 [-23.0, -13.4] ✱; GLiNER2.5 -34.7 [-38.5, -30.9] ✱ |
| docvqa | 145 | 145 | 73.0 | **75.0** | 70.8 | 50.1 | 11.4 | Clairic +2.0 [-3.9, +8.3]; NuExtract3 -2.2 [-8.4, +4.1]; Schematron-3B -22.9 [-30.9, -14.4] ✱; GLiNER2.5 -61.5 [-69.5, -53.3] ✱ |
| vrdu | 131 | 972 | 67.0 | 60.0 | **68.7** | 56.1 | 34.1 | Clairic -7.0 [-10.3, -3.7] ✱; NuExtract3 +1.7 [-2.2, +5.7]; Schematron-3B -10.8 [-15.8, -5.7] ✱; GLiNER2.5 -32.9 [-37.4, -28.6] ✱ |
| realdoc | 71 | 218 | 41.3 | 39.9 | **54.6** | 32.6 | 18.3 | Clairic -1.4 [-7.8, +6.0]; NuExtract3 +13.3 [+4.8, +21.7] ✱; Schematron-3B -8.7 [-17.6, +0.8]; GLiNER2.5 -22.9 [-31.0, -14.7] ✱ |
| kleister_nda | 31 | 93 | **94.6** | **94.6** | 81.7 | 80.6 | 45.2 | Clairic +0.0 [-5.4, +5.4]; NuExtract3 -12.9 [-21.5, -5.4] ✱; Schematron-3B -14.0 [-21.5, -7.5] ✱; GLiNER2.5 -49.5 [-58.1, -40.9] ✱ |
| kleister_charity | 44 | 352 | 82.4 | **84.9** | 76.4 | 64.2 | 38.4 | Clairic +2.6 [-0.3, +5.7]; NuExtract3 -6.0 [-9.4, -2.6] ✱; Schematron-3B -18.2 [-22.7, -13.6] ✱; GLiNER2.5 -44.0 [-48.9, -39.5] ✱ |
| emails | 45 | 45 | 80.0 | 66.7 | 88.9 | **95.6** | 46.7 | Clairic -13.3 [-33.3, +6.7]; NuExtract3 +8.9 [-6.7, +24.4]; Schematron-3B +15.6 [+4.4, +28.9] ✱; GLiNER2.5 -33.3 [-53.3, -13.3] ✱ |
| tatqa | 24 | 24 | 83.5 | 79.4 | **88.5** | 38.3 | 3.6 | Clairic -4.1 [-19.4, +11.2]; NuExtract3 +5.0 [-0.3, +11.5]; Schematron-3B -45.2 [-66.7, -23.7] ✱; GLiNER2.5 -79.9 [-93.8, -64.7] ✱ |
| squad2 | 57 | 57 | 92.1 | **94.7** | 59.0 | 41.1 | 52.6 | Clairic +2.7 [-3.1, +9.2]; NuExtract3 -33.1 [-45.8, -19.4] ✱; Schematron-3B -50.9 [-62.8, -37.5] ✱; GLiNER2.5 -39.4 [-53.3, -24.8] ✱ |
| sroie (receipts) | 143 | 572 | 59.3 | 60.7 | **61.4** | 55.2 | 29.5 | Clairic +1.4 [-0.9, +3.7]; NuExtract3 +2.1 [-0.2, +4.4]; Schematron-3B -4.0 [-7.0, -0.9] ✱; GLiNER2.5 -29.7 [-33.4, -25.9] ✱ |
| cord (receipts) | 79 | 947 | 53.0 | 50.9 | **53.9** | 48.4 | 44.0 | Clairic -2.1 [-4.5, +0.5]; NuExtract3 +0.9 [-1.5, +3.2]; Schematron-3B -4.6 [-7.5, -1.8] ✱; GLiNER2.5 -9.1 [-11.9, -6.3] ✱ |
| **Mean of 10 sets** | | | **77.8** | **75.3** | **76.1** | **59.9** | **35.7** | Clairic -2.5 [-5.4, +0.2]; NuExtract3 -1.7 [-4.3, +0.9]; Schematron-3B -17.9 [-21.1, -14.8] ✱; GLiNER2.5 -42.1 [-45.4, -38.9] ✱ |
| "Not in the document" correct (7 sets) | | | 72.2 | 59.0 | 57.0 | 54.3 | 44.1 | |

## image

| Dataset | Items | Fields | Sage 2 | Clairic + image | Δ vs Sage 2 [95% CI] |
|---|---:|---:|---:|---:|---|
| fatura | 62 | 612 | 90.7 | **92.6** | Clairic + image +2.0 [-0.7, +4.5] |
| omniai | 95 | 1,556 | **73.5** | 72.2 | Clairic + image -1.3 [-4.1, +1.3] |
| docvqa | 145 | 145 | 73.0 | **87.8** | Clairic + image +14.8 [+9.3, +21.0] ✱ |
| vrdu | 131 | 972 | **67.0** | 63.4 | Clairic + image -3.6 [-7.5, +0.1] |
| realdoc | 71 | 218 | 41.3 | **62.4** | Clairic + image +21.1 [+12.7, +30.5] ✱ |
| sroie (receipts) | 143 | 572 | 59.3 | **68.5** | Clairic + image +9.3 [+5.8, +12.8] ✱ |
| cord (receipts) | 79 | 947 | 53.0 | **82.4** | Clairic + image +29.4 [+25.5, +33.3] ✱ |
| **Mean of 5 sets** | | | **69.1** | **75.7** | Clairic + image +6.6 [+4.1, +9.1] ✱ |
| "Not in the document" correct (4 sets) | | | 70.9 | 52.4 | |

